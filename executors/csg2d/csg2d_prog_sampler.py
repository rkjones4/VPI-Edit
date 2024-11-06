from tqdm import tqdm
import sys
import random
import torch
import math
import numpy as np
import time
from copy import deepcopy
sys.path.append('executors/common')
import lutils as lu

MAX_SIG_TRIES = 2
MAX_HIER_TRIES = 2

REF_CHANCE = 0.2
HIER_TRANS_CHANCE = 0.1

MO = 0.02

ROT_CHANCE = 0.45

NUM_PRIMS_PER_SUB_PROG_DIST = (
    [1,2,3],
    lu.norm_np([0.5,0.35,0.2])
)

OP_DIST = (
    ['union', 'inter', 'diff'],
    lu.norm_np([0.5, 0.3, 0.2])
)

CONST_DIST = (
    ['all', 'left', 'right'],
    lu.norm_np([0.5, 0.25, 0.25])
)

def round_val(val, ex, fltype):
    tf_vals = ex.TFLT_INFO[fltype]
    f_vals = ex.LFLT_INFO[fltype]
    ind = (tf_vals - val).abs().argmin().item()
    return f_vals[ind]

def sample_prim_bounds(ex, sp_bounds, rhs_bounds):
        
    w_max = sp_bounds[0]
    h_max = sp_bounds[1]

    width_val = multi_norm_sample(
        ex,
        (
            (0.2, (w_max, 0.0)),
            (0.6, (w_max * 0.75, w_max * 0.25)),
            (0.2, (w_max * 0.25, w_max * 0.25))
        ),
        MO,
        w_max,
        'scale',
    )

    height_val = multi_norm_sample(
        ex,
        (
            (0.2, (h_max, 0.0)),
            (0.6, (h_max * 0.75, h_max * 0.25)),
            (0.2, (h_max * 0.25, h_max * 0.25))
        ),
        MO,
        h_max,
        'scale',
    )

    x_min = sp_bounds[2] - sp_bounds[0] + width_val
    x_max = sp_bounds[2] + sp_bounds[0] - width_val

    y_min = sp_bounds[3] - sp_bounds[1] + height_val
    y_max = sp_bounds[3] + sp_bounds[1] - height_val
    
    if rhs_bounds is not None:
        
        x_min = max(rhs_bounds[0].item() + width_val + MO, x_min)
        x_max = min(rhs_bounds[1].item() - width_val - MO, x_max)

        y_min = max(rhs_bounds[2].item() + height_val + MO, y_min)
        y_max = min(rhs_bounds[3].item() - height_val - MO, y_max)                    

    cxpos = (x_min+x_max) / 2.
    cypos = (y_min+y_max) / 2.
    
    xpos_val = multi_norm_sample(
        ex,
        (
            (0.3, (cxpos, 0.0)),
            (0.7, (cxpos, width_val)),
        ),
        x_min,
        x_max,
        'move',
    )

    ypos_val = multi_norm_sample(
        ex,
        (
            (0.3, (cypos, 0.0)),
            (0.7, (cypos, height_val)),
        ),
        y_min,
        y_max,
        'move',
    )

    return (width_val, height_val, xpos_val, ypos_val)
    
def sample_sub_prog_bounds(ex):

    WMAX = 1.0
    HMAX = 1.0

    XMIN = -1.0
    YMIN = -1.0

    XMAX = 1.0
    YMAX = 1.0
    
    width_val = multi_norm_sample(
        ex,
        (
            (0.6, (0.6, 0.3)),
            (0.4, (0.125, 0.1))
        ),
        MO,
        WMAX,
        'scale',
    )

    height_val = multi_norm_sample(
        ex,
        (
            (0.6, (0.6, 0.3)),
            (0.4, (0.125, 0.1))
        ),
        MO,
        HMAX,
        'scale',
    )

    xpos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        XMIN + width_val ,
        XMAX - width_val,
        'move',
    )

    ypos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        YMIN + height_val,
        YMAX - height_val,
        'move',
    )

    return (width_val, height_val, xpos_val, ypos_val)

class ReflectOp:
    def __init__(self, ex, axis):
        self.ex = ex
        self.axis = axis

        self.parent = None
        self.child = None

        self.move_params = None
        self.scale_params = None
        self.rot_params = None

    def record_refs_and_hops(self, I):
        I['nhop'] += len([
            n for n in [self.move_params, self.scale_params, self.rot_params]
            if n is not None
        ]) * 1.

        I['nref'] += 1.
        
        self.child.record_refs_and_hops(I)
        
    def find_transform_places(self, ho):
        c = []
        if ho == 'move' and self.move_params is None:
            c += [self]
        elif ho == 'scale' and self.scale_params is None:
            c += [self]
        elif ho == 'rot' and self.rot_params is None:
            c += [self]
        else:
            pass
            
        return c + self.child.find_transform_places(ho)
        
    def find_child_params(self, ho):
        c = []
        if ho == 'move' and self.move_params is not None:
            c += [self.move_params]
        elif ho == 'scale' and self.scale_params is not None:
            c += [self.scale_params]
        elif ho == 'rot' and self.rot_params is not None:
            c += [self.rot_params]
        else:
            pass

        return c + self.child.find_child_params(ho)


    def push_constraints(self, ho, nparams, const_type, do_set=False):

        if ho == 'move':
            if do_set:
                assert self.move_params is None
                self.move_params = nparams
            else:
                self.move_params = resolve_params(self.ex, ho, nparams, const_type, self.move_params)

        elif ho == 'scale':
            if do_set:
                assert self.scale_params is None
                self.scale_params = nparams
            else:
                self.scale_params = resolve_params(self.ex, ho, nparams, const_type, self.scale_params)
            
        elif ho == 'rot':
            if do_set:
                assert self.rot_params is None
                self.rot_params = nparams
            else:
                self.rot_params = None
            
        else:
            assert False

        self.child.push_constraints(ho, nparams, const_type)
            
    def find_ref_places(self):
        return [self] + self.child.find_ref_places()
        
    def get_tokens(self):

        tokens = []

        if self.move_params is not None:
            rmp = [
                self.ex.F2T('move', self.move_params[0]),
                self.ex.F2T('move', self.move_params[1]),
            ]
            if not self.ex.is_no_op('move', self.move_params): 
                tokens += ['move', rmp[0], rmp[1] ]

        if self.rot_params is not None:
            tokens += ['rot', self.rot_params[0]]

        if self.scale_params is not None:
            smp = [
                self.ex.F2T('scale', self.scale_params[0]),
                self.ex.F2T('scale', self.scale_params[1]),
            ]        
        
            if not self.ex.is_no_op('scale', self.scale_params): 
                tokens += ['scale', smp[0], smp[1]]
                
        return tokens + ['reflect', self.axis] + self.child.get_tokens()
        
    def make_copy(self, parent):
        R = ReflectOp(self.ex, self.axis)
        R.parent = parent
        R.child = self.child.make_copy(R)
        return R

    def find_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.find_root()

def dummy_params(ho):
    if ho == 'move':
        return [0., 0.]
    elif ho == 'scale':
        return [1., 1.]
    else:
        assert False

def correct_for(ex, ho, p, n):

    if ho == 'scale':
        if p > n:
            return 1.
        else:
            v = round_val(p/n, ex, 'scale')
            return v
            
    elif ho == 'move':

        diff = p - n
        v = round_val(diff, ex, 'move')
        return v
    else:
        assert False
        
        
def resolve_params(ex, ho, nparams, const_type, prev_params):

    if prev_params is None:
        return None
    
    if const_type == 'all' or \
       nparams == prev_params or (
           const_type == 'left' and
           nparams[1] == prev_params[1]
       ) or (
           const_type == 'right' and
           nparams[0] == prev_params[0]
       ):
        return tuple(dummy_params(ho))

    D = dummy_params(ho)

    if const_type=='left':
        D[0] = correct_for(ex, ho,prev_params[0],nparams[0])
    elif const_type=='right':
        D[1] = correct_for(ex, ho,prev_params[1],nparams[1])
    else:
        assert False

    return tuple(D)
        


        
class CombOp:
    def __init__(self, ex, op, par):
        self.ex = ex
        self.op = op
        self.parent = par
        self.children = []
        # left, right, bot, top
        self.rhs_bounds = None

        self.move_params = None
        self.scale_params = None
        self.rot_params = None

    def record_refs_and_hops(self, I):
        I['nhop'] += len([
            n for n in [self.move_params, self.scale_params, self.rot_params]
            if n is not None
        ]) * 1.

        self.children[0].record_refs_and_hops(I)
        self.children[1].record_refs_and_hops(I)
        
    def find_transform_places(self, ho):
        c = []
        if ho == 'move' and self.move_params is None:
            c += [self]
        elif ho == 'scale' and self.scale_params is None:
            c += [self]
        elif ho == 'rot' and self.rot_params is None:
            c += [self]
        else:
            pass
        
        return c + self.children[0].find_transform_places(ho) + self.children[1].find_transform_places(ho)


    def find_child_params(self, ho):
        c = []
        if ho == 'move' and self.move_params is not None:
            c += [self.move_params]
        elif ho == 'scale' and self.scale_params is not None:
            c += [self.scale_params]
        elif ho == 'rot' and self.rot_params is not None:
            c += [self.rot_params]
        else:
            pass

        return c + self.children[0].find_child_params(ho) + self.children[1].find_child_params(ho)

    def push_constraints(self, ho, nparams, const_type, do_set=False):

        if ho == 'move':
            if do_set:
                assert self.move_params is None
                self.move_params = nparams
            else:
                self.move_params = resolve_params(self.ex,ho, nparams, const_type, self.move_params)

        elif ho == 'scale':
            if do_set:
                assert self.scale_params is None
                self.scale_params = nparams
            else:
                self.scale_params = resolve_params(self.ex,ho, nparams, const_type, self.scale_params)
            
        elif ho == 'rot':
            if do_set:
                assert self.rot_params is None
                self.rot_params = nparams
            else:
                self.rot_params = None
            
        else:
            assert False

        self.children[0].push_constraints(ho, nparams, const_type)
        self.children[1].push_constraints(ho, nparams, const_type)

        
    def find_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.find_root()
        
    def find_ref_places(self):
        return [self] + self.children[0].find_ref_places() + self.children[1].find_ref_places()
        
    def add_child(self, c):
        self.children.append(c)

    def get_area(self):
        if len(self.children) == 0:
            return None
        careas = [c.get_area() for c in self.children]
        return torch.cat([a for a in careas if a is not None],dim=0)
        
    def fix_bounds(self):
        careas = self.get_area()
        self.area = torch.tensor([            
            careas[:,0].min().item(),
            careas[:,1].max().item(),
            careas[:,2].min().item(),
            careas[:,3].max().item(),
        ])

    def get_tokens(self):

        tokens = []

        if self.move_params is not None:
            rmp = [
                self.ex.F2T('move', self.move_params[0]),
                self.ex.F2T('move', self.move_params[1]),
            ]
            if not self.ex.is_no_op('move', self.move_params): 
                tokens += ['move', rmp[0], rmp[1] ]

        if self.rot_params is not None:
            tokens += ['rot', self.rot_params[0]]

        if self.scale_params is not None:
            smp = [
                self.ex.F2T('scale', self.scale_params[0]),
                self.ex.F2T('scale', self.scale_params[1]),
            ]        
        
            if not self.ex.is_no_op('scale', self.scale_params): 
                tokens += ['scale', smp[0], smp[1]]
                
        return tokens + [self.op] + self.children[0].get_tokens() + self.children[1].get_tokens()

    def make_copy(self, parent):
        C = CombOp(self.ex, self.op, parent)

        for child in self.children:
            C.children.append(child.make_copy(C))

        return C
    
class Prim:
    def __init__(self, ex, bounds, par):

        if ex is None:
            return
        
        self.ex = ex    
        
        self.parent = par
        
        if par is None:
            width, height, xpos, ypos = bounds
            self.scale_params = (width, height)
            self.move_params = (xpos, ypos)
        else:
            prim_info = sample_prim_bounds(ex, bounds, par.rhs_bounds)
            width, height, xpos, ypos = prim_info
            self.scale_params = (width, height)
            self.move_params = (xpos, ypos)

        if random.random() < ROT_CHANCE:
            self.rot_params = (random.choice(list(self.ex.FLT_INFO['rot'].keys())),)
        else:
            self.rot_params = None

            
        self.prim_params = random.choice(ex.D_PRIMS)

        self.area = self.get_bbox()

    def record_refs_and_hops(self, I):
        pass
        
    def find_transform_places(self, ho):
        return []

    def find_child_params(self, ho):

        if ho == 'move':
            return [self.move_params]
        elif ho == 'scale':
            return [self.scale_params]
        elif ho == 'rot':
            if self.rot_params is not None: 
                return [self.rot_params]
            else:
                return []
        else:
            assert False
        

    def push_constraints(self, ho, nparams, const_type):
        if ho == 'move':
            self.move_params = resolve_params(self.ex,ho, nparams, const_type, self.move_params)
        elif ho == 'scale':
            self.scale_params = resolve_params(self.ex,ho, nparams, const_type, self.scale_params)            
        elif ho == 'rot':
            self.rot_params = None            
        else:
            assert False
        
    def find_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.find_root()
        
    def make_copy(self, parent=None):
        C = Prim(None,None,None)

        C.parent = parent
        
        C.ex = self.ex        
        C.scale_params = self.scale_params
        C.move_params = self.move_params
        C.rot_params = self.rot_params
        C.prim_params = self.prim_params

        return C
        
    def find_ref_places(self):
        return [self]
        
    def get_area(self):
        return self.area.view(1,4)
    
    def get_bbox(self):
        xpos, ypos = self.move_params
        width, height = self.scale_params
        return torch.tensor([
            xpos - width,
            xpos + width,
            ypos - height,
            ypos + height
        ]).float()

    def get_tokens(self):
        tokens = []
                        
        rmp = [
            self.ex.F2T('move', self.move_params[0]),
            self.ex.F2T('move', self.move_params[1]),
        ]
        if not self.ex.is_no_op('move', self.move_params): 
            tokens += ['move', rmp[0], rmp[1] ]

        if self.rot_params is not None:
            tokens += ['rot', self.rot_params[0]]
            
        smp = [
            self.ex.F2T('scale', self.scale_params[0]),
            self.ex.F2T('scale', self.scale_params[1]),
        ]        
        
        if not self.ex.is_no_op('scale', self.scale_params): 
            tokens += ['scale', smp[0], smp[1]]
        else:
            pass
        
        tokens += ['prim', self.prim_params]

        return tokens

def multi_norm_sample(ex, dists, mi, ma, fltype):
    v = None

    if mi == ma:
        return round_val(mi, ex, fltype)
    
    for _ in range(10):
        
        r = random.random()
        if len(dists) == 2:
            p1, (m1, s1) = dists[0]
            _, (m2, s2) = dists[1]

            if r < p1:
                mean = m1
                std = s1
            else:
                mean = m2
                std = s2

        elif len(dists) == 3:
            p1, (m1, s1) = dists[0]
            p2, (m2, s2) = dists[1]
            _, (m3, s3) = dists[2]

            if r < p1:
                mean = m1
                std = s1
            elif r < p1 + p2:
                mean = m2
                std = s2
            else:
                mean = m3
                std = s3
                
        else:
            assert False    
                

        v = mean + (np.random.randn() * std)
        if v >= mi and v <= ma:
            val = round_val(v, ex, fltype)
            if val >= mi and v <= ma:
                return val

    return round_val(mi, ex, fltype)




def convert_sig_to_prims(ex, sig, bounds):

    if len(sig) == 1:
        assert sig[0] == 'prim'
        sig.pop(0)
        return Prim(ex, bounds, None)

    node = CombOp(ex, sig.pop(0), par=None)
    q = [node]
    
    while len(sig) > 0:    
        nt = sig.pop(0)

        last = q.pop(-1)
        
        if nt == 'prim':
            last.add_child(
                Prim(ex, bounds, last)
            )
            if len(last.children) == 1:
                if last.op != 'union':
                    last.rhs_bounds = last.children[0].area
                q.append(last)
            else:
                assert len(last.children) == 2                
                if last.parent is not None and last.parent.op != 'union':
                    last.fix_bounds()
                    last.parent.rhs_bounds = last.area
        else:
            assert nt in ('union', 'inter', 'diff')
            comb = CombOp(ex, nt, par=last)

            if last.rhs_bounds is not None:
                comb.rhs_bounds = last.rhs_bounds
            
            last.add_child(comb)
            if len(last.children) == 1:
                q.append(last)
            else:
                assert len(last.children) == 2
                
            q.append(comb)
                                
    assert len(node.children) == 2

    return node
    
        
def add_reflect_op(cnode):
    axis = random.choice(cnode.ex.D_AXIS)
    rnode = ReflectOp(cnode.ex, axis)
    rnode.child = cnode        
    rnode.parent = cnode.parent

    cnode.parent = rnode
    b2b = False
    if isinstance(cnode, ReflectOp):
        b2b = True
        while rnode.axis == cnode.axis:
            rnode.axis = random.choice(cnode.ex.D_AXIS)
            
    if rnode.parent is not None:
        parent = rnode.parent
        if isinstance(parent, ReflectOp):
            if b2b:
                assert False, 'too many refs'
            while parent.axis == rnode.axis:
                rnode.axis = random.choice(cnode.ex.D_AXIS)

            parent.child = rnode

        else:
            assert isinstance(parent, CombOp)
            if parent.children[0] == cnode:
                parent.children[0] = rnode
            else:
                assert parent.children[1] == cnode
                parent.children[1] = rnode

    return rnode.find_root()
                
class SubProg:
    def __init__(self, ex, sig):

        if ex is None:
            return
        
        bounds = sample_sub_prog_bounds(ex)
        self.root = convert_sig_to_prims(ex, sig, bounds)        

        self.need_hier_op = False

        if isinstance(self.root, CombOp) and self.root.op == 'union':
            self.need_hier_op = True

    def make_copy(self):

        C = SubProg(None, None)
        C.root = self.root.make_copy(None)
        C.need_hier_op = self.need_hier_op

        return C
        
    def check_valid(self, ex):
        
        self.local_tokens = self.root.get_tokens()

        sp_expr = ' '.join(['START','POS'] + self.local_tokens + ['END'])
        
        img, soft_error = ex.run(sp_expr, cse=True)    

        self.expr = sp_expr
        self.img = img.flatten()
        
        uc = sp_expr.count('union')
        dc = sp_expr.count('diff')
        ic = sp_expr.count('inter')

        info = {
            'nsp': uc + dc + ic,
            'uc': uc,
            'dc': dc,
            'ic': ic,
            'nrot': sp_expr.count('rot'),
            'nref': 0.,
            'nhop': 0.,
        }
        
        return not soft_error, info    

    def sample_hier_ops(self):

        hier_ops = []
        
        if random.random() < REF_CHANCE:
            hier_ops.append('ref')
            if random.random() < REF_CHANCE:
                hier_ops.append('ref')

        if random.random() < HIER_TRANS_CHANCE:
            hier_ops.append('move')
            if random.random() < HIER_TRANS_CHANCE:
                hier_ops.append('move')

        if random.random() < HIER_TRANS_CHANCE:
            hier_ops.append('scale')
            if random.random() < HIER_TRANS_CHANCE:
                hier_ops.append('scale')

        if random.random() < HIER_TRANS_CHANCE:
            hier_ops.append('rot')
            if random.random() < HIER_TRANS_CHANCE:
                hier_ops.append('scale')

        if self.need_hier_op and len(hier_ops) == 0:
            hier_ops = [random.choice(['ref', 'move', 'scale', 'rot'])]

        return hier_ops
            

    def add_hier_ops(self, hier_ops):
        for _ in range(hier_ops.count('ref')):
            ref_places = self.root.find_ref_places()
            rp = random.choice(ref_places)
            self.root = add_reflect_op(rp)

        while len(hier_ops) > 0:
            ho = hier_ops.pop(0)

            if ho in ('move', 'scale', 'rot'):
                places = self.root.find_transform_places(ho)
                if len(places) == 0:
                    continue

                tp = random.choice(places)

                child_params = tp.find_child_params(ho)

                if len(child_params) == 0:
                    continue
                
                nparams = random.choice(child_params)
                
                const_type = sample_const_type(ho)
                
                tp.push_constraints(ho, nparams, const_type, do_set=True)

                
            elif ho == 'ref':
                continue
            
            else:
                assert False, f'bad hier op {ho}'
                
    

def sample_const_type(ho):
    if ho == 'rot':
        return 'all'

    assert ho in ('scale', 'move')

    return lu.sample_dist(
        CONST_DIST
    )
    
                
def sample_op():
    return lu.sample_dist(
        OP_DIST
    )
    
def sample_sig():
    num_prims = lu.sample_dist(
        NUM_PRIMS_PER_SUB_PROG_DIST        
    )
    if num_prims == 1:
        return ['prim']

    elif num_prims == 2:
        op = sample_op()
        return [op, 'prim', 'prim']

    elif num_prims == 3:
        op1 = sample_op()
        op2 = sample_op()

        if random.random() < 0.5:
            return [op1, op2, 'prim', 'prim', 'prim']
        else:
            return [op1, 'prim', op2, 'prim', 'prim']
    else:
        assert False
        
    



def sample_sub_prog(ex):

    sig = sample_sig()
    
    for _ in range(MAX_SIG_TRIES):
    
        sub_prog = SubProg(ex, deepcopy(sig))

        is_valid, check_info = sub_prog.check_valid(ex)

        if not is_valid:
            sub_prog = None
            continue
        
        sub_prog.valid = is_valid
        sub_prog.check_info = check_info
                               
        hier_ops = sub_prog.sample_hier_ops()
        
        if len(hier_ops) > 0:            
            for _ in range(2):

                hier_prog = sub_prog.make_copy()
                
                hier_prog.add_hier_ops(hier_ops)
                
                is_valid, check_info = hier_prog.check_valid(ex)
                
                if is_valid:
                    hier_prog.root.record_refs_and_hops(check_info)
                    
                    hier_prog.valid = is_valid
                    hier_prog.check_info = check_info
                    
                    return hier_prog

        if sub_prog.need_hier_op:
            sub_prog = None
            continue
                        
        return sub_prog
        
    return None


def filter_valid(ex, sub_progs, prev_occ):

    valid_sub_progs = [sp for sp in sub_progs if sp is not None]

    while len(valid_sub_progs) > 0:
    
        stacked_canvas = torch.stack(
            [sp.img for sp in valid_sub_progs], dim=1
        )        
        
        overlap_occ = torch.relu((stacked_canvas.sum(dim=1) - 1.).unsqueeze(-1))

        if prev_occ is not None:            
            uniq_occ = torch.relu(
                (stacked_canvas * prev_occ.unsqueeze(-1)) - overlap_occ
            ).sum(dim=0)
        else:
            uniq_occ = torch.relu(stacked_canvas - overlap_occ).sum(dim=0)

        amin = uniq_occ.argmin()

        if uniq_occ[amin] >= ex.MIN_DIFF_PIXELS:
            return valid_sub_progs, stacked_canvas.max(dim=1).values

        valid_sub_progs.pop(amin.item())
        
    return [], None
        
        
        

def format_to_tokens(pos_sp, neg_sp):
    tokens = ['START']
    for psp in pos_sp:
        tokens += ['POS'] + psp.local_tokens

    for nsp in neg_sp:
        tokens += ['NEG'] + nsp.local_tokens

    tokens += ['END']

    return tokens
    


def sample_prog(ex, params):
    
    sampled_pos_sp = [
        sample_sub_prog(ex) for _ in range(params['num_pos_sub_progs'])
    ]

    valid_pos_sp, pos_occ = filter_valid(ex, sampled_pos_sp, None)

    if len(valid_pos_sp) == 0:
        return None
    
    sampled_neg_sp = [
        sample_sub_prog(ex) for _ in range(params['num_neg_sub_progs'])
    ]    

    valid_neg_sp, _ = filter_valid(ex, sampled_neg_sp, pos_occ)
        
    tokens = format_to_tokens(valid_pos_sp, valid_neg_sp)
    
    return tokens


