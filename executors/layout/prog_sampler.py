import random
import torch
import math
import numpy as np
from copy import deepcopy

def round_val(val, tf_vals, f_vals):
    ind = (tf_vals - val).abs().argmin()
    return f_vals[ind]

def comb_sub_progs(sub_shapes, add_end=False):
    end = []
    if add_end:
        end += ['END']
    return ['START'] + _comb_sub_progs(sub_shapes) + end

def _comb_sub_progs(ss):
    if len(ss) == 1:
        return ss[0]
    
    return ['union'] + ss[0]  + _comb_sub_progs(ss[1:])    


def reorder_prims(sampled_prims, ex):

    # list of prims
    canvas = []

    q = [(sp, False) for sp in sampled_prims]
    
    while len(q) > 0:
        sp, pa = q.pop(0)
        if len(canvas) == 0:
            canvas.append(sp)
            continue
        
        bad_colors = set()
        covered = []

        for i, cp in enumerate(canvas):
            # sI is percent of newly added that is covered
            # cI is precent of previous added that is covered

            R = calc_overlap(sp, cp)

            sI, cI, sA, cA = R
            if sI > .65:
                bad_colors.add(cp.color_params)

            if cA < sA and cI > .75:
                covered.append(i)

        if sp.color_params not in bad_colors and len(covered) == 0:
            canvas.append(sp)
            continue

        good_colors = [
            c for c in ('red', 'green', 'blue', 'grey') if c not in bad_colors
        ]

        if sp.color_params in bad_colors:
            if len(good_colors) > 0:                
                sp.color_params = random.choice(good_colors)
            else:
                # need to fix
                continue

        if len(covered) == 0:
            canvas.append(sp)
            continue

        if pa:
            continue

        rmvd = [(c,True) for i,c in enumerate(canvas) if i in covered]
        canvas = [c for i,c in enumerate(canvas) if i not in covered]
        canvas.append(sp)
        q += rmvd

    return canvas


class PrimGroup:
    def __init__(self, ex, local=None):

        self.top_info = []                
        self.ex = ex
        if local is not None:
            return self.in_context_init(ex, local)
            
        if random.random() < 0.75:
            self.color_params = random.choice(ex.D_COLORS)
        else:
            self.color_params = 'grey'
        
        r = random.random()
        if r < 0.85:
            self.num_prims = 1
        elif r < 0.95:
            self.num_prims = 2
        else:
            self.num_prims = 3
            
        self.prims = []
        # left, right, bot, top extents
        self.bboxes = torch.ones(self.num_prims, 4).float() * 1.1

        for _ in range(self.num_prims):

            # pbbox = (4)
            prim, pbbox = sample_move_scale_prim(ex)

            cbboxes = torch.stack((pbbox.unsqueeze(0).repeat(self.num_prims, 1), self.bboxes),dim=1)
            # num_prims x 2 x 4

            lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
            rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
            intersect = lhs * rhs

            min_area = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2])).min(dim=1).values
            
            max_perc_intersect = (intersect / (min_area+1e-8)).max()

            if max_perc_intersect > 0.5:
                continue
                        
            self.bboxes[len(self.prims)] = pbbox
            self.prims.append(prim)


    def in_context_init(self, ex, local):

        self.color_params = 'grey'
        if 'color' not in local:        
            if random.random() < 0.75:
                self.color_params = random.choice(ex.D_COLORS)

        r = random.random()
        if r < 0.85:
            self.num_prims = 1
        elif r < 0.95:
            self.num_prims = 2
        else:
            self.num_prims = 3
            
        self.prims = []
        # left, right, bot, top extents
        self.bboxes = torch.ones(self.num_prims, 4).float() * 1.1

        up_inds = [i for i,t in enumerate(local) if t == 'union' or 'sym' in t]
        if len(up_inds) > 0:
            last_up_ind = max(up_inds)        
            after_up_ind = local[last_up_ind:]
        else:
            after_up_ind = local

        skip_move = False
        skip_scale = False
        
        if 'move' in after_up_ind:
            skip_move = True

        if 'scale' in after_up_ind:
            skip_scale = True
            
        for _ in range(self.num_prims):
            # pbbox = (4)
            prim, pbbox = sample_move_scale_prim(ex)

            if skip_scale or skip_move:
                if skip_scale:
                    prim.scale_params = (1., 1.)
                if skip_move:
                    prim.move_params = (0., 0.)

                pbbox = prim.get_bbox()
            
            cbboxes = torch.stack((pbbox.unsqueeze(0).repeat(self.num_prims, 1), self.bboxes),dim=1)
            # num_prims x 2 x 4

            lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
            rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
            intersect = lhs * rhs

            min_area = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2])).min(dim=1).values
            
            max_perc_intersect = (intersect / (min_area+1e-8)).max()

            if max_perc_intersect > 0.5:
                continue
                        
            self.bboxes[len(self.prims)] = pbbox
            self.prims.append(prim)
                           
    def sample_sym(self, ex):
                        
        sym_opts = self.check_sym_opts()

        if len(sym_opts) == 0:            
            return
        
        sym_choice = random.choice(sym_opts)

        if 'ref' in sym_choice:
            self.add_ref_sym(sym_choice)
            if len(self.top_info) > 0:
                self.sample_extra_move(ex)
        elif 'rot' in sym_choice:
            self.add_rot_sym(sym_choice)
            if len(self.top_info) > 0:
                self.sample_extra_move(ex)
        elif 'trans' in sym_choice:
            self.add_trans_sym(sym_choice, ex)


    def check_sym_opts(self):
        left_extent = self.bboxes[:,0].min().item()
        right_extent = self.bboxes[:,1].max().item()
        bot_extent = self.bboxes[:,2].min().item()
        top_extent = self.bboxes[:,3].max().item()

        so = []

        if left_extent > 0. or right_extent < 0.:
            so.append('ref_AX')

        if bot_extent > 0. or top_extent < 0.:
            so.append('ref_AY')

        if len(so) == 2:
            so.append('rot')
        elif 'ref_AX' in so and\
             max(abs(right_extent), (left_extent)) + ((top_extent-bot_extent)/2.) < 1.0:
            so.append('rot')
        elif 'ref_AY' in so and\
             max(abs(bot_extent), (top_extent)) + ((right_extent-left_extent)/2.) < 1.0:
            so.append('rot')

        to = set()
        if left_extent > 0:
            to.add('left')

        if right_extent < 0:
            to.add('right')

        if bot_extent > 0:
            to.add('bot')

        if top_extent < 0:
            to.add('top')

        if len(to) > 0:
            so.append(f'trans_{"+".join(list(to))}')

        if 'symReflect' in self.top_info:
            so = [s for s in so if 'ref' not in s]

        if 'symRotate' in self.top_info:
            so = [s for s in so if 'rot' not in s]

        if 'symTranslate' in self.top_info:
            so = [s for s in so if 'trans' not in s]

        self.last_sym_check = so
        return so
        

    def add_ref_sym(self, sc):
        if 'AX' in sc:
            self.top_info = ['symReflect', 'AX'] + self.top_info

            rbboxes = self.bboxes.clone()

            rbboxes[:,0] = -1 * self.bboxes[:,1]
            rbboxes[:,1] = -1 * self.bboxes[:,0]

            self.bboxes = torch.cat((
                self.bboxes,
                rbboxes
            ),dim=0)
            
        elif 'AY' in sc:
            self.top_info = ['symReflect', 'AY'] + self.top_info

            rbboxes = self.bboxes.clone()

            rbboxes[:,2] = -1 * self.bboxes[:,3]
            rbboxes[:,3] = -1 * self.bboxes[:,2]

            self.bboxes = torch.cat((
                self.bboxes,
                rbboxes
            ),dim=0)
            
        else:
            assert False, f' bad {sc}'
        

    def add_rot_sym(self, sc):    

        K = random.randint(2,6)
        new_boxes = []
        for bbox in self.bboxes:

            X = (bbox[1] + bbox[0]) / 2.
            Y = (bbox[3] + bbox[2]) / 2.
            hW = (bbox[1] - bbox[0]) / 2.
            hH = (bbox[3] - bbox[2]) / 2.

            xv = np.array([1.0, 0.0])
            rv = np.array([X, Y])

            rv_norm = np.linalg.norm(rv)

            if rv_norm == 0:
                return

            dp = np.arccos(np.dot(xv, rv / rv_norm))

            if Y < 0.:
                dp = -1 * dp
                
            for k in range(1, K+1):
                
                perc = (k * 1.) / (K+1)
                
                incr = perc * (np.pi * 2.)
                
                nv = dp + incr
                
                nX = np.cos(nv) * rv_norm
                nY = np.sin(nv) * rv_norm

                nbbox = torch.tensor([
                    nX - hW,
                    nX + hW,
                    nY - hH,
                    nY + hH
                ])

                if nbbox.abs().max() > 1.0:
                    return

                new_boxes.append(nbbox)

        self.bboxes = torch.cat((
            self.bboxes,
            torch.stack(new_boxes,dim=0)
        ),dim=0)
        
        self.top_info = ['symRotate', str(K)] + self.top_info

    def parse_trans_dir(self, dr):
        left_extent = self.bboxes[:,0].min()
        right_extent = self.bboxes[:,1].max()
        bot_extent = self.bboxes[:,2].min()
        top_extent = self.bboxes[:,3].max()
        
        if dr == 'right':
            ind = 0
            min_incr = right_extent - left_extent
            max_ext = 1 - right_extent
        elif dr == 'left':
            ind = 0
            min_incr = left_extent - right_extent
            max_ext = -1 - left_extent
        elif dr == 'top':
            ind = 1
            min_incr = top_extent - bot_extent
            max_ext = 1 - top_extent
        elif dr == 'bot':
            ind = 1
            min_incr = bot_extent - top_extent
            max_ext = -1 - bot_extent
        else:
            assert False, f'bad dr {dr}'

        return ind, min_incr.item(), max_ext.item()
            
    def add_trans_sym(self, sc, ex):        

        dos = sc.split('_')[1].split('+')

        random.shuffle(dos)

        main_dir = dos.pop(0)

        if len(dos) > 0:
            sec_dir = dos.pop(0)
        else:
            sec_dir = None

        main_ind, main_min_incr, main_max = self.parse_trans_dir(main_dir)
        if sec_dir is not None:
            sec_ind, _, sec_max = self.parse_trans_dir(sec_dir)
        else:
            sec_ind = int(not bool(main_ind))
            sec_max = 0.
            
        if abs(main_min_incr) > abs(main_max):
            return

        max_K = min(abs(main_max) // abs(main_min_incr), 3)

        K = random.randint(1, max_K)
        main_max_incr = main_max / K

        r = random.random()
        
        main_incr = (main_min_incr * r) + ((1-r) * main_max_incr)

        if random.random() < 0.5:
            sec_incr = 0.
        else:
            q = random.random()
            sec_incr = q * (sec_max / K)

        params = [None, None]
        params[main_ind] = main_incr * K
        params[sec_ind] = sec_incr * K

        assert None not in params

        params = [round_val(v, ex.TD_FLTS, ex.D_FLTS) for v in params]
                
        ## update bboxes

        new_boxes = []

        dX = float(params[0])
        dY = float(params[1])
        
        for bbox in self.bboxes:

            X = (bbox[1] + bbox[0]) / 2.
            Y = (bbox[3] + bbox[2]) / 2.
            hW = (bbox[1] - bbox[0]) / 2.
            hH = (bbox[3] - bbox[2]) / 2.

            for k in range(1, K+1):
                perc = (k * 1.) / K

                nX = X + (perc * dX)
                nY = Y + (perc * dY)

                nbbox = torch.tensor([
                    nX - hW,
                    nX + hW,
                    nY - hH,
                    nY + hH
                ])

                if nbbox.abs().max() > 1.0:
                    return

                new_boxes.append(nbbox)
                
        self.bboxes = torch.cat((
            self.bboxes,
            torch.stack(new_boxes,dim=0)
        ),dim=0)

        sym_tokens = self.ex.make_symtranslate_tokens(params + [K])
        
        self.top_info = sym_tokens + self.top_info
        
    def sample_extra_move(self, ex):

        if random.random() > 0.35:
            return
        
        left_extent = self.bboxes[:,0].min()
        right_extent = self.bboxes[:,1].max()
        bot_extent = self.bboxes[:,2].min()
        top_extent = self.bboxes[:,3].max()

        max_left = -1 - left_extent
        max_right = 1 - right_extent
        max_bot = -1 - bot_extent
        max_top = 1 - top_extent

        if max([
            abs(max_left),
            abs(max_right),
            abs(max_bot),
            abs(max_top)
        ]) < ex.G_MIN_SCALE:
            return
        
        move_tokens = []

        for _ in range(5):
            xr = random.random()
            yr = random.random()
            
            x_move = max_left * xr + (max_right * (1-xr))
            y_move = max_bot * yr + (max_top * (1-yr))
            
            params = tuple([round_val(v, ex.TD_FLTS, ex.D_FLTS) for v in [x_move, y_move]])
            
            move_tokens = self.ex.make_move_tokens(params)
            
            if len(move_tokens) > 0:
                break
                        
        self.top_info = move_tokens + self.top_info
                    
    def get_prog(self):
            
        tokens = deepcopy(self.top_info)

        if self.color_params != 'grey':
            tokens += ['color', self.color_params]
            
        sub_progs = []
        for prim in self.prims:
            sub_progs.append(prim.get_tokens())

        while len(sub_progs) > 0:
            sp = sub_progs.pop(0)

            if len(sub_progs) == 0:
                tokens += sp
            else:
                tokens += ['union'] + sp

        return tokens

# Sample a prim at random
SYM_CHANCE = 0.35
def sample_prim(ex):
    prim = PrimGroup(ex)

    # first sample sym
    if random.random() < SYM_CHANCE:
        prim.sample_sym(ex)

    if len(prim.top_info) > 0:
        # sometimes try to sample another sym
        if random.random() < SYM_CHANCE:
            prim.sample_sym(ex)

    return prim


def calc_overlap(AB, BB):

    A = AB.bboxes
    B = BB.bboxes

    EA = A.repeat(B.shape[0],1)
    EB = B.unsqueeze(1).repeat(1, A.shape[0],1).view(-1,4)

    cbboxes = torch.stack((EA, EB),dim=1)
    
    lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
    rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
    intersect = lhs * rhs

    areas = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2]))

    AA = areas[:,0]
    BA = areas[:,1]

    AAS = AA.sum() / B.shape[0]
    BAS = BA.sum() / A.shape[0]
    
    return (intersect / AA).max().item(), (intersect /BA).max().item(), AAS, BAS


def sample_move_scale_prim(ex):

    width_val = multi_norm_sample(
        ex,
        (
            (0.4, (1.2, 0.6)),
            (0.6, (0.25, 0.2))
        ),
        ex.G_MIN_SCALE,
        1.9,
    )

    height_val = multi_norm_sample(
        ex,
        (
            (0.4, (1.2, 0.6)),
            (0.6, (0.25, 0.2))
        ),
        ex.G_MIN_SCALE,
        1.9,
    )

    xpos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        -1. + (width_val * 0.5),
        1. - (width_val * 0.5),
    )

    ypos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        -1. + (height_val * 0.5),
        1. - (height_val * 0.5),
    )

    prim = Prim(ex, width_val, height_val, xpos_val, ypos_val)

    return prim, prim.get_bbox()

class Prim:
    def __init__(self, ex, width, height, xpos, ypos):

        self.ex = ex
        self.scale_params = (width, height)
        self.move_params = (xpos, ypos)
        self.prim_params = random.choice(ex.D_PRIMS)
        
    def get_bbox(self):
        xpos, ypos = self.move_params
        width, height = self.scale_params
        return torch.tensor([
            xpos - (width*0.5),
            xpos + (width*0.5),
            ypos - (height*0.5),
            ypos + (height*0.5)
        ]).float()

    def get_tokens(self):
        tokens = []
        
        def rv(v):
            return str(round_val(v, self.ex.TD_FLTS, self.ex.D_FLTS))

        rmp = [rv(self.move_params[0]), rv(self.move_params[1])]
        if rmp[0] != '0.0' or rmp[1] != '0.0':
            tokens += ['move', rmp[0], rmp[1] ]
        else:
            pass
        
        smp = [rv(self.scale_params[0]), rv(self.scale_params[1])]

        if smp[0] != '1.0' or smp[1] != '1.0':
            tokens += ['scale', smp[0], smp[1]]
        else:
            pass
        
        tokens += ['prim', self.prim_params]

        return tokens


def multi_norm_sample(ex, dists, mi, ma):
    v = None

    if mi == ma:
        return mi
    
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
            val = round_val(v, ex.TD_FLTS, ex.D_FLTS)
            if val >= mi and v <= ma:
                return val

    return mi


def simplify(orig_sub_progs):

    R = []

    for osp in orig_sub_progs:
        while osp[0] == 'union':
            ci = osp.index('prim')+2
            ssp = osp[1:ci]
            R.append(ssp)
            osp = osp[ci:]

        R.append(osp)
            
    return R

def sample_prog(ex, num_sub_progs, add_end=False):

    sampled_prims = [sample_prim(ex) for _ in range(num_sub_progs)]
        
    orig_prog = comb_sub_progs([prim.get_prog() for prim in sampled_prims])
                        
    ordered_prims = reorder_prims(sampled_prims, ex)

    if len(ordered_prims) < len(sampled_prims):        
        valid = ex.check_valid_tokens(orig_prog)
        if valid:
            ordered_prims = sampled_prims
            
    sub_progs = []
    for oprim in ordered_prims:
        sub_progs.append(oprim.get_prog())

    simple_sub_progs = simplify(sub_progs)
        
    comb_prog = comb_sub_progs(simple_sub_progs, add_end)
    
    return comb_prog, simple_sub_progs
