import time
import itertools
import matplotlib.pyplot as plt
import sys
import random
from copy import deepcopy
import numpy as np
import torch
import math

VERBOSE = False

sys.path.append('executors')
sys.path.append('executors/common')
sys.path.append('executors/layout')
sys.path.append('..')
sys.path.append('../common')
import base
import lutils as lu
device=torch.device('cuda')
import prog_sampler

LAY_CONFIG = {
    'MAX_TOKENS': 128,            
    'VDIM': 64,
    'SP_PATIENCE': 10,
    'BASE_TYPE' : 'shape',
    'DEF_PRIM_SIZE': 0.5,    
    'MIN_UNIQUE_PIXELS': 8,
    'MIN_VIS_CNT': 8,
    'OUT_THRESH': 0.1,
    'MAX_PARTS': 32,
    'FLT_NTOKENS': 66,
    'metric_name': 'color_iou',
    'NUM_SUB_PROG_DIST': (
        [1,2,3,4,5,6,7,8],
        lu.norm_np([.025,.075, .125, .175, .15, .1, .05, .025])
    ),
}

EX_PTS = None
OUT_THRESH = None
DEF_PRIM_SIZE = None
BASE_TYPE = None    

CACHE = None
CACHE_COUNT = None

# Start LANG Execution Logic

class Primitive:
    def __init__(self, ptype):

        self.ptype = ptype
        self.color = 'grey'
        
        self.W = DEF_PRIM_SIZE
        self.H = DEF_PRIM_SIZE
        self.X = 0.0
        self.Y = 0.0

        self.prim_map = {
            'square': self.get_square_pixels,
            'circle': self.get_circle_pixels,
            'triangle': self.get_triangle_pixels,
        }


            
    def has_soft_error(self):
        if self.W <= 0.0:
            return True

        if self.H <= 0.0:
            return True

        if self.X - self.W < -1.0 - OUT_THRESH:
            return True

        if self.X + self.W > 1.0 + OUT_THRESH:
            return True

        if self.Y + self.H > 1.0 + OUT_THRESH:
            return True

        if self.Y - self.H < -1.0 - OUT_THRESH:
            return True

        return False
            
        
    def copy(self):
        n = Primitive(self.ptype)
        n.color = self.color
        n.W = self.W
        n.H = self.H
        n.X = self.X
        n.Y = self.Y

        return n

    def get_pixels(self):
        return self.prim_map[self.ptype]()
    
    def get_triangle_pixels(self):
        tpts = EX_PTS - torch.tensor(
            [self.X, self.Y - self.H],
            device = device
        )
        spts = tpts / torch.tensor([self.W, self.H], device=device)

        in_pixels = (spts[:,1] >= 0.0) & (((spts[:,0].abs() * 2) + spts[:,1]) <= 2.0)

        return in_pixels
        
    def get_circle_pixels(self):
        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)
        in_pixels = spts.norm(dim=1) <= 1.0

        return in_pixels                
    
    def get_square_pixels(self):

        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)
        in_pixels = (spts <= 1.0).all(dim=1)

        return in_pixels
        
    def get_sig(self):
        return (self.ptype, self.color, self.W, self.H, self.X, self.Y)
    
class Shape:
    def __init__(self):
        self.parts = []

    def printInfo(self):
        for i, p in enumerate(self.parts):
            print(f'Prim {i} : {p.ptype} | {p.color} | {p.W} | {p.H} | {p.X} | {p.Y} ')

    def get_sig(self):
        sigs = []
        for p in self.parts:
            sigs.append(p.get_sig())

        sigs.sort()
        return tuple(sigs)


class Program:
    def __init__(self, ex):
        self.state = None
        self.ex = ex
        
    def reset(self):
        self._expr = None
        self.state = None
        self.soft_error = False
        self.cmap = {
            'grey': torch.tensor([0.5, 0.5, 0.5], device=device),
            'red': torch.tensor([1.0, 0.0, 0.0], device=device),
            'green': torch.tensor([0.0, 1.0, 0.0], device=device),
            'blue': torch.tensor([0.0, 0.0, 1.0], device=device),            
        }

    def get_state_sig(self):
        return self.state.get_sig()


    def neg_reject(self):
        with torch.no_grad():

            _CMAP = {
                'grey': 1,
                'red': 2,
                'green': 3,
                'blue': 4
            }
            flat_canvas = torch.zeros(                
                self.ex.VDIM * self.ex.VDIM,
                device=device
            ).float()
            
            exp_canvas = torch.zeros(                
                self.ex.VDIM * self.ex.VDIM,
                len(self.state.parts),
                device=device
            ).float()
            
            for i, part in enumerate(self.state.parts):
                occ_pixels = part.get_pixels()
                
                j = _CMAP[part.color]

                diff_col_pixels = (flat_canvas != j)
                
                change_pixels = occ_pixels & diff_col_pixels
                
                exp_canvas[occ_pixels, :] = 0.
                exp_canvas[change_pixels, i] = 1.0
                flat_canvas[occ_pixels] = j

            min_uniq_occ = exp_canvas.sum(dim=0).min().item()

            rej = min_uniq_occ < self.ex.MIN_VIS_CNT            
                
            return rej
            
    def has_soft_error(self):
        if self.soft_error:
            return self.soft_error

        if len(self.state.parts) == 0:
            self.soft_error = True
            return self.soft_error
        
        for part in self.state.parts:
            if part.has_soft_error():
                self.soft_error = True
                return self.soft_error

        return self.soft_error
        
    def get_pixel_value(self, color):
        return self.cmap[color]

    def ex_primitive(self, cmd):
        p = Primitive(cmd)
        s = Shape()
        s.parts.append(p)
        return s

    def ex_symReflect(self, S, A):
        new_parts = []

        for r in S.parts:
            n = r.copy()

            if A == 'AX':

                n.X = -1 * n.X

            elif A == 'AY':

                n.Y = -1 * n.Y

            new_parts.append(n)

        S.parts += new_parts

        return S
                    
        
    def ex_symTranslate(self, S, X, Y, K):
        new_parts = []

        for r in S.parts:
            for k in range(1, K+1):
                n = r.copy()

                perc = (k * 1.) / K

                n.X += perc * X
                n.Y += perc * Y

                new_parts.append(n)

        S.parts += new_parts
        return S

    def ex_symRotate(self, S, K):
        new_parts = []

        for r in S.parts:

            xv = np.array([1.0, 0.0])
            rv = np.array([r.X, r.Y])

            rv_norm = np.linalg.norm(rv)

            if rv_norm == 0:
                rv_norm += 1e-8
                self.soft_error = True        
            
            dp = np.arccos(np.dot(xv, rv / rv_norm))

            if r.Y < 0.:
                dp = -1 * dp
                
            for k in range(1, K+1):
                n = r.copy()

                perc = (k * 1.) / (K+1)
                
                incr = perc * (np.pi * 2.)
                
                nv = dp + incr
                
                n.X = np.cos(nv) * rv_norm
                n.Y = np.sin(nv) * rv_norm
                
                new_parts.append(n)
                
        S.parts += new_parts
        return S
        
    
    def ex_move(self, S, X, Y):        
        for p in S.parts:
            p.X += X
            p.Y += Y
            
        return S

    def ex_color(self, S, color):        
        for p in S.parts:
            p.color = color
            
        return S

    def ex_scale(self, S, W, H):
        
        for p in S.parts:
            p.W *= W
            p.H *= H
            
        return S

    def ex_union(self, A, B):
        s = Shape()
        s.parts += A.parts + B.parts
        return s
                            
    def _execute(self, fn, params):
        
        if fn == 'prim':
            assert len(params) == 1
            params = [self.execute(p) for p in params]
            assert params[0] in self.ex.D_PRIMS
            return self.ex_primitive(params[0])

        elif fn == 'move':            
            params = [self.execute(p) for p in params]
            assert isinstance(params[2], Shape)
            assert len(params) == 3
            return self.ex_move(params[2], float(params[0]), float(params[1]))

        elif fn == 'scale':            
            params = [self.execute(p) for p in params]
            assert isinstance(params[2], Shape)
            assert len(params) == 3
            return self.ex_scale(params[2], float(params[0]), float(params[1]))

        elif fn == 'color':
            params = [self.execute(p) for p in params]            
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            assert params[0] in self.ex.D_COLORS
            
            return self.ex_color(params[1], params[0])

        elif fn == 'START':
            if self.ex.USE_END_TOKEN:
                if len(params) == 1:
                    assert params[0] == 'END'
                    return Shape()
                
                assert len(params) == 2
                assert params[1] == 'END'                        
                prog = self.execute(params[0])
                assert isinstance(prog, Shape)            
                return prog
            else:
                assert len(params) == 1
                prog = self.execute(params[0])
                assert isinstance(prog, Shape)            
                return prog
            
        elif fn == 'union':
            params = [self.execute(p) for p in params]
            assert isinstance(params[0], Shape)
            assert isinstance(params[1], Shape)
            assert len(params) == 2
            return self.ex_union(params[0], params[1])        
 
        elif fn == 'symReflect':
            params = [self.execute(p) for p in params]
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            assert params[0] in ('AX', 'AY')
            return self.ex_symReflect(params[1], params[0])

        elif fn == 'symTranslate':
            params = [self.execute(p) for p in params]
            assert len(params) == 4
            assert isinstance(params[3], Shape)            
            return self.ex_symTranslate(
                params[3], float(params[0]), float(params[1]), int(params[2])
            )

        elif fn == 'symRotate':            
            params = [self.execute(p) for p in params]
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            return self.ex_symRotate(
                params[1], int(params[0])
            )
                        
        elif fn in self.ex.TLang.params:
            return fn
        
        else:

            try:
                float(fn)
                return fn
            
            except:
                pass    
            
            assert False, f'bad function {fn}'
    
    def execute(self, expr):
        if expr == ['START', 'END']:
            return self._execute('START', ['END'])
        
        if not isinstance(expr, list):
            try:
                float(expr)
            except:
                assert expr in self.ex.TLang.params, f'bad token {expr}'
            return self._execute(expr, [])
            
        fn = expr[0]        
        
        ipc = self.ex.TLang.get_num_inp(fn)
        
        params = []        

        cur = []

        pc = 0
        
        for c in expr[1:]:
            
            cur.append(c)

            if pc > 0:            
                pc -= 1
            
            cipc = self.ex.TLang.get_num_inp(c)

            pc += cipc
            
            if pc == 0:
                if len(cur) == 1:
                    params.append(cur[0])
                else:
                    params.append(cur)
                    
                cur = []

        if len(cur) > 0:
            params.append(cur)

        assert len(params) == ipc
        assert pc == 0

        o = self._execute(fn, params)
        if isinstance(o, Shape):
            if len(o.parts) >= self.ex.MAX_PARTS:
                assert False, 'too many parts, likely bad prog'
            
        return o

    def make_image(self):

        canvas = torch.zeros(self.ex.VDIM * self.ex.VDIM, 3, device=device).float()
        for part in self.state.parts:
            pixels = part.get_pixels()
            p_val = self.get_pixel_value(part.color)
            canvas[pixels] = p_val

        return canvas.reshape(self.ex.VDIM, self.ex.VDIM, 3)

        
    def render(self, name=None):

        with torch.no_grad():
        
            # 64 x 64 x 3 image
            img = self.make_image()

            plt.imshow(img.cpu().numpy(), origin='lower')
        
            if name is not None:
                plt.imsave(name, img.cpu().numpy(), origin='lower')
            else:
                plt.show()
                            
    def run(self, expr):
        self.reset()
        self._expr = expr

        if not self.ex.USE_END_TOKEN and expr[-1] == 'END':
            self.state = self.execute(expr[:-1])        
        else:
            self.state = self.execute(expr)        


class LayExecutor(base.BaseExecutor):
    def __init__(self, config = None):
        if config is not None:
            LAY_CONFIG.update(config)

        self.name = 'lay'
        self.prog_cls = Program
        self.base_init(LAY_CONFIG)
        
        self.make_lang()
        self.init_pts()
        self.set_globals()

    def render(self, img, name):
        assert name is not None
        plt.imsave(name, img.cpu().numpy(), origin='lower')
                
    def is_no_op(self, fn, params):
        if fn == 'move':
            for p in params:
                if float(p) != 0.:
                    return False
            return True

        if fn == 'scale':
            for p in params:
                if float(p) != 1.0:
                    return False
            return True

        return False

    def make_scale_tokens(self, params):

        assert len(params) == 2
        if params[0] == 1.0 and params[1] == 1.0:
            return []
        
        return ['scale', str(params[0]), str(params[1])]

    def make_move_tokens(self, params):

        assert len(params) == 2
        if params[0] == 0.0 and params[1] == 0.0:
            return []

        return ['move', str(params[0]), str(params[1])]

    def make_symtranslate_tokens(self, params):
        assert len(params) == 3
        if params[0] == 0.0 and params[1] == 0.0:
            return []

        return ['symTranslate', str(params[0]), str(params[1]), str(params[2])]
    
    def format_tokens_for_edit(self, input_tokens, eot, eol, eos):
        return format_tokens_for_edit(self, input_tokens, eot, eol, eos)
        
    def vis_on_axes(self, ax, img):
        ax.imshow(img, origin='lower')
        
    def execute(self, expr, vis=False):
        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN
    
        P = Program(self)
        P.run(tokens)
    
        with torch.no_grad():
            img = P.make_image()
                
        if vis:
            plt.imshow(img.cpu().numpy(), origin='lower')
            plt.show()
        
        else:
            return img
            
    def ex_prog(self, tokens):
        P = Program(self)
        P.run(tokens)
        return P
        
    def set_globals(self):
        global OUT_THRESH
        OUT_THRESH = self.OUT_THRESH
        global DEF_PRIM_SIZE
        DEF_PRIM_SIZE = self.DEF_PRIM_SIZE
        
        if self.EDIT_OP_TOKENS is not None:
            self.num_eop_types = len(self.TLang.OT2T['edit_op'])
        
    def init_pts(self):
        a = (torch.arange(self.VDIM).float() * 2 / self.VDIM) - 1.0 + (1./self.VDIM)
        c = a.unsqueeze(0).repeat(self.VDIM, 1)
        d = a.unsqueeze(1).repeat(1, self.VDIM)
        pts = torch.stack((c,d), dim=2).view(-1, 2).to(device)
        global EX_PTS
        EX_PTS = pts
        
    def make_lang(self):
        self.add_token_info()
        self.set_tlang()

    def set_tlang(self):
        TLang = base.TokenLang(self)

        if self.USE_END_TOKEN:
            TLang.add_token(self.START_TOKEN, 'shape,shape', 'prog')
            TLang.add_token(self.END_TOKEN, '', 'shape')
        else:
            TLang.add_token(self.START_TOKEN, 'shape', 'prog')
            
        TLang.add_token('union', 'shape,shape', 'shape')
        TLang.add_token('scale', 'float,float,shape', 'shape')
        TLang.add_token('move', 'float,float,shape', 'shape')
        TLang.add_token('color', 'cval,shape', 'shape')
        TLang.add_token('prim', 'pval', 'shape')
        TLang.add_token('symReflect', 'axis,shape', 'shape')
        TLang.add_token('symTranslate', 'float,float,int,shape', 'shape')
        TLang.add_token('symRotate', 'int,shape', 'shape')
                
        for t in self.D_PRIMS:
            TLang.add_token(t, '', 'pval')

        for t in self.D_COLORS:
            TLang.add_token(t, '', 'cval')

        for t in self.D_AXIS:
            TLang.add_token(t, '', 'axis')

        for t in self.D_INTS:
            TLang.add_token(str(t), '', 'int')

        for t in self.D_FLTS:
            TLang.add_token(str(t), '', 'float')

        self.EOT2I = {}
        if self.EDIT_OP_TOKENS is not None:            
                    
            for EOT in self.EDIT_OP_TOKENS.keys():
                TLang.add_token(EOT, '', 'edit_op')
                self.EOT2I[EOT] = len(self.EOT2I)

        self.EOI2T = {v:k for k,v in self.EOT2I.items()}
            
        TLang.init()

        self.TLang = TLang
        self.TD_FLTS = torch.tensor(self.D_FLTS)

        self.MOVE_D_FLTS = [df for df in self.D_FLTS if abs(df) <= 1.0]
        self.SCALE_D_FLTS = [df for df in self.D_FLTS if df > 0.0]

        self.MOVE_TD_FLTS = torch.tensor(self.MOVE_D_FLTS)
        self.SCALE_TD_FLTS = torch.tensor(self.SCALE_D_FLTS)
        
    def add_token_info(self):
        
        self.D_PRIMS = ('circle', 'square', 'triangle')
        self.D_COLORS = ('red', 'green', 'blue')
        self.D_AXIS = ('AX', 'AY')
        self.D_INTS = tuple([i for i in range(1, 7)])

        self.D_FLTS = lu.make_flt_map(
            spec_vals=(-4.0,-2.0,-1.0,0.,1.0,2.0,4.0),
            min_val=-1.0,
            max_val= 2.0,
            num_tokens=self.FLT_NTOKENS
        )

        self.G_MIN_SCALE = min([v for v in self.D_FLTS if v > 0.0])

    def get_input_shape(self):
        return [self.VDIM, self.VDIM, 3]

    def get_det_sample_params(self):
        nsp = lu.sample_dist(self.NUM_SUB_PROG_DIST)
        return {
            'num_sub_progs': nsp
        }
    
    def check_valid_tokens(self, tokens, ret_vdata=False, ret_prog=False):
        
        P = Program(self)
        
        try:
            P.run(tokens)
        except Exception as e:            
            if VERBOSE:
                print(f"Exception error {e}")
                print(tokens)
            if ret_prog:
                return None, None
                
            return None

        if tokens == ['START', 'END']:
            if ret_prog:
                return P.make_image(), P
            
            return P.make_image()

        if ret_prog:        
            return self.check_valid_prog(P, ret_vdata), P

        return self.check_valid_prog(P, ret_vdata)

    def make_edit_op(self, A, B):
        return make_edit_op(self, A, B)
    
    def check_valid_prog(self, P, ret_vdata=False):

        if P.has_soft_error():
            if VERBOSE:
                print('se')
            return None

        if P.neg_reject():
            if VERBOSE:
                print('nr')
            return None

        if not ret_vdata:
            return True
        
        try:
            img = P.make_image()
        except Exception as e:
            if VERBOSE: 
                print(f"Failed to make image with {e}")
            img = None

        return img
    
    def sample_det_prog(self, sample_params):

        tokens,_ = prog_sampler.sample_prog(self, sample_params['num_sub_progs'], True)

        if not self.USE_END_TOKEN and tokens[-1] == 'END':
            return tokens[:-1]
        
        return tokens
    
    def render_group(self, images, name=None, rows=1):
        if rows == 1:
            f, axarr = plt.subplots(rows,len(images),figsize=(30,3))
            for i in range(len(images)):
                axarr[i].axis("off")
                
                if images[i] is not None:
                    axarr[i].imshow(images[i].cpu().numpy(), origin='lower')
                
        else:
            num_per_row = math.ceil(len(images) / rows)
            f, axarr = plt.subplots(rows, num_per_row, figsize=(30,3 * rows))
            j = 0
            k = 0
            
            for i in range(len(images)):
                axarr[k][j].axis("off")

                if images[i] is not None:
                    axarr[k][j].imshow(images[i].cpu().numpy(), origin='lower')
                            
                j += 1

                if j == num_per_row:
                    k += 1
                    j = 0
            
        if name is None:
            plt.show()
        else:
            plt.savefig(f'{name}.png')
            
        plt.close()

    def comb_sub_progs(self, sub_shapes):
        return ['START'] + self._comb_sub_progs(sub_shapes) + ['END']

    def _comb_sub_progs(self, ss):
        if len(ss) == 0:
            return []
        
        if len(ss) == 1:
            return ss[0]
    
        return ['union'] + ss[0]  + self._comb_sub_progs(ss[1:])    
    

def format_tokens_for_edit(ex, input_tokens, eot, eol, eos):

    inp_seq = deepcopy(input_tokens)

    if '$' not in eot:
        eo_token = f'${eot}'
    else:
        eo_token = eot
                
    tar_seq = [eo_token] + eos + ['END']

    inp_seq.insert(eol, eo_token)

    return inp_seq, tar_seq

