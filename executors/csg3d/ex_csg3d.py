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
sys.path.append('executors/csg3d')
sys.path.append('..')
sys.path.append('../common')
import csg3d_prog_sampler as prog_sampler
import base
import lutils as lu

device=torch.device('cuda')


CSG_CONFIG = {
    'MAX_TOKENS': 256,
    'VDIM': 32,
    
    'BASE_TYPE': 'shape',
    'SP_PATIENCE': 10,
    
    'DEF_PRIM_SIZE': 1.0,    

    'MIN_DIFF_VOXELS': 8,

    'OUT_FACTOR': 1.1,

    'NUM_POS_SUB_PROG_DIST': (
        [1,2,3,4,5,6,7,8],
        lu.norm_np([.025,.075, .125, .175, .15, .1, .05, .025])
    ),

    'NUM_NEG_SUB_PROG_DIST': (
        [0,1,2,3],
        lu.norm_np([.5,.25, .125, .05])
    ),

    'FLT_NTOKENS_MOVE': 99,
    'FLT_NTOKENS_SCALE': 50,
    'FLT_NTOKENS_ROT': 16,
    
    'USE_END_TOKEN': True
}

G_SF = None
OUT_PTS = None
G_PTS = None

CONVENTION = ("X", "Y", "Z")

def _axis_angle_rotation_3D(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3)).to(device)

def vis_spc(pts, name):
    with open(f'{name}.obj', 'w') as f:
        for pt in pts:
            x,y,z = pt.tolist()
            f.write(f'v {x} {y} {z}\n')

def make_flt_map(
    ex,
    fltkn,
    spec_vals,
    min_val,
    max_val,
    num_tokens,
    fprec=2
):

    m = {}
    
    sl = list(spec_vals)

    count = 0
    
    for sv in sl:
        count += 1
        m[f'{fltkn}_{count}'] = round(sv, fprec)
        
    num_tokens -= count

    diff = (max_val - min_val) 
    
    for _i in range(1, num_tokens+1):
        val = min_val + ((_i / (num_tokens+1.)) * diff)
        m[f'{fltkn}_{_i+count}'] = round(val, fprec)

    return m

class Program:
    
    def __init__(self, ex):
        self.ex = ex
        self.state = None
        
        self.prim_map = {
            'cube': self.get_cube_sdf,
            'cylinder': self.get_cylinder_sdf,
            'sphere': self.get_sphere_sdf,
        }
        self.soft_error = False
        self.check_soft_errors = False
        
    def get_cylinder_sdf(self, points):
        r = 1.0
        h = 1.0
        
        xy_vec = torch.norm(points[..., 0::2], dim=-1) - r
        height = torch.abs(points[..., 1]) - h
        vec2 = torch.stack([xy_vec, height], -1)
        base_sdf = torch.amax(vec2, -1) + torch.norm(torch.clip(vec2, min=0.0) + 1e-9, -1)
        return base_sdf
                                        
    def get_sphere_sdf(self, points):
        base_sdf = points.norm(dim=-1)
        base_sdf = base_sdf - 1.0
        return base_sdf
        
    def get_cube_sdf(self, points):
        points = torch.abs(points)
        points -= 1.0
        base_sdf = torch.norm(
            torch.clip(points, min=0), dim=-1
        ) + torch.clip(
            torch.amax(points, -1), max=0
        )        
        return base_sdf
            
    def ex_primitive(self, cmd, pts, sf):
        sdfs = self.prim_map[cmd](pts / self.ex.DEF_PRIM_SIZE)
        return sdfs, sf
    
    def ex_rot(self, pts, x_ang, y_ang, z_ang):

        params = [
            torch.tensor((-ang / 180.) * 3.14)
            for ang in (x_ang, y_ang, z_ang)
        ]

        matrices = [
            _axis_angle_rotation_3D(c, e) for c, e in zip(CONVENTION, params)
        ]
        TM = torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])[:3, :3]

        n_pts = torch.einsum("ij,mj->mi", TM, pts)        

        return n_pts

    def reflect_transform(self, pts, axis):
        if axis == 'AX':
            return pts * torch.tensor([[-1, 1, 1]], device=pts.device)
        elif axis == 'AY':
            return pts * torch.tensor([[1, -1, 1]], device=pts.device)
        elif axis == 'AZ':
            return pts * torch.tensor([[1, 1, -1]], device=pts.device)
        else:
            assert False, 'bad reflect'
            
    def ex_move(self, pts, sf, X, Y, Z):
        return pts - ((1/sf) * torch.tensor([[X, Y, Z]], device=pts.device)), sf
    
    def ex_scale(self, pts, sf, W, H, D):
        nsf = torch.tensor([[W, H, D]], device=pts.device) + 1e-8
        return pts / nsf, nsf * sf

    def ex_part_union(self, A, B):
        C = torch.stack((A,B),dim=2)
        r = C.min(dim=2).values
        return r

    def ex_part_diff(self, A, _B):
        B = _B.unsqueeze(-1).repeat(1,A.shape[1])
        C = torch.stack((A,B),dim=2)
        return C.max(dim=2).values

    def ex_part_diff(self, A, _B):
        B = _B.unsqueeze(-1).repeat(1,A.shape[1])
        C = torch.stack((A,-B),dim=2)
        return C.max(dim=2).values
    
    def ex_union(self, A, B):
        C = torch.stack((A,B),dim=1)
        return C.min(dim=1).values

    def ex_inter(self, A, B):
        C = torch.stack((A,B),dim=1)
        return C.max(dim=1).values

    def ex_diff(self, A, B):
        C = torch.stack((A,-B),dim=1)
        return C.max(dim=1).values

    def ex_part_op(self, op, L_AO, L_PO, R_AO, R_PO):
        if op == 'union':
            return self.ex_part_union(L_PO, R_PO)
        elif op == 'diff':
            return self.ex_part_diff(L_PO, R_AO)
        elif op == 'inter':
            return self.ex_part_inter(L_PO, R_AO)
        else:
            assert False
        
    def ex_op(self, op, A, B, sf):
        if op == 'union':
            res = self.ex_union(A, B)

        elif op == 'diff':
            res = self.ex_diff(A, B)
                                                            
        elif op == 'inter':
            res= self.ex_inter(A, B)

        else:
            assert False

        if self.check_soft_errors:
            if (sf != 1.0).all() and op != 'union':
                self.soft_error = True
            else:
                A_occ = (A<0.0).float()
                B_occ = (B<0.0).float()
                C_occ = (res<0.0).float()
                if A_occ.sum() < self.ex.MIN_DIFF_VOXELS \
                   or B_occ.sum() < self.ex.MIN_DIFF_VOXELS \
                   or C_occ.sum() < self.ex.MIN_DIFF_VOXELS \
                   or (A_occ - C_occ).abs().sum() < self.ex.MIN_DIFF_VOXELS \
                   or (B_occ - C_occ).abs().sum() < self.ex.MIN_DIFF_VOXELS:
                    self.soft_error = True
                
        return res
            
    def get_move_ftoken(self, t):
        if '.' in t:
            return float(t)
        return self.ex.T2F('move', t)

    def get_scale_ftoken(self, t):
        if '.' in t:
            return float(t)
        return self.ex.T2F('scale', t)

    def get_rot_ftoken(self, t):
        if '.' in t:
            return float(t)
        return self.ex.T2F('rot', t)
    
    def blank_canvas(self):
        return torch.zeros(self.VDIM * self.VDIM * self.VDIM, device=device).float()
    
    def resolve_canvas(self):

        if len(self.prim_info['pos']) > 1:
            pos_union = torch.stack(self.prim_info.pop('pos'),dim=1).min(dim=1).values
            
        elif len(self.prim_info['pos']) == 1:
            pos_union = self.prim_info.pop('pos')[0]
        else:
            pos_union = self.blank_canvas()
            
        if len(self.prim_info['neg']) > 0:
            if len(self.prim_info['neg']) > 1:
                neg_union = torch.stack(self.prim_info.pop('neg'),dim=1).min(dim=1).values
            else:
                neg_union = self.prim_info.pop('neg')[0]
                
            canvas = self.ex_diff(pos_union, neg_union)

            A_occ = (pos_union < 0.).float()
            B_occ = (neg_union < 0.).float()
            C_occ = (canvas < 0.).float()
            
            if A_occ.sum() < self.ex.MIN_DIFF_VOXELS \
               or B_occ.sum() < self.ex.MIN_DIFF_VOXELS \
               or C_occ.sum() < self.ex.MIN_DIFF_VOXELS \
               or (A_occ - C_occ).abs().sum() < self.ex.MIN_DIFF_VOXELS \
               or (B_occ - C_occ).abs().sum() < self.ex.MIN_DIFF_VOXELS:
                self.soft_error = True
            
        else:
            canvas = pos_union
            
        return canvas
        
    def _execute(self, fn, params, pts, sf):

        if fn == self.ex.START_TOKEN:
            self.prim_info = {
                'pos': [],
                'neg': []
            }
                        
            assert len(params) == 1
            self.execute(params[0], pts, sf)

            return self.resolve_canvas()

        elif fn == self.ex.END_TOKEN:
            return None
        
        elif fn == 'POS':
            assert len(params) == 2
            out = self.execute(params[0], pts, sf)[0]

            self.prim_info['pos'].append(out)

            return self.execute(params[1], pts, sf)

        elif fn == 'NEG':
            assert len(params) == 2
            out = self.execute(params[0], pts, sf)[0]

            self.prim_info['neg'].append(out)

            return self.execute(params[1], pts, sf)
            
        elif fn == 'prim':            
            assert len(params) == 1
            assert params[0] in self.ex.D_PRIMS
            return self.ex_primitive(params[0], pts, sf)

        elif fn == 'move':            
            assert len(params) == 4
            n_pts, sf = self.ex_move(
                pts,
                sf,
                self.get_move_ftoken(params[0]),
                self.get_move_ftoken(params[1]),
                self.get_move_ftoken(params[2])
            )
            return self.execute(params[3], n_pts, sf)

        elif fn == 'scale':            
            assert len(params) == 4
            n_pts, sf = self.ex_scale(
                pts,
                sf,
                self.get_scale_ftoken(params[0]),
                self.get_scale_ftoken(params[1]),
                self.get_scale_ftoken(params[2])
            )

            _sdf, _sf = self.execute(params[3], n_pts, sf) 
            return _sdf * sf[:,0], _sf

        elif fn == 'rot':
            assert len(params) == 4
            n_pts = self.ex_rot(
                pts,
                self.get_rot_ftoken(params[0]),
                self.get_rot_ftoken(params[1]),
                self.get_rot_ftoken(params[2]),
            )
            return self.execute(params[3], n_pts, sf)

        elif fn == 'reflect':
            assert len(params) == 2

            axis = params[0]
            assert axis in self.ex.D_AXIS
            
            m_pts = self.reflect_transform(pts, axis)
            
            s1, _ = self.execute(params[1], pts, sf)
            s2, _ = self.execute(params[1], m_pts, sf)

            if self.check_soft_errors:
                A_occ = (s1<0.0).float()
                B_occ = (s2<0.0).float()
                if A_occ.sum() < self.ex.MIN_DIFF_VOXELS \
                   or B_occ.sum() < self.ex.MIN_DIFF_VOXELS \
                   or (A_occ - B_occ).abs().sum() < self.ex.MIN_DIFF_VOXELS:
                    self.soft_error = True
            
            return self.ex_union(s1, s2), sf
            
        elif fn in self.ex.D_OPS:
            params = [self.execute(p, pts, sf)[0] for p in params]            
            return self.ex_op(fn, params[0], params[1], sf), sf
                        
        elif fn in self.ex.TLang.tokens:
            return fn
        
        else:
            assert False, f'bad function {fn}'
            
    def execute(self, expr, pts, sf):

        if not isinstance(expr, list):
            assert expr in self.ex.TLang.tokens
            return self._execute(expr, [], pts, sf)
        
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

        return self._execute(fn, params, pts, sf)
    
class CSGExecutor(base.BaseExecutor):
    def __init__(self, config = None):
        if config is not None:
            CSG_CONFIG.update(config)

        self.name = 'csg3d'
        self.prog_cls = Program
        self.base_init(CSG_CONFIG)
        self.make_lang()
        self.init_pts()
        self.set_globals()
        
    def vis_on_axes(self, ax, img):
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_proj_type('persp')
        ax.set_box_aspect(aspect = (1,1,1))
        ax.set_xlim(0, self.VDIM)
        ax.set_ylim(0, self.VDIM)
        ax.set_zlim(0, self.VDIM)
        
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            
        ax.voxels(
            filled=(img >= 0.5).permute(2,0,1).flip(dims=[1]),
            shade=True
        )
                
    def execute(self, expr, vis=False):

        # Slight hack to avoid very slow execution for obviously bad programs
        if expr.count('reflect') >= 8:
            assert False
            
        return self.run(expr,vis, cse=False)
    
    def render(self, img, name=None):

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1, projection='3d')        

        self.vis_on_axes(ax, img)

        if name is None:
            plt.show()        
        else:
            plt.savefig(f'{name}.png')
            
        plt.close('all')
        plt.clf()

        
    def run(self, expr, vis=False, cse=False):
        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN
                        
        with torch.no_grad():
            P = Program(self)
            P.check_soft_errors = cse        
            image_sdf = P.execute(tokens, G_PTS, G_SF)

            img = (image_sdf <= 0.0).float().view(self.VDIM, self.VDIM, self.VDIM)
                
        if vis:
            self.render(img, None)

        if cse:
            se = P.soft_error
            if not se:
                P = Program(self)
                out_sdf = P.execute(tokens, OUT_PTS, G_SF)
                se = (out_sdf <= 0.0).any()

            return img, se
            
        else:
            return img
        
    def set_globals(self):
        pass
        
    def init_pts(self):

        global G_PTS
        global OUT_PTS
        global G_SF

                
        a = (torch.arange(self.VDIM).float() * 2 / self.VDIM) - 1.0 + (1./self.VDIM)

        b = a.unsqueeze(0).unsqueeze(0).repeat(self.VDIM, self.VDIM, 1)
        c = a.unsqueeze(0).unsqueeze(2).repeat(self.VDIM, 1, self.VDIM)
        d = a.unsqueeze(1).unsqueeze(2).repeat(1, self.VDIM, self.VDIM)
        pts = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)        
        
        G_PTS = pts

        face_pts_x = a.unsqueeze(0).repeat(self.VDIM, 1)
        face_pts_y = a.unsqueeze(1).repeat(1, self.VDIM)

        face_pts = torch.stack((face_pts_x, face_pts_y), dim =2).view(-1, 2)
        
        OUT_PTS = torch.cat((
            torch.stack((face_pts[:,0], face_pts[:,1], torch.ones(face_pts.shape[0])*self.OUT_FACTOR), dim=1),
            torch.stack((face_pts[:,0], torch.ones(face_pts.shape[0])*self.OUT_FACTOR, face_pts[:,1]), dim=1),
            torch.stack((torch.ones(face_pts.shape[0])*self.OUT_FACTOR, face_pts[:,0], face_pts[:,1]), dim=1),

            torch.stack((face_pts[:,0], face_pts[:,1], torch.ones(face_pts.shape[0])*self.OUT_FACTOR*-1), dim=1),
            torch.stack((face_pts[:,0], torch.ones(face_pts.shape[0])*self.OUT_FACTOR*-1, face_pts[:,1]), dim=1),
            torch.stack((torch.ones(face_pts.shape[0])*self.OUT_FACTOR*-1, face_pts[:,0], face_pts[:,1]), dim=1),            
        ),dim=0)
        OUT_PTS = OUT_PTS.to(device)
        
        G_SF = torch.ones(1,3,device=device).float()

        
    def make_lang(self):
        self.add_token_info()
        self.set_tlang()

    def set_tlang(self):
        TLang = base.TokenLang(self)

        TLang.add_token(self.START_TOKEN, 'shape', 'prog')        
        TLang.add_token(self.END_TOKEN, '', 'shape')
            
        TLang.add_token('POS', 'shape,shape', 'shape')
        TLang.add_token('NEG', 'shape,shape', 'shape')
        
        for t in self.D_OPS:
            TLang.add_token(t, 'shape,shape', 'shape')
        
        TLang.add_token('scale', 'sflt,sflt,sflt,shape', 'shape')
        TLang.add_token('move', 'mflt,mflt,mflt,shape', 'shape')
        
        TLang.add_token('rot', 'rflt,rflt,rflt,shape', 'shape')
        
        TLang.add_token('prim', 'pval', 'shape')

        TLang.add_token('reflect', 'axis,shape', 'shape')
                
        for t in self.D_PRIMS:
            TLang.add_token(t, '', 'pval')        

        for t in self.D_AXIS:
            TLang.add_token(t, '', 'axis')

        for ktype, KEY_SET in [
            ('mflt', self.MAP_MOVE_FLTS.keys()),
            ('sflt', self.MAP_SCALE_FLTS.keys()),
            ('rflt', self.MAP_ROT_FLTS.keys()),
        ]:
            for v in KEY_SET:
                TLang.add_token(v, '', ktype)

        self.EOT2I = {}
        if self.EDIT_OP_TOKENS is not None:            
                    
            for EOT in self.EDIT_OP_TOKENS.keys():
                TLang.add_token(EOT, '', 'edit_op')
                self.EOT2I[EOT] = len(self.EOT2I)

        self.EOI2T = {v:k for k,v in self.EOT2I.items()}
            
        TLang.init()

        self.TLang = TLang

    def add_token_info(self):

        self.D_PRIMS = ('cube', 'sphere', 'cylinder')
        self.D_OPS = ('union', 'inter', 'diff')
        self.D_AXIS = ('AX', 'AY', 'AZ')
        
        self.MAP_MOVE_FLTS = make_flt_map(
            self,
            'mflt',
            spec_vals=(),
            min_val=-1.0,
            max_val= 1.0,
            num_tokens=self.FLT_NTOKENS_MOVE
        )

        self.MAP_SCALE_FLTS = make_flt_map(
            self,
            'sflt',
            spec_vals=(1.,),
            min_val=0,
            max_val= 1.,
            num_tokens=self.FLT_NTOKENS_SCALE
        )

        self.MAP_ROT_FLTS = make_flt_map(
            self,
            'rflt',
            spec_vals=(0.,),
            min_val= 0.,
            max_val= 360.,
            num_tokens=self.FLT_NTOKENS_ROT
        )

        self.FLT_INFO = {
            'move': self.MAP_MOVE_FLTS,
            'scale': self.MAP_SCALE_FLTS,
            'rot': self.MAP_ROT_FLTS,
        }

        self.TFLT_INFO = {
            k: torch.tensor(list(self.FLT_INFO[k].values()))
            for k in self.FLT_INFO                            
        }
        self.LFLT_INFO = {
            k: list(self.FLT_INFO[k].values())
            for k in self.FLT_INFO                            
        }
        
        self.REV_FLT_INFO = {
            op: {str(v):k for k,v in self.FLT_INFO[op].items()}
            for op in self.FLT_INFO
        }

        self.G_MIN_MOVE = min(list(self.FLT_INFO['move'].values()))
        self.G_MAX_MOVE = max(list(self.FLT_INFO['move'].values()))

        self.G_MIN_SCALE = min(list(self.FLT_INFO['scale'].values()))
        self.G_MAX_SCALE = max(list(self.FLT_INFO['scale'].values()))
                

    def T2F(self, op, t):
        return self.FLT_INFO[op][t]

    def F2T(self, op, v):
        return self.REV_FLT_INFO[op][str(v)]
        
    def get_input_shape(self):
        return [self.VDIM, self.VDIM, self.VDIM, 1]

    def sample_det_prog(self, sample_params):
        with torch.no_grad():
            try:            
                tokens = prog_sampler.sample_prog(
                    self, sample_params,
                )
            except Exception as e:
                print(f"Failed sample prog with {e}")
                return None
            
        return tokens

    def format_tokens_for_edit(self, input_tokens, eot, eol, eos):
        return format_tokens_for_edit(self, input_tokens, eot, eol, eos)
    
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
    
    def get_det_sample_params(self):
        np_sp = lu.sample_dist(self.NUM_POS_SUB_PROG_DIST)
        nn_sp = lu.sample_dist(self.NUM_NEG_SUB_PROG_DIST)
        
        return {
            'num_pos_sub_progs': np_sp,
            'num_neg_sub_progs': nn_sp,
        }
                    
    def check_valid_tokens(self, tokens, ret_vdata=False):
        try:
            img, soft_error = self.run(' '.join(tokens), vis=False, cse=True)
        except Exception as e:
            return None
                
        if ret_vdata:
            if soft_error:
                return None
            else:
                return img

        if soft_error:
            return None
        else:
            return True

    def make_plot_render(self, num, gs, fs):

        fig = plt.figure(figsize=fs)

        c = 0
        axes = {}
        for i in range(num):
            for j in range(gs):
                c += 1
                ax = fig.add_subplot(num, gs, c, projection='3d')
                ax.axis('off')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_zlabel('y')
                ax.set_proj_type('persp')
                ax.set_box_aspect(aspect = (1,1,1))                
                axes[(i,j)] = ax
                                
        return fig, axes
        
        
    def render_group(self, images, name=None, rows=1):

        if rows == 1:
            fig = plt.figure(figsize=(16,2))
        else:
            fig = plt.figure(figsize=(16,8))

        for i, image in enumerate(images):
            ax = fig.add_subplot(rows, math.ceil(len(images)/rows), i+1, projection='3d')
            ax.axis("off")
            self.vis_on_axes(ax, image.cpu())

        plt.tight_layout()
            
        if name is None:
            plt.show()
        else:
            plt.savefig(f'{name}.png')
            
        plt.close()
        plt.clf()

    def get_sp_bounds_dim_pdist(self):
        return (
            (0.6, (0.6, 0.3)),
            (0.4, (0.125, 0.1))
        )

    def get_prim_bounds_dim_pdist(self, v):
        return (
            (0.2, (v, 0.0)),
            (0.6, (v * 0.75, v * 0.25)),
            (0.2, (v * 0.25, v * 0.25))
        )

    def get_sp_bounds_pos_pdist(self):
        return (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        )

    def get_prim_bounds_pos_pdist(self, p, v):
        return (
            (0.3, (p, 0.0)),
            (0.7, (p, v)),
        )



def format_tokens_for_edit(ex, input_tokens, eot, eol, eos):

    inp_seq = deepcopy(input_tokens)

    if '$' not in eot:
        eo_token = f'${eot}'
    else:
        eo_token = eot
                
    tar_seq = [eo_token] + eos + ['END']

    inp_seq.insert(eol, eo_token)

    return inp_seq, tar_seq
