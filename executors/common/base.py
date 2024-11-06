import random
from tqdm import tqdm
import math
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

VERBOSE = False

BASE_CONFIG = {    
    'START_TOKEN': 'START',
    'END_TOKEN': 'END',
}

def round_val(sval, tvals):
    bv = None
    be = 1e8
    for t in tvals:
        err = abs(t-sval)
        if err < be:
            bv = t
            be = err

    return bv


def make_edit_batch(ex, data, args):
    B = {

        # visual inputs
        'inp_vdata': torch.zeros(
            tuple([len(data)] + ex.get_input_shape())
        ).float(),
        'vdata': torch.zeros(
            tuple([len(data)] + ex.get_input_shape())
        ).float(),

        # Pred edit op sequence input
        'ps_inp_seq': torch.zeros(
            len(data),
            args.max_seq_len,
        ).long(),
        'ps_inp_seq_weight': torch.zeros(
            len(data),
            args.max_seq_len,
        ).float(),
        
        # Pred edit op sequence targets
        'ps_tar_seq': torch.zeros(
            len(data),
            args.max_edit_seq_len,
        ).long(),
        'ps_tar_seq_weight': torch.zeros(
            len(data),
            args.max_edit_seq_len,
        ).float(),                
                

        # Pred edit type and loc sequence input
        'ptl_inp_seq': torch.zeros(
            len(data),
            args.max_seq_len,
        ).long(),
        'ptl_inp_seq_weight': torch.zeros(
            len(data),
            args.max_seq_len,
        ).float(),

        # Pred edit type and loc targets

        'eop_type_ind': torch.zeros(
            len(data)            
        ).long(),
        
        'eop_loc_ind': torch.zeros(
            len(data),
        ).long(),
        
    }

    for i, D in enumerate(data):

        tar_img = D['tar_vdata']
        inp_img = D['corr_vdata']
        inp_tokens = D['corr_tokens']
                
        eot, eol, eos = D['edit_ps_info']

        inp_eop_seq, tar_eop_seq = ex.format_tokens_for_edit(
            inp_tokens, eot, eol, eos
        )
        
        if len(inp_tokens) > args.max_seq_len or \
           len(inp_eop_seq) > args.max_seq_len or \
           len(tar_eop_seq) > args.max_edit_seq_len:
            if VERBOSE:
                print("Seq too long")
            continue

        try:
            B['vdata'][i] = tar_img
        except:
            B['vdata'][i] = tar_img.unsqueeze(-1)
        try:            
            B['inp_vdata'][i] = inp_img        
        except:
            B['inp_vdata'][i] = inp_img.unsqueeze(-1)
            
        ps_inp_prog_tensor = ex.TLang.tokens_to_tensor(inp_eop_seq)
        ps_tar_prog_tensor = ex.TLang.tokens_to_tensor(tar_eop_seq)                

        B['ps_inp_seq'][i,:ps_inp_prog_tensor.shape[0]] = ps_inp_prog_tensor
        B['ps_inp_seq_weight'][i,:ps_inp_prog_tensor.shape[0]] = 1.
        
        B['ps_tar_seq'][i,:ps_tar_prog_tensor.shape[0]] = ps_tar_prog_tensor
        B['ps_tar_seq_weight'][i,:ps_tar_prog_tensor.shape[0]] = 1.

        edit_tl_info = D['edit_tl_info']

        assert len(edit_tl_info) == 2
        seot, seol = edit_tl_info
        
        ptl_inp_prog_tensor = ex.TLang.tokens_to_tensor(inp_tokens)
        B['ptl_inp_seq'][i,:ptl_inp_prog_tensor.shape[0]] = ptl_inp_prog_tensor
        B['ptl_inp_seq_weight'][i,:ptl_inp_prog_tensor.shape[0]] = 1.

        seoi = ex.EOT2I[f'${seot}']
        
        B['eop_type_ind'][i] = seoi
        B['eop_loc_ind'][i] = seol        
        
    return B
    
def make_os_batch(ex, data, args):

    B = {        
        'seq': torch.zeros(
            len(data),
            args.max_seq_len,
        ).long(),
        'seq_weight': torch.zeros(
            len(data),
            args.max_seq_len,
        ).float(),
        
        'vdata': torch.zeros(tuple([len(data)] + ex.get_input_shape())
        ).float()            
    }

    for i, tokens in enumerate(data):

        if isinstance(tokens, ex.prog_cls):
            img = tokens.make_image()
            if len(img.shape) == 2:
                img = img.unsqueeze(-1)
            
            B['vdata'][i] = img
            continue            
        
        if len(tokens) > args.max_seq_len:
            continue
                
        try:
            img = ex.execute(' '.join(tokens))
        except Exception as e:
            print(f"Had an error in os batch, unexpected: {e}")
            continue

        prog_tensor = ex.TLang.tokens_to_tensor(tokens)        
        
        B['seq'][i,:prog_tensor.shape[0]] = prog_tensor
        B['seq_weight'][i,:prog_tensor.shape[0]] = 1.

        try:
            B['vdata'][i] = img
        except:            
            img = img.unsqueeze(-1)
            B['vdata'][i] = img
            
    return B

class Token:
    def __init__(self, name, inp_types, out_type, use_type):
        self.name = name
        self.inp_types = inp_types
        self.out_type = out_type
        self.use_type = use_type

    def num_inp_tokens(self):
        if self.inp_types == '':
            return 0
        else:            
            return len(self.inp_types.split(','))
        
class TokenLang:
    def __init__(self, executor):
        self.ex = executor
        self.tokens = {}
        
    def add_token(self, name, inp_types, out_type, use_type='def'):
        assert use_type in ('def', 'inp_only', 'out_only')

        t = Token(name, inp_types, out_type, use_type)
        self.tokens[name] = t
        
    def make_params(self):
        self.params = set([
            t.name for t in self.tokens.values() \
            if t.inp_types == '' and t.use_type != 'inp_only'
        ])

    def make_ot2t(self):
        OT2T = {}
        for t in self.tokens.values():
            if t.use_type == 'inp_only':
                continue
            ot = t.out_type
            if ot not in OT2T:
                OT2T[ot] = []
            OT2T[ot].append(t.name)

        self.OT2T = OT2T
        
    def make_t2ipc(self):
        self.T2IPC = {
            t.name:t.num_inp_tokens() for t in self.tokens.values()
        }
        
    def init(self):
        self.make_params()
        self.make_t2ipc()
        self.make_ot2t()
        if 'float' in self.OT2T:
            self.float_vals = [float(f) for f in self.OT2T['float']]
        self.make_token_maps()


    def make_token_maps(self):
        t2i = {}
        self.nt_inp = 0
        self.nt_out = 0
        for ut in ['def', 'out_only', 'inp_only']:            
            for t in self.tokens.values():
                if t.use_type == ut:
                    t2i[t.name] = len(t2i)

                    self.nt_inp += 1

                    if ut in ('def', 'out_only'):
                        self.nt_out += 1

        self.T2I = t2i
        self.I2T = {v:k for k,v in self.T2I.items()}
        self.nt = self.nt_inp
    
    def get_num_tokens(self):
        return self.nt_inp

    def tensor_to_tokens(self, tensor, keep_last=False):
        if keep_last:
            return [self.I2T[t.item()] for t in tensor]
        else:
            return [self.I2T[t.item()] for t in tensor[:-1]]
    
    def tokens_to_tensor(self, tokens):

        vls = []

        for t in tokens:
            if t in self.T2I:
                vls.append(self.T2I[t])
            else:
                val = str(round_val(float(t), self.float_vals))
                vls.append(self.T2I[val])
                
        return torch.tensor(vls)        
            
    def get_num_inp(self, t):
        
        if t in self.T2IPC:
            return self.T2IPC[t]
        else:
            return 0

    def get_num_inp_eq(self, t, eq):
        return self.tokens[t].inp_types.count(eq)

    def get_inp_types(self, t):

        if t not in self.tokens:
            return []
        
        return [ip for ip in self.tokens[t].inp_types.split(',') if len(ip) > 0]

    def get_out_type(self, t):
        if t not in self.tokens:
            return ''
        else:
            return self.tokens[t].out_type

class BaseExecutor:

    def make_plot_render(self, num, gs, fs):

        fig, axes = plt.subplots(num, gs, figsize=fs)

        for i in range(num):
            for j in range(gs):
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
                axes[i,j].axis('off')
                
        return fig, axes
    
    def base_init(self, prm_config):
    
        BASE_CONFIG.update(prm_config)

        for k,v in BASE_CONFIG.items():
            setattr(self,k,v)
                        
    def sample_prog_struct(self, params):
        assert False, f'need to impl sample prog struct'

    def make_os_batch(self, data, args):
        return make_os_batch(self, data, args)

    def make_edit_batch(self, data, args):
        return make_edit_batch(self, data, args)
        
    def make_batch(self, data, args):
        if args.pred_mode == 'os':
            return self.make_os_batch(data, args)
        elif args.pred_mode == 'edit':
            return self.make_edit_batch(data, args)
        else:
            assert False
                
    def det_prog_random_sample(
            self, num, vis_progs = False, use_pbar=True, print_stats=False, ret_data=False
    ):
        with torch.no_grad():
            return self._det_prog_random_sample(num, vis_progs, use_pbar, ret_data)

    def _det_prog_random_sample(self, num, vis_progs, use_pbar, ret_data):
        max_tokens = self.MAX_TOKENS

        data = []
        vis_data = []
        
        if not use_pbar:
            pbar = None
        else:
            pbar = tqdm(total=num)        
        
        sample_params = None
        t = 0
        
        while len(data) < num:
            if t > self.SP_PATIENCE or sample_params is None:
                sample_params = self.get_det_sample_params()
                t = 0

            tokens = self.sample_det_prog(sample_params)
            
            t += 1

            if tokens is None or len(tokens) >= max_tokens:
                continue
            
            vdata = self.check_valid_tokens(tokens, ret_vdata=True)
            
            if vdata is None or vdata is False:
                continue

            t = 0
            sample_params = None
            
            if pbar:
                pbar.update(1)
        
            if vis_progs or ret_data:
                vis_data.append(vdata)

            data.append(tokens)

            
        if pbar:
            pbar.close()

        if vis_progs:
            self.render_group(vis_data, name=None, rows= math.ceil(math.sqrt(len(vis_data))))

        if ret_data:
            return data, vis_data
        
        return data
            
