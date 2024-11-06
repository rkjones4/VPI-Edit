import executors.layout.ex_layout as ex
import executors.layout.lay_prog_diff as lpd
from tqdm import tqdm
from utils import device
import torch
import numpy as np
import utils
import random
import math
import domains.dom_common as dc
from copy import deepcopy

LYT_CMN_ARGS = [
    ('-msl', '--max_seq_len', 128, int), # maximum program length
    ('-mesl', '--max_edit_seq_len', 32, int), # maximum edit length
    ('-rdp', '--real_data_path', 'data/layout/tar', str), # target data path
    ('-mn', '--metric_name', 'color_iou', str) # eval metric name
]

LYT_PT_ARGS = [
    ('-bs', '--batch_size', 128, int), # batch size for pretraining 
    ('-beams', '--beams', 10, int), # beam size of pretraining evaluation
]

LYT_FT_ARGS = [
    ('-bs', '--batch_size', None, int), # Keep as None
    ('-os_bs', '--os_batch_size', 20, int), # beam size for OS finetuning 
    ('-ed_bs', '--edit_batch_size', 128, int), # beam size for edit finetuning

    ('-evs', '--eval_size', 144,  int), # Layout has more validation target shapes

    
]

def load_oneshot(old_domain, load_path=None, is_gen_model=False):
    domain = deepcopy(old_domain)
    del(domain.executor)
    
    config = {
        'EDIT_OP_TOKENS': None,
        'USE_END_TOKEN': False,
    }    

    domain.executor = ex.LayExecutor(config)

    return dc.load_os_net_from_domain(domain, load_path, is_gen_model)

class TargetDataset(dc.BaseTargetDataset):
    def __init__(self, domain):
        self.device = domain.device
        self.eval_batch_size = 1
        args = domain.args
        self.eval_size = args.eval_size
        self.mode = 'eval'
        
        path = args.real_data_path
        
        self.vinput = []
        
        self.train_keys = []
        self.val_keys = []
        self.test_keys = []
        
        for name, key_set, num in [
            ('train', self.train_keys, args.train_size),
            ('val', self.val_keys, args.eval_size),
            ('test', self.test_keys, args.etest_size),
        ]:
        
            data = torch.load(f'{path}_{name}.pt')

            torch.manual_seed(0)
            rinds = torch.randperm(data.shape[0])[:num]
            
            for ri in rinds:
                key_set.append(len(self.vinput))
                self.vinput.append(data[ri])

        self.vinput = torch.stack(self.vinput,dim=0)

        self.train_keys = torch.tensor(self.train_keys).long()
        self.val_keys = torch.tensor(self.val_keys).long()
        self.test_keys = torch.tensor(self.test_keys).long()
        
        print(f"Loaded {self.vinput.shape} with split {(self.train_keys.shape[0],self.val_keys.shape[0],self.test_keys.shape[0])}")        
        

# Class that defines the domain
class LAYOUT_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.base_init()
        self.name='layout'
        self.device = device

    def get_oneshot_net(self, load_model_path=None, is_gen_model=False):
        return load_oneshot(
            old_domain=self,
            load_path=load_model_path,
            is_gen_model=is_gen_model
        )
        
    def load_oneshot_net(self):

        if 'os_net' in self.__dict__:
            pass
        else:
            self.os_net = load_oneshot(self)        

    def load_prog_diff(self):
        PD = lpd.LayProgDiff(self.executor)            
        self.prog_diff = PD
            
    def make_executor(self, args):
        
        edit_op_tokens = {
            '$ACA': 'comb_add_after',
            '$ACB': 'comb_add_before',
            '$CR': 'comb_rm',
            '$TA': 'trans_add',
            '$TR': 'trans_rm',
            '$TM': 'trans_mod',
            '$PM': 'param_mod',
        }

        if args.pred_mode == 'os':
            config = {
                'MAX_TOKENS': args.max_seq_len,
                'EDIT_OP_TOKENS': None,
                'USE_END_TOKEN': False,
            }
        else:        
            config = {
                'MAX_TOKENS': args.max_seq_len,
                'EDIT_OP_TOKENS': edit_op_tokens,            
                'USE_END_TOKEN': True,            
            }
                    
        self.executor = ex.LayExecutor(config)

    def get_vis_metric(self, prog, gt):

        try:
            vdata = self.executor.execute(prog)
        except Exception as e:
            return None, None, None

        recon_metrics = self.pixel_recon_metrics(vdata, gt)
        mval = recon_metrics[self.metric_name]
        recon_metrics['mval_cnt'] = 1.
        
        return vdata, mval, recon_metrics
            
    def pixel_recon_metrics(self, pixels, gt):
                    
        pixels = pixels.to(gt.device) 
        
        def get_occ(inp):
            assert len(inp.shape) == 3
            finp = inp.view(-1, 3)

            o_a = (finp > 0.).any(dim=1)
            o_r = (finp[:,0] > 0.5)
            o_gn = (finp[:,1] > 0.5)
            o_b = (finp[:,2] > 0.5)
            o_gr = (finp.sum(dim=1) > 1.0)

            return o_a, o_r, o_gn, o_b, o_gr
            
        p_a, p_r, p_gn, p_b, p_gr = get_occ(pixels)
        g_a, g_r, g_gn, g_b, g_gr = get_occ(gt)

        def get_iou(a, b):
            I = (a & b).sum().item()
            O = (a | b).sum().item()
            return I / ( O + 1e-8), float(O > 0.0)
        
        a_iou, _ = get_iou(p_a, g_a)

        r_iou, r_w = get_iou(p_r, g_r)
        gn_iou, gn_w = get_iou(p_gn, g_gn)
        b_iou, b_w = get_iou(p_b, g_b)
        gr_iou, gr_w = get_iou(p_gr, g_gr)

        all_score = a_iou
        color_score = ((r_iou * r_w) +\
                       (gn_iou * gn_w) +\
                       (b_iou * b_w) +\
                       (gr_iou * gr_w)) /\
                       (r_w + gn_w + b_w + gr_w + 1e-8)

        match = (0.5 * all_score) + (0.5 * color_score)

        CI = (p_r & g_r).sum() +\
            (p_gn & g_gn).sum() +\
            (p_b & g_b).sum() +\
            (p_gr & g_gr).sum()

        CU = (p_a | g_a).sum()

        cIoU = (CI / (CU + 1e-8)).item()
        
        return {
            'match': match,
            'color_iou': cIoU
        }

    def is_perfect_recon(self, v):
        assert self.metric_name in ('color_iou', 'match')
        return abs(1. - v) < 0.0001
    
    # more early stopping logic, what should the "bad" value of the metric be
    def init_metric_val(self):
        return 0.

    def get_obj_dir(self):
        return 'high'
                   
    def get_cmn_args(self):
        return LYT_CMN_ARGS

    def get_pt_arg_list(self):
        return LYT_PT_ARGS

    def get_ft_arg_list(self):
        return LYT_FT_ARGS

    def extra_eval_log_info(self):
        if 'edit' not in self.args.pred_mode:
            return []
        
        ELI = [
            ('Mval Start', 'mval_start', 'count'),            
            ('Mval Best', 'mval_best', 'count'),
            ('Mval Imp', 'mval_imp', 'count'),
        ]
            
        return ELI

    def extra_train_log_info(self):
        if 'edit' not in self.args.pred_mode:
            return []

        dc.DOM_TRAIN_LOG_INFO.pop(-1)
        
        return [
            ('Type Loss', 'type_loss', 'batch_count'),
            ('Loc Loss', 'loc_loss', 'batch_count'),
            ('Seq Loss', 'seq_loss', 'batch_count'),
            ('Type Acc', 'type_corr', 'type_total'),
            ('Loc Acc', 'loc_corr', 'loc_total'),
            ('Seq Acc', 'seq_corr', 'seq_total'),
        ]


    def load_target_dataset(self):
        return TargetDataset(self)
