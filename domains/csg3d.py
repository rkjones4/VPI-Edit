import executors.csg3d.ex_csg3d as ex
import executors.csg2d.csg_prog_diff as cpd
from tqdm import tqdm
from utils import device
import torch
import numpy as np
import utils
import random
import math
import domains.dom_common as dc
from copy import deepcopy
import h5py, time

MPOOL32 = torch.nn.MaxPool3d(kernel_size=2)
    
def convert_to_v32(A):
    with torch.no_grad():
        return MPOOL32(A.float()).bool()

CATS = ['chair', 'table', 'couch', 'bench']

LYT_CMN_ARGS = [
    ('-msl', '--max_seq_len', 256, int), # maximum program length
    ('-mesl', '--max_edit_seq_len', 48, int), # maximum edit length
    ('-mp', '--max_prim_enc', 8, int), # 3D CNN only produces 8 visual tokens
    ('-rdp', '--real_data_path', 'data/csg3d/', str), # target data path
    ('-vd', '--voxel_dim', 32, int), # voxel dimension
    ('-mn', '--metric_name', 'iou', str) # eval metric name
]

LYT_PT_ARGS = [
    ('-bs', '--batch_size', 64, int), # batch size for pretraining
    ('-beams', '--beams', 10, int), # beam size of pretraining evaluation
]

LYT_FT_ARGS = [
    ('-bs', '--batch_size', None, int), # Keep as None
    ('-os_bs', '--os_batch_size', 20, int), # beam size for OS finetuning
    ('-ed_bs', '--edit_batch_size', 64, int), # beam size for edit finetuning
]

def load_oneshot(old_domain, load_path=None, is_gen_model=False):
    domain = deepcopy(old_domain)
    del(domain.executor)
    
    config = {
        'EDIT_OP_TOKENS': None,
    }    

    domain.executor = ex.CSGExecutor(config)

    return dc.load_os_net_from_domain(domain, load_path, is_gen_model)

    
class TargetDataset(dc.BaseTargetDataset):
    def __init__(self, domain):
        self.device = domain.device
        self.eval_batch_size = 1
        args = domain.args
        self.eval_size = args.eval_size                
        self.mode = 'eval'
        
        path = args.real_data_path

        all_data = {}

        for key in ['train', 'val', 'test']:
            all_data[key] = []
            for cat in CATS:
                data = h5py.File(f'{path}/{cat}/{cat}_{key}_vox.hdf5', 'r')
                raw_voxels = torch.from_numpy(data['voxels'][:,:,:,:,0]).flip(dims=[3]).transpose(1,3)                    
                all_data[key].append(raw_voxels)
                                
            all_data[key] = torch.cat(all_data[key],dim=0)
                
        self.vinput = []
        
        self.train_keys = []
        self.val_keys = []
        self.test_keys = []        
        
        for name, key_set, num in [
            ('train', self.train_keys, args.train_size),
            ('val', self.val_keys, args.eval_size),
            ('test', self.test_keys, args.etest_size),
        ]:
        
            data = all_data[name]
            torch.manual_seed(0)
            rinds = torch.randperm(data.shape[0])[:num]
        
            for ri in rinds:
                key_set.append(len(self.vinput))
                self.vinput.append(data[ri])
                
        self.vinput = torch.stack(self.vinput,dim=0)
        self.vinput = convert_to_v32(self.vinput)
            
        self.train_keys = torch.tensor(self.train_keys).long()
        self.val_keys = torch.tensor(self.val_keys).long()
        self.test_keys = torch.tensor(self.test_keys).long()
        
        print(f"Loaded {self.vinput.shape} with split {(self.train_keys.shape[0],self.val_keys.shape[0],self.test_keys.shape[0])}")        
    
# Class that defines the domain
class CSG3D_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.base_init()
        self.name='csg3d'
        self.device = device
                        
    def get_oneshot_net(self, load_model_path=None, is_gen_model=False):
        return load_oneshot(
            self,
            load_model_path,
            is_gen_model
        )
        
    def load_oneshot_net(self):
        if 'os_net' in self.__dict__:
            pass
        else:
            self.os_net = load_oneshot(self)


    def load_prog_diff(self):
        PD = cpd.CSGProgDiff(self.executor)
        self.prog_diff = PD
        
    def make_executor(self, args):

        edit_op_tokens = {
            '$ACA': 'comb_add_after',
            '$ACB': 'comb_add_before',
            '$CM': 'comb_mod',
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
                'VDIM': args.voxel_dim,
            }
        else:        
            config = {
                'MAX_TOKENS': args.max_seq_len,
                'EDIT_OP_TOKENS': edit_op_tokens,
                'VDIM': args.voxel_dim,
            }
                    
        self.executor = ex.CSGExecutor(config)

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
        
        def get_iou(a, b):
            I = (a & b).sum().item()
            O = (a | b).sum().item()
            return I / ( O + 1e-8)

        iou = get_iou(pixels.flatten().bool(), gt.flatten().bool())

        return {
            'iou': iou,
        }

    def is_perfect_recon(self, v):
        if 'iou' in self.metric_name:
            return abs(1. - v) < 0.0001
        else:
            False, f'bad metric name {self.metric_name}'
    
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
            return [
                ('IoU', 'iou', 'count'),
            ]
        
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

