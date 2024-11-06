import cv2
import executors.csg2d.ex_csg2d as ex
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
import h5py

LYT_CMN_ARGS = [
    ('-msl', '--max_seq_len', 164, int), # maximum program length
    ('-mesl', '--max_edit_seq_len', 32, int), # maximum edit length
    ('-rdp', '--real_data_path', 'data/csg2d/cad.h5', str), # target data path
    ('-mn', '--metric_name', 'inv_cd', str) # eval metric name
]

LYT_PT_ARGS = [
    ('-bs', '--batch_size', 128, int), # batch size for pretraining
    ('-beams', '--beams', 10, int), # beam size of pretraining evaluation
]

LYT_FT_ARGS = [
    ('-bs', '--batch_size', None, int), # Keep as None
    ('-os_bs', '--os_batch_size', 20, int), # beam size for OS finetuning
    ('-ed_bs', '--edit_batch_size', 128, int), # beam size for edit finetuning
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
        
        with h5py.File(path, "r") as hf:
            
            for key in ['train', 'val', 'test']:
            
                all_data[key] = torch.from_numpy(np.array(hf.get(f'{key}_images'))).float().flip(dims=[1])
                                
        
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

            rinds = torch.arange(min(num, len(data)))

            for ri in rinds:
                key_set.append(len(self.vinput))
                self.vinput.append(data[ri])
                
        self.vinput = torch.stack(self.vinput,dim=0)

        self.train_keys = torch.tensor(self.train_keys).long()
        self.val_keys = torch.tensor(self.val_keys).long()
        self.test_keys = torch.tensor(self.test_keys).long()
        
        print(f"Loaded {self.vinput.shape} with split {(self.train_keys.shape[0],self.val_keys.shape[0],self.test_keys.shape[0])}")        
    
# Class that defines the domain
class CSG2D_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.base_init()
        self.name='csg2d'
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
            }
        else:        
            config = {
                'MAX_TOKENS': args.max_seq_len,
                'EDIT_OP_TOKENS': edit_op_tokens,            
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

        cd = image_chamfer(
            pixels.unsqueeze(0).cpu().numpy(),
            gt.unsqueeze(0).cpu().numpy()
        )[0]
        
        return {
            'iou': iou,
            'cd': cd,
            'inv_cd': 10. - cd
        }

    def is_perfect_recon(self, v):
        if self.metric_name == 'iou':
            return abs(1. - v) < 0.0001
        elif self.metric_name == 'inv_cd':
            return abs(10. - v) < 0.0001
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
                ('CD', 'cd', 'count'),
            ]
        
        ELI = [
            ('Mval Start', 'mval_start', 'count'),            
            ('Mval Best', 'mval_best', 'count'),
            ('Mval Imp', 'mval_imp', 'count'),
        ]
        return ELI
        

    def load_target_dataset(self):
        return TargetDataset(self)


def image_chamfer(images1, images2):
    """
    Chamfer distance on a minibatch, pairwise.
    :param images1: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :param images2: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :return: pairwise chamfer distance
    """
    # Convert in the opencv data format
    images1 = images1.astype(np.uint8)
    images1 = images1 * 255
    images2 = images2.astype(np.uint8)
    images2 = images2 * 255
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size**2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (summ1[i] == 0) or (summ2[i] == 0) or (summ1[i] == filled_value) or (summ2[\
                i] == filled_value):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3)

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3)
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (
        np.sum(E2, (1, 2)) + 1) + np.sum(D2 * E1, (1, 2)) / (np.sum(E1, (1, 2)) + 1)

    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 10
    return distances
