import sys
from utils import device
import utils
import torch
import random
import numpy  as np
from os_models import OneShotNet

VERBOSE = False

DOM_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Accuracy', 'corr', 'total')
]

DOM_EVAL_LOG_INFO = [    
    ('Obj', 'mval', 'count'),
    ('Obj Imp', 'mval_imp', 'count'),    
    ('Errors', 'errs', 'count'),
]

# COMMON ARGS
DOM_CMN_ARGS = [
        
    ('-en', '--exp_name', None,  str), # name of experiment
    ('-pm', '--pred_mode', 'edit', str), # prediction mode (edit vs os [oneshot])        
    ('-rd', '--rd_seed', 42,  int), # random seed
    ('-o', '--outpath', 'model_output',  str), # where experiment output will be written
    ('-dp', '--dropout', .1, float), # dropout
    ('-lr', '--lr', 0.0001,  float), # learning rate

    # transformer arch settings
    ('-nl', '--num_layers', 8, int), 
    ('-nh', '--num_heads', 16, int),
    ('-hd', '--hidden_dim', 256, int),

    # visual encoder settings
    ('-vhd', '--venc_hidden_dim', 256, int),    
    ('-mp', '--max_prim_enc', 16, int), # number of visual tokens
    
    # where to load model from
    ('-lmp', '--load_model_path', None, str),
    ('-os_lmp', '--os_load_model_path', None, str), # load os model when joint finetuning

    # used to restart runs
    ('-os_glmp', '--os_gen_load_model_path', None, str), # load os gen model when joint finetuning    
    ('-lrp', '--load_res_path', None, str), # load results from previous experiments
    
    ('-logp', '--log_period', 10, int), # how often to log results during pretraining (set to 100 for edit pretraining)
    ('-accp', '--acc_period', 1, int), # accumulation period over batches
    ('-svp', '--save_per', None, int), # how often to save model during pretraining, defaults to eval_per
    
    ('-ebs', '--eval_batch_size', 1, int), # keep at 1    

    ('-inf_isb', '--infer_is_beam', 3, int), # inner search beam size
    ('-inf_ismo', '--infer_is_mode', 'beam', str), # inner search mode (sample, beam)
    
    ('-spcp', '--synth_pair_cache_path', None, str), # for edit pretraining, where to find pairs of target programs and oneshot network guesses

    ('-thr', '--threshold', 0.0001, float), # threshold for early stopping minimum improvement

    ('-inf_ps', '--infer_pop_size', 32, int), # population size of inference algorithm
    ('-inf_rs', '--infer_rounds', 8, int), # number of rounds of inference algorithm

    ('-ets', '--etest_size', 1,  int), # Number of test set shapes to use, keep at 1 except when doing eval    
]

# PRETRAINING ARGS
DOM_PT_ARGS = [    
        
    ('-mi', '--max_iters', 100000000, int), # total number of examples

    ('-prp', '--print_per', 100000, int), # how often to print results / make plots
    ('-evp', '--eval_per', 2500000, int), # how often to run evaluation logic
    
    ('-strm', '--stream_mode', 'y', str), # data streaming mode: 'y' -> yes (do stream), 's' -> static (do not stream)
    ('-nw', '--num_write', 10, int), # number of shapes to create visualizations for
    ('-dcm', '--data_conv_mode', 'super_ap', str), # setting for how to convert findEdits output -> training data for edit network

    ('-ts', '--train_size', 100, int), # how many train shapes to use when stream_mode is static
    ('-evs', '--eval_size', 100,  int), # how many validation shapes to use
    
]

# FINETUNING ARGS
DOM_FT_ARGS = [
    ('-ftm', '--ft_mode', 'LEST_ST_WS', str), # PLAD finetuning mode
    ('-evp', '--eval_per', 1, int), # how often to eval
    ('-nw', '--num_write', 20, int), # how many shapes to visualize
    ('-mi', '--max_iters', 10000, int), # max number of finetuning iterations

    # patience settings
    ('-edtp', '--edit_train_patience', 5, int), # edit network
    ('-getp', '--gen_train_patience', 10, int), # gen OS network
    ('-ostp', '--os_train_patience', 10, int), # inf OS network
    ('-itrp', '--iter_patience', 250, int), # finetuning loop

    ('-eps', '--epochs', 100, int), # maximum number of finetuning epochs in a round of finetuning

    # Can set these to control PLAD finetuning distribution, otherwise set automatically
    ('-lest_w', '--lest_weight', 0., float),
    ('-st_w', '--st_weight', 0., float),
    ('-ws_w', '--ws_weight', 0., float),

    # Number of shapes to use during finetuning
    ('-ts', '--train_size', 1000, int), # Training shapes from target data
    ('-evs', '--eval_size', 100,  int), # Validation shapes from target data
    ('-ws', '--ws_train_size', 10000, int), # Generated shapes used during wake-sleep

    # For one-shot network
    ('-os_es_bs', '--os_es_beams', 5, int), # early stopping beams
    ('-os_inf_bs', '--os_inf_beams', 10, int), # outer loop inference beams
    
    ('-dcm', '--data_conv_mode', 'super_ap_5_hold', str), # setting for how to convert findEdits output -> training data for edit network
]

def load_os_net_from_domain(domain, load_path, is_gen_model):
    args = domain.args
    net = OneShotNet(
        domain, is_gen_model=is_gen_model
    )

    if load_path is None:
        if is_gen_model and domain.args.os_gen_load_model_path is not None:
            load_path = domain.args.os_gen_load_model_path
            utils.log_print(f"Loading gen os model from {load_path}", args)
        else:
            assert domain.args.os_load_model_path is not None, 'set os lmp'
            load_path = domain.args.os_load_model_path
            utils.log_print(f"Loading inf os model from {load_path}", args)

    weights = torch.load(
        load_path
    )

    if is_gen_model:
        net.seq_net.load_state_dict({k[8:]:v for k,v in weights.items() if 'seq_net' in k})        
    else:
        net.load_state_dict(weights)
        
    net.to(domain.device)
    net.eval()
    
    return net

class DOMAIN_BASE:
    
    def base_init(self):
        self.obj_name = 'Obj'
        self.device = device        
        
    def vis_metric(self, pixels, gt):
        assert False, 'domain should impl vis metric'

    def load_real_data(self):
        assert False, 'domain should impl load real data'

    def init_metric_val(self):
        assert False

    def get_obj_dir(self):
        assert False

    # early stopping logic using evaluation metric, and threshold
    def should_save(self, cur_val, best_val, thresh):

        if self.get_obj_dir() == 'high':
            thresh_val = cur_val - thresh
        elif self.get_obj_dir() == 'low':
            thresh_val = cur_val + thresh
        else:
            assert False, f'bad obj dir {self.get_obj_dir()}'
            
        if self.comp_metric(thresh_val, best_val):
            return True
        else:
            return False


    # is it better for the evaluation metric to be high or low?
    def comp_metric(self, a, b):

        if self.get_obj_dir() == 'high':
            if a > b:
                return True
            else:
                return False
            
        elif self.get_obj_dir() == 'low':
            if a < b:
                return True
            else:
                return False
        else:
            assert False, f'bad obj dir {self.get_obj_dir()}'    
        
    # what shape do the visual inputs take
    def get_input_shape(self):
        return self.executor.get_input_shape()

    def extra_train_log_info(self):
        return []

    def extra_eval_log_info(self):
        return []

    # pretraining arguments helper function
    def get_pt_args(self, extra_args = []):

        ARGS = utils.mergeArgs(
            extra_args + self.get_cmn_args() + self.get_pt_arg_list(),
            DOM_CMN_ARGS + DOM_PT_ARGS,
        )
        
        self.args = utils.getArgs(ARGS)        

        self.make_executor(self.args)
                
        self.TRAIN_LOG_INFO = DOM_TRAIN_LOG_INFO + self.extra_train_log_info()
        self.EVAL_LOG_INFO = DOM_EVAL_LOG_INFO + self.extra_eval_log_info()
                            
        utils.init_pretrain_run(self.args)        

        self.metric_name = self.args.metric_name
        
        return self.args

    def get_ft_args(self, extra_args = []):

        ARGS = utils.mergeArgs(
            extra_args + self.get_cmn_args() + self.get_ft_arg_list(),
            DOM_CMN_ARGS + DOM_FT_ARGS,
        )

        args = utils.getArgs(ARGS)

        args.infer_path = f"model_output/{args.exp_name}/train_out/"
        args.ws_save_path = f"model_output/{args.exp_name}/ws_out/"

        if 'LEST' in args.ft_mode.split('_'):
            args.lest_weight = 1.0

        if 'ST' in args.ft_mode.split('_'):
            args.st_weight = 1.0

        if 'WS' in args.ft_mode.split('_'):
            args.ws_weight = 1.0
        
        norm = args.lest_weight + args.st_weight  + args.ws_weight

        if norm > 0:
                
            args.lest_weight = args.lest_weight / norm
            args.st_weight = args.st_weight / norm
            args.ws_weight = args.ws_weight / norm
        
        self.args = args

        self.make_executor(self.args)
        
        utils.init_exp_model_run(self.args)

        self.metric_name = args.metric_name
        
        return self.args

    def make_et_to_mel(self, max_len):

        ex = self.executor

        mtl = max(ex.TLang.T2IPC.values()) + 3
        
        L = [
            (('$ACA', '$ACB'), max_len),
            (('$PM', '$TA', '$TM'), mtl),            
            (('$CR', '$TR'), 2),
            (('$CM'), 3), 
        ]

        if self.name == 'layout':
            L.pop(-1)
            
        return L
    
class BaseTargetDataset:

    def get_train_vinput(self):
        return self.vinput
        
    def get_set_size(self, name):
        if name == 'train':
            return self.train_keys.shape[0]
        elif name == 'val':
            return self.val_keys.shape[0]
        elif name == 'test':
            return self.test_keys.shape[0]
        else:
            assert False
        
    def train_eval_iter(self):
        keys = self.train_keys
        yield from self.eval_iter(keys)

    def val_eval_iter(self):
        keys = self.val_keys
        yield from self.eval_iter(keys)

    def test_eval_iter(self):
        keys = self.test_keys        
        yield from self.eval_iter(keys)

    def eval_iter(self, keys):
        assert self.eval_batch_size == 1
        for start in range(
            0, keys.shape[0], self.eval_batch_size
        ):
            binds = keys[start:start+self.eval_batch_size]            
            vinput = self.vinput[binds].float().to(self.device)
                
            yield {
                'bkey': binds.view(1).item(),
                'vdata': vinput,                
            }
            
    def __iter__(self):
        yield from self.val_eval_iter()
