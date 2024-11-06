import sys, os, torch, json, time, random, ast, utils, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import device
from copy import deepcopy, copy
from tqdm import tqdm

import train_utils as tru
from edit_models import EditNet

VERBOSE = False

def make_prog_pairs(domain, set_name, num_pairs):
    
    ex = domain.executor
    args = domain.args        
    batch_size = args.batch_size

    data = []
    pbar = tqdm(total=num_pairs)

    while len(data) < num_pairs:
        
        synth_progs, synth_vdata = ex.det_prog_random_sample(batch_size, use_pbar=False, ret_data=True)
                
        inp_vdata = torch.stack(synth_vdata,dim=0)
        samples = domain.os_net.eval_batch_sample_prog(inp_vdata)
            
        for i, tar_tokens in enumerate(synth_progs):
            if i not in samples or samples[i] is None:
                continue            

            try:
                if 'END' not in samples[i][-1]:
                    assert domain.name == 'layout'
                    corr_tokens = samples[i] + ['END']
                else:
                    corr_tokens = samples[i]
                        
                PNF, EI, edit_ops = domain.prog_diff.get_edit_info(corr_tokens, tar_tokens)
            except Exception as e:
                print(f"Failed get info with {e}")
                continue        
            
            data.append((
                corr_tokens,
                tar_tokens,
                (PNF, EI, edit_ops),
                'edit'
            ))
            
            pbar.update(1)

    return data

    

def convert_to_eval_data(
    domain, eval_prog_pairs
):
    eval_data = []

    ex = domain.executor
    
    print("Loading eval data")

    with torch.no_grad():
        for start_tokens, tar_tokens, _, etype in tqdm(eval_prog_pairs):
        
            tar_vdata = ex.execute(' '.join(tar_tokens))            
               
            eval_data.append({
                'start_tokens': start_tokens,
                'tar_vdata': tar_vdata,
                'edit_type': etype,                
            })

            
    return eval_data


def convert_all_prog_pairs_to_data(domain, prog_pairs):
    pair_data = []
        
    ex = domain.executor
    
    print("Loading all pair data")
    start_time = time.time()
    errors = 0
    num_pairs = 0
    
    for start_tokens, tar_tokens, (PNF, EI, edit_ops), etype in tqdm(prog_pairs):
        
        try:
            conv_data = domain.prog_diff.conv_to_data(
                'super_ap', 
                start_tokens, tar_tokens, PNF, EI, edit_ops
            )
        except Exception as e:
            errors += 1
            continue        

        pair_data.append({
            'start_tokens': start_tokens,
            'tar_tokens': tar_tokens,
            'conv_data': conv_data,
            'edit_type': etype,
        })
        
        num_pairs += len(conv_data)
        
    utils.log_print(f"Sampled {num_pairs} data pairs in {round(time.time() - start_time)} seconds with {errors} errors", domain.args)
    
    return pair_data


class SynthDataset:
    def __init__(
        self, domain, set_name, 
    ):
        
        args = domain.args

        self.os_net = domain.os_net
        
        self.mode = 'train'
        self.args = domain.args
        self.domain = domain
        self.ex = domain.executor
        self.device= domain.device
        
        self.set_name = set_name
                    
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        
        self.iter_num = 0
        self.inds = []

        self.data = []
        self.eval_data = []
            
        with torch.no_grad():
            self.make_data_pairs()
                                
        self.size = len(self.data)            
        self.eval_size = len(self.eval_data)

        print(f"Set {set_name} : Train {self.size} | Eval {self.eval_size}")

    def make_data_pairs(self):
        assert len(self.data) == 0

        cache_path = self.args.synth_pair_cache_path
        print(f"Loading cache from {cache_path}")

        args = self.args
        
        CACHE = torch.load(cache_path)

        if self.set_name == 'train':
            size = args.train_size
        elif self.set_name == 'val':
            size = args.eval_size
        else:
            assert False
                        
        prog_pairs = CACHE[f'{self.set_name}_prog_pairs']
                
        self.data_conv_mode = args.data_conv_mode

        assert len(self.eval_data) == 0
        self.eval_data = convert_to_eval_data(
            self.domain, prog_pairs[:args.eval_size]
        )

        self.pair_data = convert_all_prog_pairs_to_data(
            self.domain, prog_pairs[:size]
        )
        self.pair_data_q = []
        with torch.no_grad():
            self.load_next_pair_data()             
                
        self.inds = list(range(len(self.data)))
        random.shuffle(self.inds)
        
        
    def load_next_pair_data(self):
                
        print("Loading next pair data in all pair mode")
        self.data = []

        ex = self.domain.executor

        pred_mode = self.domain.args.pred_mode
        
        if len(self.pair_data_q) == 0:
            self.pair_data_q = torch.randperm(len(self.pair_data)).tolist()
            
        errors = 0
        count = 0
        pbar = tqdm(total = min(len(self.pair_data_q), 10000))
        while len(self.pair_data_q) > 0 and count <= 10000:
            count += 1
            
            pdata = self.pair_data[self.pair_data_q.pop(0)]

            tar_tokens = pdata['tar_tokens']

            try:
                tar_vdata =  ex.execute(' '.join(tar_tokens)).cpu()
            except Exception as e:
                if VERBOSE:
                    print(f"unexpectedly couldn't execute tar with {e}")
                errors += 1
                continue
            
            for pcd in pdata['conv_data']:
                corr_tokens = pcd['corr_tokens']
                try:
                    corr_vdata =  ex.execute(' '.join(corr_tokens)).cpu()
                except Exception as e:
                    if VERBOSE:
                        print(f"unexpectedly couldn't execute corr with {e}")
                    errors += 1
                    continue

                assert pred_mode == 'edit'
                eoi = random.choice(pcd['eoi'])
                eot, eol, eos = eoi
                                    
                self.data.append({
                    'tar_vdata': tar_vdata,
                    'corr_tokens': corr_tokens,
                    'corr_vdata': corr_vdata,
                    'edit_ps_info':  eoi,
                    'edit_tl_info': (eot, eol)
                })
                    
            pbar.update(1)
        pbar.close()
        print(f"Added {len(self.data)} pairs to data with {errors} errors" )
            
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_static_iter()
                            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def train_static_iter(self):

        if len(self.inds) == 0:

            if self.set_name == 'train':                
                with torch.no_grad():                        
                    self.load_next_pair_data()                        

                    
            self.inds = list(range(len(self.data)))
            random.shuffle(self.inds)
                        
        while len(self.inds) > 0:
            binds = self.inds[:self.batch_size]
            self.inds = self.inds[self.batch_size:]

            bdata = [self.data[bi] for bi in binds]            
            
            with torch.no_grad():

                batch = self.ex.make_batch(bdata, self.args)

                g_batch = {
                    k: v.to(self.device) for k,v in
                    batch.items()
                }
                
            yield g_batch
    
    def eval_iter(self):
        inds = torch.arange(len(self.data[:self.eval_size]))
        assert self.args.eval_batch_size == 1
        
        for start in range(
            0, inds.shape[0], self.args.eval_batch_size
        ):
                        
            bind = inds[start]
            
            bdata = self.eval_data[bind]
            args = self.args
            
            if start < self.num_write:
                name = f'{args.outpath}/{args.exp_name}/vis/{self.set_name}_shape_{bind}_itn_{self.iter_num}_{bdata["edit_type"]}'
            else:
                name = None
                
            g_batch = {
                'vdata': bdata['tar_vdata'].to(self.device),
                'name': name,
                'os_net': self.os_net
            }
            
            yield g_batch


def get_synth_datasets(domain):
    train_loader = SynthDataset(
        domain,
        'train',
    )
    
    val_loader = SynthDataset(
        domain,
        'val',        
    )    
    
    eval_size = min(
        [
            v for v in
            (domain.args.eval_size, train_loader.eval_size, val_loader.eval_size)
            if v is not None
        ]
    )

    train_loader.eval_size = eval_size
    val_loader.eval_size = eval_size

    train_loader.num_write = min(eval_size-1, domain.args.num_write)
    val_loader.num_write = min(eval_size-1, domain.args.num_write)
    
    return train_loader, val_loader

def get_edit_net(
    domain, model_path=None
):

    net = EditNet(domain)
        
    net.acc_count = 0
    net.acc_period = domain.args.acc_period
    net.log_period = domain.args.log_period
    if model_path is not None:
        print(f"Loading from {model_path}")
        net.load_state_dict(
            torch.load(model_path)
        )

    net.to(device)
    return net
    
def pretrain(domain):
    args = domain.get_pt_args()

    domain.load_prog_diff()    
    domain.load_oneshot_net()

    cache_path = args.synth_pair_cache_path
    assert cache_path is not None
    if cache_path.split('/')[-1] not in os.listdir('/'.join(cache_path.split('/')[:-1])):
    
        print(f"Saving cache at {cache_path}")
        with torch.no_grad():
            train_prog_pairs = make_prog_pairs(domain, 'train', args.train_size)
            val_prog_pairs = make_prog_pairs(domain, 'val', args.eval_size)

        torch.save({
            'train_prog_pairs': train_prog_pairs,
            'val_prog_pairs': val_prog_pairs
        }, cache_path)        
                                           
    train_loader, val_loader = get_synth_datasets(
        domain,
    )
        
    net = get_edit_net(domain, args.load_model_path)
    net.prog_diff = domain.prog_diff
    
    if args.load_res_path is not None:
        res = json.load(open(args.load_res_path))
        try:
            starting_iter = int(res['eval_iters'][-1])
        except:
            starting_iter = 0            
    else:
        res = {
            'train_plots': {'train':{'iters':[]}, 'val':{'iters':[]}},
            'eval_plots': {'val':{}},
            'eval_iters': []            
        }
        starting_iter = 0
        
    train_loader.iter_num = starting_iter
    last_print = starting_iter
    last_eval = starting_iter
    last_save = starting_iter

    if args.save_per is None:
        args.save_per = args.eval_per
        
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )

    save_model_count = 0

    eval_data = [
        ('val', val_loader),
    ]
    
    print("Starting Training")
    pbar = None

    while True:
        
        if pbar is None:
            pbar = tqdm(total=args.print_per)
            
        itn = train_loader.iter_num

        if itn > args.max_iters:
            break
        
        if itn - last_print >= args.print_per:
            do_print = True
            last_print = itn
            pbar.close()
            pbar = None
        else:
            do_print = False

        tru.run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            val_loader,
            domain.TRAIN_LOG_INFO,
            do_print,
        )
        
        if pbar is not None:
            pbar.update(train_loader.iter_num-itn)

        if itn - last_eval >= args.eval_per:                    
            last_eval = itn

            val_loader.iter_num = itn
            
            tru.run_eval_epoch(
                args,
                res,
                net,
                eval_data,
                domain.EVAL_LOG_INFO,
                itn,
            )        

        if itn - last_save >= args.save_per:
            last_save = itn
                
            utils.save_model(
                net.state_dict(),
                f"{args.outpath}/{args.exp_name}/models/net_CKPT_{save_model_count}.pt"
            )

            save_model_count += 1
