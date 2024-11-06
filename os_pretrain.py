import sys, os, torch, json, time, random, ast, utils, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import device
from copy import deepcopy, copy
from tqdm import tqdm
import train_utils as tru
from tqdm import tqdm
from os_models import OneShotNet

class SynthDataset:
    def __init__(
        self, args, set_name, ex, device
    ):
        
        self.mode = 'train'
        self.args = args
        self.ex = ex
        self.device= device
        
        self.set_name = set_name
                    
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
                
        self.data = []                

        self.iter_num = 0
        self.inds = []
        
        assert args.stream_mode in ('s', 'y')
        if set_name == 'train':
            if args.stream_mode == 'y':
                self.do_stream = True
                self.size = None
            else:
                self.do_stream = False
                self.size = args.train_size
        else:
            self.do_stream = False
            self.size = args.eval_size
            
        self.eval_size = None
        
        if self.size is None:
            return

        with torch.no_grad():
            self.sample_data(self.size, print_info=True)

    def sample_data(self, num, print_info=False):

        if print_info:
            print(f"Preloading Det Data for {self.set_name} ({self.size})")
        
        self.data = self.ex.det_prog_random_sample(
            num,                
            use_pbar = print_info
        )
            
    def __iter__(self):

        if self.mode == 'train':
            if self.do_stream:
                yield from self.stream_iter()
            else:            
                yield from self.train_static_iter()
                            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def make_stream_data(self):

        self.sample_data(
            self.batch_size * self.args.log_period
        )
        
        inds = list(range(len(self.data)))
        random.shuffle(inds)

        self.inds = inds
        
    def stream_iter(self):
        if len(self.inds) == 0:
            with torch.no_grad():
                self.make_stream_data()

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

            
    def train_static_iter(self):

        if len(self.inds) == 0:
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
        
        for start in range(
            0, inds.shape[0], self.args.eval_batch_size
        ):

            assert self.args.eval_batch_size == 1
            
            binds = inds[start:start+self.args.eval_batch_size]

            bdata = [self.data[bi] for bi in binds]
                
            with torch.no_grad():
                batch = self.ex.make_batch(bdata, self.args)                
                
                g_batch = {
                    k: v.to(self.device) for k,v in
                    batch.items() if 'vdata' in k 
                }
                            
            yield g_batch


def get_synth_datasets(domain):
    train_loader = SynthDataset(
        domain.args,
        'train',
        domain.executor,
        domain.device
    )
    
    val_loader = SynthDataset(
        domain.args,
        'val',
        domain.executor,
        domain.device
    )
    
    eval_size = min(
        [
            v for v in
            (domain.args.eval_size, train_loader.size, val_loader.size)
            if v is not None
        ]
    )

    train_loader.eval_size = eval_size
    val_loader.eval_size = eval_size

    train_loader.num_write = min(eval_size-1, domain.args.num_write)
    val_loader.num_write = min(eval_size-1, domain.args.num_write)
    
    return train_loader, val_loader

def get_os_net(
    domain, model_path=None
):

    net = OneShotNet(
        domain,
    )
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

    assert args.pred_mode == 'os'
    net = get_os_net(domain, args.load_model_path)
        
    # synthetic data sampled from the grammar randomly
    train_loader, val_loader = get_synth_datasets(domain)

    target_loader = domain.load_target_dataset()
    
    if args.load_res_path is not None:
        res = json.load(open(args.load_res_path))
        try:
            starting_iter = int(res['eval_iters'][-1])
        except:
            starting_iter = 0            
    else:
        res = {
            'train_plots': {'train':{'iters':[]}, 'val':{'iters':[]}},
            'eval_plots': {'val':{}, 'target': {}},
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
        ('target', target_loader),
    ]

    if args.stream_mode == 's':
        eval_data[0] = ('train', train_loader)
        res['eval_plots']['train'] = res['eval_plots'].pop('val')
        

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
            
            tru.run_eval_epoch(
                args,
                res,
                net,
                eval_data,
                domain.EVAL_LOG_INFO,
                itn,
                do_vis= True
            )

        if itn - last_save >= args.save_per:
            last_save = itn
                                
            utils.save_model(
                net.state_dict(),
                f"{args.outpath}/{args.exp_name}/models/net_CKPT_{save_model_count}.pt"
            )

            save_model_count += 1


