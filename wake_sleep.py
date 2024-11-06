import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import time
import utils
from tqdm import tqdm
import random

WS_TRAIN_LOG_INFO = [
    ('Train Loss', 'train_loss', 'nc'),
    ('Val Loss', 'val_loss', 'nc'),    
]

class WSDataGen:
    def __init__(
        self,
        domain,
        gen_ex,
        pbest,
    ):

        args = domain.args

        self.args = args
        self.domain = domain
        self.batch_size = args.os_batch_size
        
        self.keys = []
        self.data = []

        for keys, (_, d) in pbest.data.items():            
            self.keys.append(keys)
            self.data.append(d)
        
        self.train_size = len(self.keys)

        with torch.no_grad():
            self.ex_data = gen_ex.make_os_batch(self.data, self.args)
        
    def train_iter(self):
        inds = torch.randperm(len(self.data))

        while len(inds) > 0:

            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]

            with torch.no_grad():
                g_batch = {
                    k: v[binds].to(self.domain.device) for k,v in
                    self.ex_data.items() if k != 'vdata'
                }
                g_batch['vdata'] = len(binds)

            yield g_batch

def make_ws_gens(
    domain, gen_model, train_pbest, val_pbest
):

    gen_model, ge = train_gen_model(
        domain, gen_model, train_pbest, val_pbest
    )

    with torch.no_grad():
        print("Sampling gen model")

        gen_data = sample_ws_gens(domain, gen_model)
        
        images = []
        for g in gen_data[:50]:
            img = gen_model.ex.execute(' '.join(g))
            images.append(img)
            
        num_rows = 5

        if domain.args.num_write > 0:
            try:                
                domain.executor.render_group(
                    images,
                    name=f'{domain.args.ws_save_path}/drm_render_{gen_model.gen_epoch}',
                    rows=num_rows
                )
            except Exception as e:
                utils.log_print("Failed to save dream images with {e}", domain.args)
                
    gen_model.gen_epoch += 1
    
    return gen_model, gen_data, ge


def train_gen_model(
    domain, gen_model, train_pbest, val_pbest
):
    
    args = domain.args

    path = args.ws_save_path
    
    epochs = args.epochs

    train_gen = WSDataGen(
        domain,
        gen_model.ex,
        train_pbest,
    )

    val_gen = WSDataGen(
        domain,
        gen_model.ex,
        val_pbest
    )

    opt = optim.Adam(
        gen_model.parameters(),
        lr=args.lr
    )

    best_test_metric = 100.

    utils.save_model(gen_model.state_dict(), f"{path}/best_gen_dict.pt")

    patience = args.gen_train_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        train_losses = []
        val_losses = []
        
        gen_model.train()
        
        for batch in train_gen.train_iter():
            loss, _ = gen_model.model_train_batch(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())            

        gen_model.eval()
        with torch.no_grad():
            for batch in val_gen.train_iter():
                loss, _ = gen_model.model_train_batch(batch)
                val_losses.append(loss.item())

        eval_res = {
            'train_loss': torch.tensor(train_losses).float().mean().item(),
            'val_loss': torch.tensor(val_losses).float().mean().item(),
            'nc': 1.0
        }

        results = utils.print_results(
            WS_TRAIN_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )
        
        ## EVAL

        METRIC = eval_res['val_loss']
            
        if METRIC >= best_test_metric:
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
            utils.save_model(gen_model.state_dict(), f"{path}/best_gen_dict.pt")

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            gen_model.load_state_dict(torch.load(f"{path}/best_gen_dict.pt"))
            break

        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Train / Val : {round(eval_res['train_loss'], 3)} / {round(eval_res['val_loss'], 3)} "
            f"| {end-start}"
            ,args
        )
        
    return gen_model, epochs


def sample_ws_gens(domain, gen_model):

    gen_data = []
    pbar = tqdm(total = domain.args.ws_train_size)
    while len(gen_data) < domain.args.ws_train_size:
        batch_size = domain.args.os_batch_size            
        try:
            samples = gen_model.gen_sample_progs(
                batch_size
            )
        except Exception as e:
            utils.log_print(f"FAILED WAKE SLEEP batch with {e}", domain.args)
            continue
        
        gen_data += samples
        pbar.update(len(samples))

    return gen_data[:domain.args.ws_train_size]
