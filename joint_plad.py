from copy import deepcopy
import sys
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import time
import utils
from tqdm import tqdm
import random
import edit_pretrain
import infer_progs

VERBOSE = False

TRAIN_LOG_INFO = [
    ('Train Loss', 'train_loss', 'nc'),
    ('Val Loss', 'val_loss', 'nc'),    
]

class DataGen:
    def __init__(
        self,
        domain,
        inf_ex,
        train_pbest,
        target_vinput,
        gen_data,
    ):

        args = domain.args

        self.args = args
        self.domain = domain
        self.device = domain.device
        self.batch_size = args.os_batch_size
                
        self.target_vinput = target_vinput
                
        self.keys = []
        self.data = []
            
        for keys, (_, d) in train_pbest.data.items():
            
            self.keys.append(keys)
            self.data.append(d)
               
        self.gen_data = gen_data

        self.st_weight = self.args.st_weight
        self.lest_weight = self.args.lest_weight
        self.ws_weight = self.args.ws_weight
        
        assert len(self.gen_data) > 0
            
        self.train_size = len(self.keys) + len(self.gen_data)

        with torch.no_grad():
            self.preload_data(inf_ex)

    def preload_data(self, inf_ex):

        self.lest_data = {}
        self.st_data = {}
        self.ws_data = {}            
        
        if self.lest_weight > 0.:
            print("Pre loading LEST data")
            self.preload_mode(
                inf_ex,
                self.lest_data,
                self.data,
                None,
                None
            )

            
        if self.ws_weight > 0:
            print("Pre loading WS data")
            self.preload_mode(
                inf_ex,
                self.ws_data,
                self.gen_data,
                None,
                None
            )

        else:
            self.gen_data = []
            
        if self.st_weight > 0.:
            print("Pre loading ST data")
            self.preload_mode(
                inf_ex,
                self.st_data,
                self.data,
                self.keys,
                self.target_vinput
            )


        if self.lest_weight <= 0. and self.st_weight <= 0.:
            self.data = []

    def preload_mode(self, inf_ex, sdata, idata, ikeys, vdata):

        for ind in tqdm(list(range(len(idata)))):
            d = [idata[ind]]

            b = inf_ex.make_os_batch(d, self.args)
            
            for k,v in b.items():
                if k not in sdata:
                    sdata[k] = []
                sdata[k].append(v[0])

        sdata.update({
            k:torch.stack(V,dim=0) for k,V in sdata.items()
        })

        if vdata is None:
            return

        
        sdata['vdata'] = torch.zeros(
            sdata['vdata'].shape,                
            device=torch.device('cpu')
        )     
        
        for i,ik in tqdm(list(enumerate(ikeys))):
            t_ind = ik
            pixels = vdata[t_ind]
            
            try:
                sdata['vdata'][i] = pixels.cpu()
            except:
                if len(pixels.shape) == 2:
                    sdata['vdata'][i,:,:,0] = pixels.cpu()        
                elif len(pixels.shape) == 3:
                    sdata['vdata'][i,:,:,:,0] = pixels.cpu()
                else:
                    assert False
                    
    def sample_plad_mode(self):
        comb_modes = ['lest', 'st', 'ws']
        
        comb_weights = [self.lest_weight, self.st_weight, self.ws_weight]

        return np.random.choice(
            comb_modes,
            p = comb_weights
        )
        
    def train_iter(self):
        tar_inds = list(range(len(self.data)))
        random.shuffle(tar_inds)

        gen_inds = list(range(len(self.gen_data)))
        random.shuffle(gen_inds)
        
        while len(tar_inds) > 0 or len(gen_inds) > 0:
            
            pmode = self.sample_plad_mode()

            if pmode == 'ws':
                if len(gen_inds) <= 0:
                    continue
                else:
                    binds = torch.tensor(gen_inds[:self.batch_size])
                    gen_inds = gen_inds[self.batch_size:]
                    yield from self.mode_batch(
                        self.ws_data,
                        binds
                    )
            
            elif pmode in ('st', 'lest'):
                if len(tar_inds) == 0:
                    continue
                else:                    
                    binds = torch.tensor(tar_inds[:self.batch_size])
                    tar_inds = tar_inds[self.batch_size:]            
                
                    if pmode == 'lest':
                        yield from self.mode_batch(
                            self.lest_data,
                            binds
                        )
                        
                    elif pmode == 'st':
                        yield from self.mode_batch(
                            self.st_data,
                            binds
                        )
                    

    def mode_batch(self, data, binds):
        batch = {
            k: V[binds].to(self.device) for k,V in data.items()
        }

        yield batch
        

def train_os_plad(domain, os_inf_net, gen_data, target_data, train_pbest):

    args = domain.args

    dargs = deepcopy(args)
    dargs.infer_is_mode = 'beam'
    os_inf_net.beams = args.os_es_beams
    
    path = args.infer_path
    
    epochs = args.epochs
        
    train_gen = DataGen(
        domain,
        os_inf_net.ex,
        train_pbest,
        target_data.get_train_vinput(),
        gen_data
    )

    val_gen = target_data.val_eval_iter

    opt = optim.Adam(
        os_inf_net.parameters(),
        lr=args.lr
    )

    best_test_metric = domain.init_metric_val()

    utils.save_model(os_inf_net.state_dict(), f"{path}/best_os_dict.pt")

    patience = args.os_train_patience
    num_worse = 0
    eval_count = 0
        
    for epoch in range(epochs):
        start = time.time()
        losses = []
        os_inf_net.train()
        
        for batch in train_gen.train_iter():

            loss, _ = os_inf_net.model_train_batch(batch)
                                        
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())            
            
        eval_count += 1
        
        if (eval_count % args.eval_per) != 0:
            num_worse += 1
            end = time.time()
            utils.log_print(
                f"Epoch {epoch}/{epochs} => TRAIN ONLY "
                f"|  LOSS : {round(torch.tensor(losses).mean().item(), 3)} | {end-start}"
                , args
            )
            continue        
                        
        os_inf_net.eval()
        eval_res = {'errors': 0., 'count': 0.}

        with torch.no_grad():
            for batch in val_gen():

                key = batch['bkey']
                vinput = batch['vdata']

                eres, einfo = infer_progs.run_inference(
                    os_inf_net,
                    None,
                    dargs,
                    vinput,
                )

                if eres is None:
                    assert einfo is None
                    eval_res['errors'] += 1
                    continue

                for k,v in eres.items():
                    if k not in eval_res:
                        eval_res[k] = 0.
                    eval_res[k] += v
                                
        results = utils.print_results(
            infer_progs.FT_EVAL_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )
        
        ## EVAL
        if domain.obj_name not in results:
            METRIC = 0.
        else:        
            METRIC = results[domain.obj_name]
            
        ERR = eval_res['errors']

        # Always save network, if we improved the metric
        if domain.should_save(METRIC, best_test_metric, 0.0):
            utils.save_model(os_inf_net.state_dict(), f"{path}/best_os_dict.pt")

        # Only reset count if we passed the threshold
        if not domain.should_save(METRIC, best_test_metric, args.threshold):
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
                                
        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Obj : {round(METRIC, 3)}[{round(ERR,2)}] "
            f"|  LOSS : {round(torch.tensor(losses).mean().item(), 3)} | {end-start}"
            ,args
        )

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            utils.log_print("Early stopping inner loop", args)
            os_inf_net.load_state_dict(torch.load(f"{path}/best_os_dict.pt"))
            return epoch + 1

    return epochs
    

class EditData:
    def __init__(self, domain, edit_ex, prog_pairs, max_inds):

        self.domain = domain
        self.do_split = max_inds is None
        self.ex = edit_ex
        args = domain.args
        self.args = args
        self.device = domain.device        
        self.batch_size = args.edit_batch_size

        self.data_conv_mode = args.data_conv_mode

        if 'super_ap' in self.data_conv_mode:
            self.hold_super_sample = 'hold' in self.data_conv_mode

            if self.data_conv_mode.count('_') > 1:
                MASN = int(self.data_conv_mode.split('_')[2])
                domain.prog_diff.set_gparams(A=MASN)

            with torch.no_grad():
                pair_data = edit_pretrain.convert_all_prog_pairs_to_data(
                    domain, prog_pairs
                )
                self.data, inf = self.conv_pair_data(domain, pair_data)

        else:
            assert False, f'bad data conv mode {self.data_conv_mode}'

        
        if self.do_split:
            assert max_inds is None
            self.inds_list = []
        else:
            assert max_inds is not None
            all_inds = list(range(len(self.data)))
            random.shuffle(all_inds)
            self.inds = all_inds[:max_inds]
            
        utils.log_print(f"Found {len(self.data)} edit pairs from {len(prog_pairs)} prog pairs | Err: {inf['errors']} | Time: {inf['time']}", args)

    def conv_pair_data(self, domain, pair_data):

        data = []

        ex = domain.executor

        pred_mode = domain.args.pred_mode

        errors = 0
        count = 0

        T = time.time()
        
        for pdata in tqdm(pair_data):
            count += 1    

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
                                            
                if self.hold_super_sample:
                    data.append({
                        'tar_vdata': tar_vdata,
                        'corr_tokens': corr_tokens,
                        'corr_vdata': corr_vdata,
                        'hold_info':  pcd['eoi']
                    })                    
                else:
                    eoi = random.choice(pcd['eoi'])
                    eot, eol, eos = eoi
                    data.append({
                        'tar_vdata': tar_vdata,
                        'corr_tokens': corr_tokens,
                        'corr_vdata': corr_vdata,
                        'edit_ps_info':  eoi,
                        'edit_tl_info': (eot, eol)
                    })
                                            
        info = {
            'errors': errors,
            'time': round(time.time() - T)
        }
        
        return data, info
        
    def train_iter(self):

        if self.do_split:

            if len(self.inds_list) == 0:
            
                all_inds = list(range(len(self.data)))
                random.shuffle(all_inds)

                split_num = int(len(all_inds) * .1)
                while len(all_inds) >= self.batch_size:
                    self.inds_list.append(all_inds[:split_num])
                    all_inds = all_inds[split_num:]            

            inds = self.inds_list.pop(0)
        else:
            inds = self.inds
                                
        while len(inds) >= self.batch_size:
            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]
            
            bdata = [self.data[bi] for bi in binds]            

            if self.hold_super_sample:
                hold_data = []
                for d in bdata:
                    eoi = random.choice(d['hold_info'])
                    eot, eol, eos = eoi
                    
                    hold_data.append({
                        'tar_vdata': d['tar_vdata'],
                        'corr_tokens': d['corr_tokens'],
                        'corr_vdata': d['corr_vdata'],
                        'edit_ps_info':  eoi,
                        'edit_tl_info': (eot, eol)
                    })
                    
                bdata = hold_data
                
            with torch.no_grad():

                batch = self.ex.make_batch(bdata, self.args)
                g_batch = {
                    k: v.to(self.device) for k,v in
                    batch.items()
                }
                
            yield g_batch

            
def make_prog_pairs(domain, os_inf_net, progs):
    inf_ex = os_inf_net.ex
    edit_ex = domain.executor
    
    args = domain.args

    batch_size = args.edit_batch_size

    NIP = len(progs)
    
    data = []

    pbar = tqdm(total=len(progs))

    T = time.time()
    
    while len(progs) > 0:
        
        synth_progs = progs[:batch_size]
        progs = progs[batch_size:]

        cc = len(synth_progs)
        
        synth_vdata = []
        for sp in synth_progs:            
            synth_vdata.append(inf_ex.execute(' '.join(sp)))

        inp_vdata = torch.stack(synth_vdata,dim=0)
        samples = os_inf_net.eval_batch_sample_prog(inp_vdata)
        
        for i, _tar_tokens in enumerate(synth_progs):
            if i not in samples or samples[i] is None:
                continue            

            
            if _tar_tokens[-1] != 'END':
                tar_tokens = _tar_tokens + ['END']
            else:
                tar_tokens = _tar_tokens
                
            if samples[i][-1] != 'END':
                corr_tokens = samples[i] + ['END']
            else:
                corr_tokens = samples[i]
                    
            try:
                PNF, EI, edit_ops = domain.prog_diff.get_edit_info(
                    corr_tokens, tar_tokens
                )
            except Exception as e:
                print(f"Failed get info with {e}")
                continue
            
            data.append((
                corr_tokens,
                tar_tokens,
                (PNF, EI, edit_ops),
                'dummy'
            ))
            
        pbar.update(cc)

    T = round(time.time() - T, 1)
    
    utils.log_print(f"From {NIP} progs found {len(data)} valid pairs in {T} seconds", args)
        
    return data
    
def train_edit_plad(domain, edit_net, gen_data, os_inf_net):

    args = domain.args
    path = args.infer_path

    num_train = int(len(gen_data) * 0.9)

    train_progs = gen_data[:num_train]
    val_progs = gen_data[num_train:]
    
    with torch.no_grad():
                
        print("Making training prog pairs")
        train_prog_pairs = make_prog_pairs(domain, os_inf_net, train_progs)
        print("Making val prog pairs")
        val_prog_pairs = make_prog_pairs(domain, os_inf_net, val_progs)

    TrainData = EditData(
        domain,
        edit_net.ex,
        train_prog_pairs,
        max_inds = None
    )
    ValData = EditData(
        domain,
        edit_net.ex,
        val_prog_pairs,
        max_inds = args.edit_batch_size * 20,
    )

    opt = optim.Adam(
        edit_net.parameters(),
        lr=args.lr
    )

    best_test_metric = 100.

    utils.save_model(edit_net.state_dict(), f"{path}/best_edit_dict.pt")

    patience = args.edit_train_patience
    num_worse = 0

    epochs = args.epochs
    
    for epoch in range(epochs):
        start = time.time()
        train_losses = []
        val_losses = []

        edit_net.train()
        
        for batch in TrainData.train_iter():
            loss, _ = edit_net.model_train_batch(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())            

        edit_net.eval()

        with torch.no_grad():
            for batch in ValData.train_iter():
                loss, _ = edit_net.model_train_batch(batch)
                val_losses.append(loss.item())

        eval_res = {
            'train_loss': torch.tensor(train_losses).float().mean().item(),
            'val_loss': torch.tensor(val_losses).float().mean().item(),
            'nc': 1.0
        }

        results = utils.print_results(
            TRAIN_LOG_INFO,
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
            utils.save_model(edit_net.state_dict(), f"{path}/best_edit_dict.pt")
        
        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Train / Val : {round(eval_res['train_loss'], 3)} / {round(eval_res['val_loss'], 3)} "
            f"| {end-start}"
            ,args
        )

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            utils.log_print("Early stopping inner loop", args)
            edit_net.load_state_dict(torch.load(f"{path}/best_edit_dict.pt"))
            return epoch + 1
        
    return epochs
