import sys, os, torch, json, time, utils
import math
from tqdm import tqdm

def model_train_batch(batch, net, opt):

    loss, br = net.model_train_batch(batch)
               
    if opt is not None:
        if net.acc_count == 0:
            opt.zero_grad()            

        aloss = loss / (net.acc_period * 1.)
        aloss.backward()
        net.acc_count += 1
        
        if net.acc_count == net.acc_period:
            opt.step()
            net.acc_count = 0
        
    return br

def model_train(loader, net, opt):
    
    if opt is None:
        net.eval()
        log_period = 1e20
    else:
        net.train()
        if 'log_period' in net.__dict__:
            log_period = net.acc_period * net.log_period
        else:
            log_period = 1e20
    
    ep_result = {}
    bc = 0.

    for batch in loader:
        bc += 1.

        if bc > log_period:
            break

        if isinstance(batch, dict):
            if 'vdata' in batch:
                bk = 'vdata'
            else:
                assert False, f'batch missing vdata {batch.keys()}'
        else:
            bk = 0

        if 'iter_num' in loader.__dict__:
            loader.iter_num += batch[bk].shape[0]
        
        batch_result = model_train_batch(batch, net, opt)
        for key in batch_result:                        
            if key not in ep_result:                    
                ep_result[key] = batch_result[key]
            else:
                ep_result[key] += batch_result[key]

                
    ep_result['batch_count'] = bc
    
    return ep_result


def run_train_epoch(
    args,
    res,
    net,
    opt,
    train_loader,
    val_loader,
    LOG_INFO,
    do_print,
    epoch = None
):
    
    json.dump(res, open(f"{args.outpath}/{args.exp_name}/res.json" ,'w'))

    t = time.time()

    if epoch is None:
        itn = train_loader.iter_num
        if do_print:
            utils.log_print(f"\nBatch Iter {itn}:", args)
    else:
        itn = epoch
        if do_print:
            utils.log_print(f"\nEpoch {itn}:", args)
        

    if train_loader is not None:
        train_loader.mode = 'train'

    if val_loader is not None:
        val_loader.mode = 'train'

    train_result = model_train(
        train_loader,
        net,
        opt
    )
    if epoch is None:
        train_itn = train_loader.iter_num
        slice_name = 'iters'
    else:
        train_itn = epoch
        slice_name = 'epochs'
        
    utils.update_res(
        LOG_INFO,
        res['train_plots']['train'],
        train_result,
        slice_name,
        train_itn
    )    
    
    if do_print:            
        
        with torch.no_grad():
            val_result = model_train(
                val_loader,
                net,
                None,
            )

        utils.update_res(
            LOG_INFO,
            res['train_plots']['val'],
            val_result,
            slice_name,
            train_itn,
        )    
                        
        utils.log_print(
            f"Train results: ", args
        )
            
        utils.pre_print_results(
            LOG_INFO,
            train_result,
            args,
        )

        utils.log_print(
            f"Val results: ", args
        )
            
        utils.pre_print_results(
            LOG_INFO,
            val_result,
            args,
        )
                 
        utils.make_info_plots(
            LOG_INFO,
            res['train_plots'],
            slice_name,
            'train',
            args,
        )
            
        utils.log_print(
            f"    Time = {time.time() - t}",
            args
        )



def run_eval_epoch(
    args,
    res,
    net,
    eval_data,
    EVAL_LOG_INFO,
    itn,
    do_vis=False    
):
                
    with torch.no_grad():
        
        net.eval()        
                    
        t = time.time()                

        eval_results = {}
        for key, loader in eval_data:

            if loader.mode == 'train':
                loader.mode = 'eval'

            if do_vis:
                net.vis_mode = (key, itn)
            else:
                net.vis_mode = None

            net.init_vis_logic()
                
            eval_results[key] = model_eval(
                args,
                loader,
                net,
            )

            if do_vis:
                net.save_vis_logic()
                    
            utils.log_print(
                f"Evaluation {key} set results:",
                args
            )

            utils.pre_print_results(
                EVAL_LOG_INFO,
                eval_results[key],
                args
            )
                        
        utils.log_print(f"Eval Time = {time.time() - t}", args)

        res['eval_iters'].append(itn)
                
        utils.make_comp_plots(
            EVAL_LOG_INFO,
            eval_results,            
            res['eval_plots'],
            res['eval_iters'],
            args,
            'eval'
        )

        return eval_results

def model_eval(
    args,
    loader,
    net,
):

    res = {}

    pbar = tqdm(total=math.ceil(loader.eval_size / loader.eval_batch_size))    
    
    for count, batch in enumerate(loader):

        _res = net.model_eval_fn(
            batch
        )

        for k,v in _res.items():                        
            if k not in res:
                if isinstance(v, list):
                    res[k] = [v]
                else:
                    res[k] = v
            else:
                if isinstance(v, list):
                    res[k].append(v)
                else:
                    res[k] += v
                    
        pbar.update(1)
                        
    res['nc'] = 1
    pbar.close()
    
    return res
