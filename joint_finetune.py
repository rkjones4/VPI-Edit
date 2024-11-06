import time
import dill
import utils
import torch
import json
from copy import deepcopy
import joint_plad
import wake_sleep
from edit_models import EditNet
import infer_progs

class BestPrograms:
    def __init__(self, domain, name):
        self.domain = domain
        self.data = {}
        self.name = name
        
        IMV = domain.init_metric_val()
        
    def update(self, key, program, mval):                                
        
        if key in self.data and (not self.domain.comp_metric(
            mval, self.data[key][0]
        )):
            return
                
        self.data[key] = (mval, program)

        
class Logger:
    def __init__(self, domain):
        self.Round = 0
        self.domain = domain

        if domain.args.load_res_path is not None:
            res = json.load(open(domain.args.load_res_path))
            self.inf_epochs = res.pop('epochs')
            self.gen_epochs = res.pop('gen_epochs')
            self.edit_epochs = res.pop('edit_epochs')

            self.total_time = res.pop('total_time')
            self.best_val = max(res['val']['Obj'])
            self.best_epoch = max(self.inf_epochs) + 1
            self.res = res

            self.inf_epochs += [self.best_epoch]
            self.gen_epochs += [self.best_epoch]
            self.edit_epochs += [self.best_epoch]
            
            return
        
        self.res = {
            'train': {},
            'val': {},
        }
        
        self.inf_epochs = [0]
        self.gen_epochs = [0]
        self.edit_epochs = [0]
        self.total_time = [0]
        
        self.best_val = domain.init_metric_val()
        self.best_epoch = 0
        

    def log(self, iter_res, os_inf_net, edit_net):
        for sname, svals in iter_res.items():
            for mname, mval in svals.items():
                if mname not in self.res[sname]:
                    self.res[sname][mname] = []
                self.res[sname][mname].append(mval)

        json.dump(
            {**self.res, **{
                'epochs':self.inf_epochs,
                'gen_epochs':self.gen_epochs,
                'edit_epochs': self.edit_epochs,
                'total_time': self.total_time
            }},
            open(f"model_output/{self.domain.args.exp_name}/res.json" ,'w')
        )

        utils.make_joint_plots(
            self.res, self.inf_epochs, self.domain.args
        )        

        if self.domain.should_save(iter_res['val']['Obj'], self.best_val, self.domain.args.threshold):
            utils.log_print("Replacing best model", self.domain.args)
            self.best_val = iter_res['val']['Obj']
            self.best_epoch = self.inf_epochs[-1]                    
            utils.save_model(os_inf_net.state_dict(), f"model_output/{self.domain.args.exp_name}/os_inf_net.pt")
            if edit_net is not None:
                utils.save_model(edit_net.state_dict(), f"model_output/{self.domain.args.exp_name}/edit_net.pt")
            
    def check_early_stop(self):
        if self.inf_epochs[-1] >= self.domain.args.max_iters:
            return True
        utils.log_print(f"ROUND {self.Round} (Inf Epochs: {self.inf_epochs[-1]})", self.domain.args)
            
    def add_epochs(self, ed_ie, os_ie, ge, tt):
        self.inf_epochs.append(os_ie + self.inf_epochs[-1])
        self.edit_epochs.append(ed_ie + self.edit_epochs[-1])
        self.gen_epochs.append(ge + self.gen_epochs[-1])
        self.total_time.append(tt)
        self.Round += 1

                       
def get_edit_net(
    domain, model_path
):

    args = domain.args
    if args.pred_mode == 'edit':
        edit_net = EditNet(domain)
    else:
        assert False, f'bad pred mode: {args.pred_mode}'

    if model_path is None:
        utils.log_print("Warning, returning unititialized edit net", args)
        edit_net.to(domain.device)
        return edit_net
        
    utils.log_print(f"Loading edit net from {model_path}", args)
    edit_net.load_state_dict(
        torch.load(model_path)
    )

    edit_net.to(domain.device)
    return edit_net

def eval(domain):

    args = domain.get_ft_args()
    os_inf_net = domain.get_oneshot_net()                

    domain.load_prog_diff()
    if args.load_model_path is None:
        edit_net = None
    else:
        edit_net = get_edit_net(domain, args.load_model_path)
        edit_net.prog_diff = domain.prog_diff

    target_data = domain.load_target_dataset()
    target_data.mode = 'finetune'
        
    with torch.no_grad():

        os_inf_net.iter_num = 0
        
        iter_res = infer_progs.infer_for_eval(
            domain,
            os_inf_net,
            edit_net,
            target_data,
        )                    
        
# Fine-tune a recognition network towards a domain of interest
def fine_tune(domain):

    # Load args, rec net, target distribution of real_data
    args = domain.get_ft_args()
    
    assert domain.args.batch_size is None
    
    domain.args.batch_size = domain.args.os_batch_size

    os_inf_net = domain.get_oneshot_net()    
    os_gen_net = domain.get_oneshot_net(is_gen_model=True)    
        
    domain.args.batch_size = domain.args.edit_batch_size

    domain.load_prog_diff()
    if args.load_model_path is None:
        edit_net = None
    else:
        edit_net = get_edit_net(domain, args.load_model_path)
        edit_net.prog_diff = domain.prog_diff
        
    domain.args.batch_size = None
    
    target_data = domain.load_target_dataset()
    target_data.mode = 'finetune'
    
    assert 'WS' in args.ft_mode
    assert args.ws_train_size is not None
    
    train_pbest = BestPrograms(domain, 'train')
    val_pbest = BestPrograms(domain, 'val')

    logger = Logger(domain)

    TT = time.time()
    
    while True:
        if logger.check_early_stop(): break
                
        os_inf_net.iter_num = logger.inf_epochs[-1]
        # Run Inf Net over real_data to update best_prog data structure
        with torch.no_grad():

            iter_res = infer_progs.infer_programs(
                domain,
                os_inf_net,
                edit_net,
                target_data,
                train_pbest,
                val_pbest
            )
            
        logger.log(iter_res, os_inf_net, edit_net)
        
        # Stop early based on val metric
        if logger.inf_epochs[-1] - logger.best_epoch > args.iter_patience:
            utils.log_print("Stopping early", args)
            break                    
            
        utils.log_print("Training gen model", args)
        # next gen model, training data from gen, number of gen epochs

        os_gen_net, gen_data, ge = wake_sleep.make_ws_gens(
            domain, os_gen_net, train_pbest, val_pbest
        )
        utils.save_model(
            os_gen_net.state_dict(),
            f'model_output/{domain.args.exp_name}/os_gen_net.pt'
        )

        if edit_net is not None:            
            edit_ft_data = gen_data
            ed_ie = joint_plad.train_edit_plad(
                domain,
                edit_net,
                edit_ft_data,
                os_inf_net,                        
            )

        else:
            ed_ie = 0

        os_ie = joint_plad.train_os_plad(
            domain,
            os_inf_net,
            gen_data,            
            target_data,
            train_pbest,
        )

        logger.add_epochs(ed_ie, os_ie, ge, time.time() - TT)
        


            
