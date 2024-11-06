from tqdm import tqdm
import time
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model_utils as mu
import utils

def joint_infer(edit_net, os_inf_net, info, ret_extra_info=False):
    try:
        return _joint_infer(edit_net, os_inf_net, info, ret_extra_info)
    except Exception as e:
        print(f'Failed joint infer with {e}')
        return None, None    
        
def make_oneshot_population(os_inf_net, domain, pop_size, target):

    pop = []
    ex = domain.executor
    while len(pop) < pop_size:

        os_pop = os_inf_net.eval_sample_progs(
            target,
            10
        )    

        for a,b,iprog,d in os_pop:

            try:
                assert iprog.count('START') <= 1
                assert 'START' in iprog[0]
            except:
                continue
        
            assert domain.args.pred_mode == 'edit'

            if 'END' not in iprog[-1] and domain.name == 'layout':
                iprog.append('END')

            eprog = ex.TLang.tokens_to_tensor(iprog)
                                
            pop.append((a, b, eprog, d))
        
    return pop[:pop_size]

def get_init_population(os_inf_net, domain, pop_size,  target):
    IP = make_oneshot_population(os_inf_net, domain, pop_size, target)
    return IP

def make_edits(net, population, target, args):
    
    inner_search_fn = mu.inner_search_beam_logic
    beams = args.infer_is_beam
    assert beams is not None, 'set beams'
    inner_search_fn_args = {
        'beams': beams
    }
        
    edits, error_perc = net.make_edits(
        population,
        target,
        inner_search_fn,
        inner_search_fn_args
    )
    
    return edits, error_perc
            

class OuterSearch:
    def __init__(self, domain, population, rei):
        self.domain = domain
        self.pop_size = len(population)

        self.uniq_pop = True
        
        self.res = {
            'rounds': [],
            'round_best_mval': [],
            'round_best_exec': [],
            'round_best_prog': [],
            'err_perc': []
        }

        args = domain.args
        
        self.record_res(-1, population)

        start_mval = torch.tensor([p[1] for p in population]).mean().item()

        self.rei = rei
        
    def make_population(self, prev_pop, edits):
        assert len(prev_pop) == len(edits)

        next_pop = []            
        for _pp,_edits in zip(prev_pop, edits):
            assert len(_pp) == 4
            next_pop.append(_edits + [
                (_pp[0] - 1000., _pp[1], _pp[2], _pp[3])
            ])
        return next_pop
        
    def make_dist(self, M):

        T = torch.tensor(M).float()
        T = (T - T.min()) + 1e-8    
        T /= T.sum()
        return T.numpy()
            
    def score_population(self, population):

        metric_vals = [mval for _, mval,_, _ in population]
        
        seen = set()
        clean_metric_vals = []

        for _,mval,tprog,_ in population:
            sig = tuple(tprog.tolist())
            if sig in seen:
                clean_metric_vals.append(0.01 * mval)
            else:
                clean_metric_vals.append(mval)
                seen.add(sig)                
                
        metric_dist = self.make_dist(clean_metric_vals)

        return metric_dist, metric_vals

    def test_time_record(self, TE, RE):
        ex = self.domain.executor
        self.test_time_info.append((
            TE,
            RE,
            self.res['round_best_mval'][-1],
            ex.TLang.tensor_to_tokens(self.res['round_best_prog'][-1])
        ))
        
    def record_res(self, ir, pop):
        # Record results
        metric_vals = [mval for _, mval,_, _ in pop]

        if len(metric_vals) == 0:
            print(f"Something has gone wrong on round {ir}")
            return pop
        
        best_ind = torch.tensor(metric_vals).argmax().item()
        best_mval = metric_vals[best_ind]

        if len(self.res['rounds']) == 0 or \
           self.domain.comp_metric(best_mval, self.res['round_best_mval'][-1]):
            rb_mval =  pop[best_ind][1]
            rb_prog =  pop[best_ind][2]
            rb_exec =  pop[best_ind][3]
        else:
            rb_mval = self.res['round_best_mval'][-1]
            rb_prog = self.res['round_best_prog'][-1]
            rb_exec = self.res['round_best_exec'][-1]
            
        self.res['rounds'].append(ir)
        self.res['round_best_mval'].append(rb_mval)
        self.res['round_best_prog'].append(rb_prog.cpu())

        self.res['round_best_exec'].append(rb_exec.cpu())

    def get_top_opts(self, P, S, num):
        assert num > 0
        
        L = [(s,i) for i,s in enumerate(S)]
            
        L.sort(reverse=True)

        NP = []

        for _,i in L[:num]:
            NP.append(P[i])

        return NP
    
    def choose_from_top(self, population, mvals):
        np = self.get_top_opts(population, mvals, self.pop_size)
        return np
        
    def select_next_pop(self, ir, prev_pop, edits):
        
        # Create next population
        nested_population = self.make_population(prev_pop, edits)
        
        population = []
        for np in nested_population:
            population += np
                                            
        pop_dist, metric_vals = self.score_population(population)

        next_pop = self.choose_from_top(population, metric_vals)
                                
        self.record_res(ir, next_pop)
                                
        return next_pop

    def get_results(self, target):

        res = {
            'mval_best': self.res['round_best_mval'][-1],
            'mval_start': self.res['round_best_mval'][0],
            'count': 1,
        }
        
        res['mval_imp'] = res['mval_best'] - res['mval_start']
        
        tokens = self.res['round_best_prog'][-1]
        
        program = self.domain.executor.TLang.tensor_to_tokens(tokens)

        if self.domain.name != 'layout':        
            program.append(self.domain.executor.END_TOKEN)        
                
        if isinstance(target, dict):
            res_tar = {k:v[0].cpu() for k,v in target.items()}
        else:
            res_tar = target[0]
            
        if self.rei:            
            self.res['target'] = res_tar
            return res, self.res
            
        info = {
            'mval': res['mval_best'],
            'program': program,
            'tar_exec': res_tar,
            'start_exec': self.res['round_best_exec'][0],
            'best_exec': self.res['round_best_exec'][-1],
        }
                    
        return res, info
    
def _joint_infer(edit_net, os_inf_net, info, ret_extra_info):

    domain = edit_net.domain
    args = edit_net.domain.args
    
    pop_size = args.infer_pop_size
    infer_rounds = args.infer_rounds
    
    if 'vdata' in info:
        target = info['vdata']
    else:
        target = {k:v for k,v in info.items() if 'vdata' in k}
        
    population = get_init_population(
        os_inf_net, domain, pop_size, target
    )
        
    OS = OuterSearch(domain, population, ret_extra_info)
    
    for ir in range(infer_rounds):

        if OS.domain.is_perfect_recon(OS.res['round_best_mval'][-1]):
            # Dummy edits if we have perfectly reconstructed target
            edits = [[] for _ in population]
        else:        
            edits,err_perc = make_edits(edit_net, population, target, args)
            OS.res['err_perc'].append(err_perc)
            
        population = OS.select_next_pop(
            ir,
            population,
            edits,            
        )

        if len(population) == 0:
            # Something went wrong
            print(f"Saw zero population at round {ir}")
            break
        
    return OS.get_results(target)

def test_time_infer(edit_net, os_inf_net, args, info, key):
    try:
        return _test_time_infer(edit_net, os_inf_net, args, info, key)
    except Exception as e:
        utils.log_print(f'Failed test time infer for {key} with {e}', args)
        return None

def _test_time_infer(edit_net, os_inf_net, args, info, key):

    T = time.time()

    if edit_net is None:
        domain = os_inf_net.domain
    else:
        domain = edit_net.domain
    
    pop_size = args.infer_pop_size
    infer_rounds = args.infer_rounds
    
    if 'vdata' in info:
        target = info['vdata']
    else:
        target = {k:v for k,v in info.items() if 'vdata' in k}
                
    rounds_left = infer_rounds + 1
    
    population = []

    while len(population) == 0:        
        if rounds_left == 0:
            break
    
        population = get_init_population(
            os_inf_net, domain, pop_size, target
        )

        rounds_left -= 1
        
    if len(population) == 0:
        assert False, f'no valid oneshot samples for {key}'
        
    OS = OuterSearch(domain, population, True)
    OS.test_time_info = []
    OS.test_time_record(T - time.time(), infer_rounds - rounds_left)        
    ir = 0
                
    while rounds_left > 0:
        rounds_left -= 1
        
        if OS.domain.is_perfect_recon(OS.res['round_best_mval'][-1]):
            edits = [[] for _ in population]
        else:
            if edit_net is not None:
                edits,_ = make_edits(edit_net, population, target, args)
            else:
                flat_edits = get_init_population(
                    os_inf_net, domain, pop_size, target
                )
                edits = []
                for _ in range(len(population)):
                    if len(flat_edits) > 0:
                        fe = [flat_edits.pop(0)]
                    else:
                        fe = []
                        
                    edits.append(fe)
                
        population = OS.select_next_pop(
            ir,
            population,
            edits,            
        )
        ir += 1
        OS.test_time_record(time.time() - T, infer_rounds - rounds_left)
        
    return OS.get_results(target)[1]

