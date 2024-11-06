import matplotlib.pyplot as plt
import json
import torch
import time
import math
from tqdm import tqdm
import utils
import joint_infer as ji

VERBOSE = False

FT_EVAL_LOG_INFO = [
    ('Obj', 'mval_best', 'count'),
    ('Start Obj', 'mval_start', 'count'),
    ('Error Rate', 'errors', 'count')
]

def os_inference(os_inf_net, args, vinput, ret_extra_info=False):
    if args.infer_is_mode == 'beam':
        try:
            eval_info = os_inf_net.eval_infer_progs(
                vinput,
                beams = os_inf_net.beams,
            )
            assert len(eval_info['mval']) == 1, 'more than one'
        except Exception as e:
            if VERBOSE:
                print(f"Failed os infer with {e}")
            return None, None
    elif args.infer_is_mode == 'sample':
        try:
            eval_info = os_inf_net.eval_infer_progs_sample(
                vinput,
                pop_size = args.infer_pop_size ,
                rounds = args.infer_rounds,
                rei=ret_extra_info
            )            
            assert len(eval_info['mval']) == 1, 'more than one'            
            assert eval_info['exec'][0] is not None, 'none inference'
        except Exception as e:
            if VERBOSE:
                print(f"Failed os infer with {e}")
            return None, None
    
    else:
        assert False, f'bad os im {args.infer_is_mode}'
    
    try:
        res = {
            'mval_best': eval_info['mval'][0],
            'count': 1,
        }
        
        if ret_extra_info:
            info = eval_info['extra']
        else:
            info = {
                'mval': res['mval_best'],
                'program': eval_info['info'][0]['expr'].split(),
                'tar_exec': vinput[0],
                'best_exec': eval_info['exec'][0],
                'start_exec': eval_info['exec'][0]
            }

    except Exception as e:
        if VERBOSE:
            print(f"Failed os infer with {e}")
        return None, None
    
    return res, info
    
def run_inference(os_inf_net, edit_net, args, vinput, ret_extra_info=False):
    if edit_net is None:
        return os_inference(os_inf_net, args, vinput, ret_extra_info)
    else:
        return ji.joint_infer(edit_net, os_inf_net, {'vdata': vinput}, ret_extra_info)

class VisStruct:
    def __init__(self, name, itn, args):
        self.save_path_base = f'{args.outpath}/{args.exp_name}/vis/{name}'
        self.itn = itn
        self.num_to_save = args.num_write
        self.num_per_render = 5

        self.data = []
        
    def add_vis_ex(self, tar, start, best):

        if len(self.data) >= self.num_to_save:
            return

        self.data.append((tar, start, best))
                    
    def save_res(self, domain):
        
        i = -1        
        
        while len(self.data) > 0:
            i += 1
            r1 = []
            r2 = []
            r3 = []

            count = 0
            while len(self.data) > 0 and count < self.num_per_render:
                count += 1

                tar, start, best = self.data.pop(0)
                r1 += [tar]
                r2 += [start]
                r3 += [best]

            domain.executor.render_group(
                r1 + r2 + r3,
                f'{self.save_path_base}_grp_{i}_itn_{self.itn}',
                rows = 3
            )

class EvalVisStruct:
    def __init__(self):
        self.data = []
        self.error_count = 0
        
    def add_res(self, eres, time):

        if eres is None:
            self.error_count += 1
        else:
            self.data.append((eres, time))

    def get_and_save_info(self, domain):

        args = domain.args
        ex = domain.executor

        R = {}
        AT = []
        
        for eres, time in self.data:
            AT.append(time)

            if 'rounds' not in R:
                R['rounds'] = eres['rounds']

            assert len(R['rounds']) == len(eres['rounds'])

            for i, exc, prg in zip(eres['rounds'],eres['round_best_exec'], eres['round_best_prog']):
                
                rec_mets = domain.pixel_recon_metrics(
                    exc,
                    eres['target']
                )

                rec_mets['prog_len'] = prg.shape[0]
                
                for k,v in rec_mets.items():
                    if k not in R:
                        R[k] = {}
                    if i not in R[k]:
                        R[k][i] = []

                    R[k][i].append(v)

        def avg(L):
            return torch.tensor(L).float().mean().item()

        srounds = R.pop('rounds')
        srounds.sort()
        
        AR = {'rounds': srounds}
        for mn, V in R.items():
            AR[mn] = []
            for i in AR['rounds']:
                ml = V[i]
                AR[mn].append(avg(ml))
                
        utils.log_print(
            f"Inference time {round(avg(AT), 3)} | Errors : {self.error_count}", args
        )

        json.dump(AR, open(f'model_output/{args.exp_name}/eval_inf_res.json', 'w'))

        rounds = AR.pop('rounds')
        
        for mn, V in AR.items():
            best_val = round(max(V), 3)
            start_val = round(V[0], 3)
            imp_amt = round(best_val - start_val, 3)

            utils.log_print(
                f"  {mn} : {best_val} | Start : {start_val} | Imp {imp_amt}", args
            )

            out_name = f'model_output/{args.exp_name}/plots/{mn}_over_rounds.png'
            
            plt.clf()
            plt.plot(rounds, V)
            plt.grid()
            plt.savefig(out_name)
            plt.close('all')
            
    def save_results(self, domain):
        with torch.no_grad():
            self.get_and_save_info(domain)

        RNDS = [-1, 0, 1, 3, 7, 15, 31, 63]

        num_write = domain.args.num_write
        
        for c, (eres, _) in enumerate(self.data[:num_write]):

            row_info = {}

            for i, exc in zip(eres['rounds'],eres['round_best_exec']):
                if i in RNDS:
                    row_info[i] = exc

            if 'max' in RNDS:
                row_info['max'] = exc
                
            row_execs = [row_info[i] for i in RNDS if i in row_info] + [eres['target']]

            domain.executor.render_group(
                row_execs,
                f'model_output/{domain.args.exp_name}/vis/inf_eval_{c}',
                rows = 1
            )
                        
def infer_programs(domain, os_inf_net, edit_net, data, train_pbest, val_pbest):
    args = domain.args

    path = args.infer_path
    
    os_inf_net.eval()
    if edit_net is not None:
        edit_net.eval()
    else:
        os_inf_net.beams = args.os_inf_beams
        
    results = {}

    ITER_DATA = [
        (data.train_eval_iter, train_pbest, data.get_set_size('train'), 'train'),
        (data.val_eval_iter, val_pbest, data.get_set_size('val'), 'val'),
    ]
    
    for gen, record, num, name in ITER_DATA:

        VS = VisStruct(name, os_inf_net.iter_num, args)
        
        eval_res = {
            'errors': 0.,
            'count': 0.
        }
        
        utils.log_print(f"Inferring for {name}", args)

        assert args.eval_batch_size == 1
        
        for batch in \
            tqdm(gen(), total = math.ceil(num / args.eval_batch_size)):
            # Inference network runs beam search on each entry in vinput, and returns the beam with highest metric against the entry

                        
            key = batch['bkey']
            vinput = batch['vdata']

            eres, einfo = run_inference(
                os_inf_net,
                edit_net,
                args,
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
                
            if record is not None:                                                                
                record.update(
                    key,
                    einfo['program'],
                    einfo['mval']
                )

            VS.add_vis_ex(
                einfo['tar_exec'],
                einfo['start_exec'],
                einfo['best_exec'],
            )

        results[name] = utils.print_results(
            FT_EVAL_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )

        utils.log_print(f'Eval res {name}:', args)

        for k,v in results[name].items():
            rv = round(v,3)
            utils.log_print(f"    {k}: {rv}", args)
            
        VS.save_res(domain)
        
    return results



def infer_for_eval(domain, os_inf_net, edit_net, data):
    args = domain.args

    path = args.infer_path
    
    os_inf_net.eval()
    if edit_net is not None:
        edit_net.eval()
    else:
        os_inf_net.beams = args.os_inf_beams
        
    results = {}

    ITER_DATA = [
        (data.test_eval_iter, None, data.get_set_size('test'), 'test')
    ]

    count = -1
        
    for gen, record, num, name in ITER_DATA:        
        EVS = EvalVisStruct()
                
        assert args.eval_batch_size == 1
        
        for batch in \
            tqdm(gen(), total = math.ceil(num / args.eval_batch_size)):
            # Inference network runs beam search on each entry in vinput, and returns the beam with highest metric against the entry

            count += 1
                        
            key = batch['bkey']
            vinput = batch['vdata']
                            
            t = time.time()

            eres = ji.test_time_infer(
                edit_net,
                os_inf_net,                
                args,
                {'vdata': vinput},
                key
            )

            EVS.add_res(eres, time.time()-t)
            
        EVS.save_results(domain)
