import math
import time
from tqdm import tqdm
import sys
sys.path.append('executors')
sys.path.append('executors/layout')
sys.path.append('executors/csg2d')
sys.path.append('executors/csg3d')

import ex_layout
import ex_csg2d
import ex_csg3d
import lay_prog_diff
import csg_prog_diff

def exec_test(ex, prog):
    print(prog)
    ex.execute(prog, vis=True)

def main():
    nm = sys.argv[1] 
    
    config = {'USE_END_TOKEN': True, 'EDIT_OP_TOKENS': None}

    config['EDIT_LOC_TOKENS'] = None
        
    if nm == 'lay':        
        ex= ex_layout.LayExecutor(config)

    elif nm == 'csg2d':        
        ex= ex_csg2d.CSGExecutor(config)

    elif 'csg3d' in nm:
        ex = ex_csg3d.CSGExecutor(config)
        
    else:
        assert False

    mode = sys.argv[2]

    if mode == 'exec':
        return exec_test(ex, ' '.join(sys.argv[3:]))
    else:
        return sample_test(ex, mode)

def sample_test_os(ex):
    efn = ex.det_prog_random_sample
    num = int(sys.argv[3])
    md = sys.argv[4]        
    efn(num, vis_progs=('vis' in md), print_stats =('stats' in md), use_pbar=True)

def sample_test_edit(ex):
    nm = sys.argv[1]

    if nm == 'lay':
        PE = lay_prog_diff.LayProgDiff(ex)
    elif nm == 'csg2d':
        PE = csg_prog_diff.CSGProgDiff(ex)
    elif nm == 'csg3d':
        PE = csg_prog_diff.CSGProgDiff(ex)
    else:
        assert False

    num = int(sys.argv[3])
    md = sys.argv[4]

    if md == 'vis':

        count = 0
        while count < num:            
            start_prog, end_prog = ex.det_prog_random_sample(2, use_pbar=False)
            try:
                edit_info = PE.get_edit_info(start_prog, end_prog)
            except Exception as e:
                continue
            
            if edit_info is None:
                continue
            
            PNF, EI, edit_ops = edit_info
            edit_seq = PE.verify(start_prog, end_prog, PNF, EI, edit_ops)

            ex.render_group(edit_seq, name=None, rows=int(math.sqrt(len(edit_seq))))
            
            count += 1
            
    elif md == 'stats':
        start_progs = ex.det_prog_random_sample(num, use_pbar=True)
        end_progs = ex.det_prog_random_sample(num, use_pbar=True)

        t = time.time()

        num = 0
        cost = 0
        count = 0

        errors = 0
        
        for sp, ep in tqdm(list(zip(start_progs, end_progs))):            
            try:            
                edit_info = PE.get_edit_info(sp, ep)
                PNF, EI, edit_ops = edit_info
            except Exception as e:
                errors += 1.
                continue

            count += 1.
            num += len(edit_ops)
            cost += PE.calc_true_edit_cost(edit_ops)
            
        t = round(time.time() -t, 2)
        print(f"Sampled {count} in {t} with {errors} errors")
            
        print(f"Avg Num Edits : {round(num/count,2)}")
        print(f"Avg Cost Edits : {round(cost/count,2)}")
                
        
def sample_test(ex, mode):
    if 'os' == mode:
        sample_test_os(ex)
    elif 'edit' == mode:
        sample_test_edit(ex)
    else:
        assert False
        
    
 
if __name__ == '__main__':
    main()
