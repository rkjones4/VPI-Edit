import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model_utils as mu
from tqdm import tqdm
import random
from utils import device
import time
import matplotlib.pyplot as plt
import utils

VERBOSE = False

class OneShotNet(nn.Module):
    def __init__(
        self,
        domain,
        is_gen_model = False    
    ):

        super(OneShotNet, self).__init__()
            
        self.vis_mode = None
        self.domain = domain
        args = self.domain.args
        
        self.mode = 'recon'
                
        self.device = domain.device
        self.mp = args.max_prim_enc
        self.hd = args.hidden_dim
        self.ex = domain.executor

        self.is_gen_model = is_gen_model

        if self.is_gen_model:
            self.gen_epoch = 0
            self.encoder = nn.Embedding(self.mp, self.hd)
            self.enc_dummy_arange = torch.arange(self.mp,device=self.device).unsqueeze(0)
            self.encode = self.gen_encode
        else:
            self.encoder = mu.load_vis_encoder(domain)
            self.encode = self.inf_encode
            
        self.eval_info = None
        self.eval_count = {}
        self.eval_res = {}
        
        self.num_write = args.num_write
        
        self.seq_net = mu.TDecNet(
            domain,
            self.mp,
            args.max_seq_len
        )
        
        self.TTNN = torch.zeros(
            self.ex.TLang.nt,
            device=self.device
        ).long()
        
        for t in self.ex.TLang.tokens.keys():
            tind = self.ex.TLang.T2I[t]
            self.TTNN[tind] = self.ex.TLang.get_num_inp(t)            

    ###################
    ### TRAIN LOGIC ###
    ###################

    def gen_encode(self, pixels):
        assert isinstance(pixels, int)
        return self.encoder(
            self.enc_dummy_arange.repeat(pixels,1)
        )
    
    def inf_encode(self, pixels):
        return self.encoder(pixels)

    def get_codes(self, batch):
        if 'vdata' in batch:
            vdata = batch['vdata']        
            codes = self.encode(vdata)
        else:
            codes = self.encode({
                k:v for k,v in batch.items() if 'vdata' in k
            })
        return codes

    def make_prog_preds(self, batch, codes, res):
        seq = batch['seq']
        seq_weight = batch['seq_weight']
        
        seq_preds = self.seq_net.infer_prog(
            codes,
            seq
        )
        
        flat_seq_preds = seq_preds[:,:-1,:].reshape(-1,seq_preds.shape[2])
        flat_seq_targets = seq[:,1:].flatten()    
        flat_seq_weights = seq_weight[:,1:].flatten()
        
        loss, corr, total = mu.calc_token_loss(
            flat_seq_preds,
            flat_seq_targets,
            flat_seq_weights
        )

        res['loss'] = loss
        res['corr'] = corr
        res['total'] = total

        
    def model_train_batch(self, batch):

        res = {}
        
        codes = self.get_codes(batch)

        self.make_prog_preds(batch, codes, res)
            
        loss = res['loss']        

        res = {k:v.item() for k,v in res.items()}

        return loss, res
            

    
    ###################
    ### EVAL LOGIC ###
    ###################
        
    def eval_infer_progs(
        self,
        pixels,
        beams = None,
    ):
        if beams is None:
            beams = self.domain.args.beams
        
        vdata = pixels
        codes = self.encode(pixels)        
        
        final_preds = self.infer_prog_eval(codes, beams)
        
        ret_info = self.make_batch_ret_info(
            final_preds,
            pixels,
            beams
        )

        return ret_info

    def make_inst_ret_info(
        self, batch_ind, final_preds, pixels, beams
    ):

        best_mval = self.domain.init_metric_val()
        best_exec = None
        best_info = None

        for _,pred in final_preds[batch_ind]:
        
            tokens = self.ex.TLang.tensor_to_tokens(pred)
            expr = ' '.join(tokens)

            exc, mval, info = self.domain.get_vis_metric(
                expr,
                pixels[batch_ind],
            )
            if exc is None or mval is None:
                continue

            if self.domain.comp_metric(mval, best_mval):
                info['expr'] = expr
                best_exec = exc
                best_info = info
                best_mval = mval

        return {
            'exec': best_exec,
            'mval': best_mval,
            'info': best_info
        }

    
    def make_batch_ret_info(
        self, final_preds, pixels,  beams
    ):
        R = {}

        batch_num = pixels.shape[0]        

        for batch_ind in range(batch_num):
            r = self.make_inst_ret_info(
                batch_ind, final_preds, pixels, beams
            )
            for k,v in r.items():
                if k not in R:
                    R[k] = []
                R[k].append(v)

        return R


    def eval_infer_progs_sample(
        self,
        pixels,
        pop_size,
        rounds,
        rei=False    
    ):

        codes = self.encode(pixels)
        
        batch_codes = codes.repeat(pop_size,1,1)

        best_mval = self.domain.init_metric_val()
        best_exec = None
        best_expr = None

        if rei:
            res = {
                'rounds': [],
                'round_best_exec': [],
                'round_best_prog': [],
                'target': pixels[0]
            }
        
        for ri in range(rounds):

            if rei:
                res['rounds'].append(ri)
            
            info = {
                'batch': pop_size,
                'bprefix': batch_codes
            }

            samples = self.ar_sample_logic(
                self.seq_net,
                info,
                1,
                self.seq_net.ms
            )        

            for pred in samples.values():
            
                try:

                    tokens = self.ex.TLang.tensor_to_tokens(pred[0])
                    expr = ' '.join(tokens)                
                    exc, mval, _ = self.domain.get_vis_metric(
                        expr,
                        pixels[0],
                    )
                    
                    if exc is None or mval is None:
                        continue

                    if self.domain.comp_metric(mval, best_mval):
                        best_mval = mval
                        best_exec = exc.cpu()
                        best_expr = expr

                except Exception as e:
                    pass

            
            if rei:
                res['round_best_exec'].append(best_exec)                    
                res['round_best_prog'].append(
                    self.ex.TLang.tokens_to_tensor(best_expr.split())
                )
                    
        return {
            'mval': [best_mval],
            'info': [{'expr': best_expr}],
            'exec': [best_exec],
            'extra': res if rei else None
        }
           
    def eval_sample_progs(
        self,
        pixels,
        num
    ):
        codes = self.encode(pixels)
        batch_codes = codes.repeat(num,1,1)
        
        info = {
            'batch': num,
            'bprefix': batch_codes
        }
        
        samples = self.ar_sample_logic(
            self.seq_net,
            info,
            1,
            self.seq_net.ms
        )        
        
        gens = []
        
        for pred in samples.values():
            
            try:                

                tokens = self.ex.TLang.tensor_to_tokens(pred[0])
                expr = ' '.join(tokens)                
                exc, mval, _ = self.domain.get_vis_metric(
                    expr,
                    pixels[0],
                )
                    
                if exc is None or mval is None:
                    continue

                gens.append((
                    0.,
                    mval,
                    tokens,
                    exc
                ))
                
            except Exception as e:
                if VERBOSE:
                    print(f'Failed sample: {e}')
                pass

        return gens

    def gen_sample_progs(
        self,
        num
    ):
        batch_codes = self.encode(num)        
        info = {
            'batch': num,
            'bprefix': batch_codes
        }
        
        samples = self.ar_sample_logic(
            self.seq_net,
            info,
            1,
            self.seq_net.ms
        )        
                
        gens = []
        
        for pred in samples.values():
            
            try:
                tokens = self.ex.TLang.tensor_to_tokens(pred[0])                                
                expr = ' '.join(tokens)
                self.ex.execute(expr)
                gens.append(tokens)
                
            except Exception as e:
                if VERBOSE:
                    print(f'failed v3 : {e}')


        return gens


    def eval_batch_sample_prog(
        self,
        pixels,
    ):

        codes = self.encode(pixels)
        
        info = {
            'batch': codes.shape[0],
            'bprefix': codes
        }
        
        samples = self.ar_sample_logic(
            self.seq_net,
            info,
            1,
            self.seq_net.ms
        )        
        
        ret = {}

        miss = 0
        total = 1
        mvals = []
        
        for i, pred in samples.items():
            total += 1

            try:
                tokens = self.ex.TLang.tensor_to_tokens(pred[0])
                expr = ' '.join(tokens)
                exc, mval, info = self.domain.get_vis_metric(
                    expr,
                    pixels[i],
                )
                if exc is None or mval is None:
                    continue
                
                ret[i] = tokens
                mvals.append(mval)
            except Exception as e:
                ret[i] = None
                miss += 1
        
        return ret
    
    def infer_prog_eval(self, prefix, beams):
        batch = prefix.shape[0]
        
        bprefix = prefix

        net = self.seq_net
        net.is_eval_fn = net.infer_prog
        
        bseqs = torch.zeros(1, net.ms, device=self.device).long()
        bseqs[:,0] = self.ex.TLang.T2I[self.ex.START_TOKEN]
        
        info = {
            'batch': batch,
            'bprefix': bprefix,
            'bseqs': bseqs, 
        }

        if self.TTNN is not None:
            bqc = torch.ones(
                batch,
                device=self.device
            ).long()
            info['bqc'] = bqc
            info['ttnn'] = self.TTNN
            info['_extra'] = 2
        else:
            info['END_TOKEN_IND'] = self.ex.TLang.T2I[self.ex.END_TOKEN]
        
        struct_preds = self.ar_eval_logic(
            net,
            info,
            beams,
            self.seq_net.ms
        )        
        
        for i, fp in struct_preds.items():            
            fp.sort(reverse=True, key=lambda a: a[0])
            struct_preds[i] = fp[:beams]

        return struct_preds

    def ar_eval_logic(self, net, info, beams, max_len):
        return mu.inner_search_beam_logic(
            net,
            info,
            max_len,
            {'beams': beams}
        )
        
    def ar_sample_logic(
        self,
        net,
        info,
        beams,
        max_len    
    ):

        assert beams == 1
        
        batch = info['batch']

        bprefix = info['bprefix']

        if 'bseqs' not in info:                                
            bseqs = torch.zeros(batch * beams, net.ms, device=self.device).long()
            bseqs[:,0] = self.ex.TLang.T2I[self.ex.START_TOKEN]
        else:
            bseqs = info['bseqs']

        if 'bpp_left' not in info:        
            bpp_left = torch.ones(batch * beams, device=self.device).float()
        else:
            bpp_left = info['bpp_left']

        if 'bpp_nloc' not in info:
            bpp_nloc = torch.zeros(batch * beams, device=self.device).long()
        else:
            bpp_nloc = info['bpp_nloc']

        if 'bsinds' not in info:
            bsinds = torch.zeros(batch * beams, device=self.device).long()            
        else:
            bsinds = info['bsinds']

        TTNN = self.TTNN
        assert TTNN is not None
        
        if 'blls' not in info:
            # batch log liks
            blls = torch.zeros(batch, beams, device=self.device)        
            blls = blls.flatten()
            
        else:
            blls = info['blls']

        if 'bqc' not in info:
            bqc = torch.ones(
                batch * beams,
                device=self.device
            ).long()
            _extra = 1
            
        else:
            bqc = info['bqc']
            _extra = info['bqc_extra']
            
        # [batch, beam, O]

        max_token_num = self.seq_net.nt
        
        fin_progs = {i:[] for i in range(batch)}

        fin_count = torch.zeros(
            batch,
            device=self.device
        )
        
        break_cond = torch.zeros(batch, beams, device=self.device).bool()
        
        fin_lls = [[] for _ in range(batch)]
        
        dummy_arange = torch.arange(beams * batch, device=bprefix.device)        

        max_ind = bseqs.shape[1] - 1 
        
        for PL in range(max_len-1):            

            break_cond = break_cond | (bpp_left.view(batch, beams) <= 0)
            
            E_blls = blls.view(batch, beams)
            
            for i in (fin_count >= beams).nonzero().flatten():
                fin_nll = -1 * torch.tensor([
                    np.partition(fin_lls[i], beams-1)[beams-1]
                ], device=self.device)

                if (E_blls[i] < fin_nll).all():
                    break_cond[i] = True
            
            if break_cond.all():                  
                break
                        
            exp_bpreds = net.infer_prog(bprefix, bseqs)
            
            bpreds = exp_bpreds[dummy_arange, bsinds]
            
            bdist = torch.softmax(bpreds, dim = 1)            
            
            nt = torch.distributions.categorical.Categorical(bdist).sample()
            bsinds += 1
            
            bsinds = torch.clamp(bsinds, 0, max_ind)
            bseqs[dummy_arange, bsinds] = nt

            bqc -= 1            
            bqc += TTNN[nt]
            
            p_fin_inds = (bqc == 0.).nonzero().flatten()
            
            if p_fin_inds.shape[0] > 0:

                bpp_left[p_fin_inds] -= 1
                bqc[p_fin_inds] += 1                
                bsinds[p_fin_inds] += 1                

                bsinds = torch.clamp(bsinds, 0, max_ind)                
                
                bseqs[
                    dummy_arange[p_fin_inds],
                    bsinds[p_fin_inds]
                ] = bpp_nloc[p_fin_inds]
                
                bpp_nloc[p_fin_inds] += 1
                bpp_nloc = torch.clamp(bpp_nloc, 0, max_token_num-1)
                
            fin_inds = (bpp_left == 0.).nonzero().flatten().tolist()

            for i in fin_inds:
                                                                                
                if blls[i] > mu.MIN_LL_THRESH:
                    
                    beam_ind = i

                    _ll = 0.0
                    fin_progs[beam_ind].append(
                        bseqs[i,:bsinds[i]+_extra]
                    )
                    fin_count[beam_ind] += 1
                    fin_lls[beam_ind].append(-1 * _ll)                        
                        
                blls[i] += mu.MIN_LL_PEN
                bqc[i] += 1

        return fin_progs

    def flush_detp_images(
        self, images, name, itn, group_size = 5
    ):
        i = 0
        while len(images) > 0:
            bimages = images[:group_size * 2]
            images = images[group_size * 2:]
            self._flush_detp_images(bimages,name,i,itn)
            i += 1

    def _flush_detp_images(self, images,name,ind,itn):

        fig, axes = self.ex.make_plot_render(
            2, len(images)//2, (16,6)
        )
                
        for I in range(len(images)):
            i = I % 2
            j = I // 2
            
            if images[I] is None:
                continue

            self.ex.vis_on_axes(axes[i,j], images[I])        

        plt.savefig(f'{name}_b_{ind}_itn_{itn}')
        plt.close('all')
        plt.clf()
                                    
    def model_eval_fn(
        self,
        vdata,
        beams=None,
        ret_info = False
    ):
                
        eval_info = self.eval_infer_progs(
            vdata['vdata'],
            beams,
        )
        eval_res = self.make_result(vdata['vdata'], eval_info)        
        
        if ret_info:
            return eval_info, eval_res
        else:
            return eval_res
        
    def save_vis_logic(self):
        
        args = self.domain.args
        try:
            set_name, itn = self.vis_mode
            name = f'{args.outpath}/{args.exp_name}/vis/{set_name}'

            if len(self.eval_res[self.vis_mode]) == 0:
                return
        
            self.flush_detp_images(self.eval_res[self.vis_mode], name, itn)
            self.eval_res.pop(self.vis_mode)

        except Exception as e:
            utils.log_print(f"Failed to save vis for {set_name} ({itn}) with: {e}", args)

    def record_vis_logic(self, vdata, eval_info):

        for i in range(vdata.shape[0]):
            if self.eval_count[self.vis_mode] >= self.num_write:
                return
                                                                                    
            pred = eval_info['exec'][i]
            target = vdata[i]                
            self.eval_count[self.vis_mode] += 1
            self.eval_res[self.vis_mode] += [
                target.cpu().numpy() if target is not None else None,
                pred.cpu().numpy() if pred is not None else None
            ]                
        
    def init_vis_logic(self):
        set_name, itn = self.vis_mode        
        
        if self.vis_mode not in self.eval_count:
            self.eval_count[self.vis_mode] = 0
            self.eval_res[self.vis_mode] = []
            
        
    def make_result(self, vdata, eval_info):

        if self.vis_mode is not None:
            self.record_vis_logic(vdata, eval_info)
        
        res = {
            'mval': sum(eval_info['mval']) * 1. / len(eval_info['mval']),
            'count': 1.
        }

        for info in eval_info['info']:
            if info is None:
                continue
            for k,v in info.items():                
                if isinstance(v, float):
                    if k not in res:
                        res[k] = 0.
                        
                    res[k] += v
        
        return res

        

        
