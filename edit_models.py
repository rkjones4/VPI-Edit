from copy import deepcopy
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
import sys
import joint_infer as ji

VERBOSE = False

bceloss = torch.nn.BCELoss(reduction='none')

class EditNet(nn.Module):    
    def __init__(
        self,
        domain,
    ):
        super(EditNet, self).__init__()
        
        self.vis_mode = None
        self.domain = domain

        args = domain.args
        
        self.device = domain.device

        # how many tokens come from visual encoder
        self.mp = args.max_prim_enc
        self.inp_ms = args.max_seq_len
        self.edit_ms = args.max_edit_seq_len
        self.hd = args.hidden_dim
        self.ex = domain.executor

        self.token_net = EditTokenNet(
            domain,
            self.mp + self.mp + self.inp_ms,
            self.edit_ms
        )

        self.token_net.is_eval_fn = self.token_net.infer_eop_seq
                            
        self.encoder = mu.load_vis_encoder(domain)

        self.ET2MEL = domain.make_et_to_mel(self.edit_ms)
        
            
    def init_vis_logic(self):
        pass
        
    def encode_img(self, pixels):
        return self.encoder(pixels)
    
    def get_train_codes(self, batch):
        
        inp_img_code, tar_img_code = self.get_vis_codes(batch)

        ps_inp_seq = batch['ps_inp_seq']
        ps_inp_seq_weight = batch['ps_inp_seq_weight']

        ptl_inp_seq = batch['ptl_inp_seq']
        ptl_inp_seq_weight = batch['ptl_inp_seq_weight']

        ps_codes = self.get_inp_codes(
            ps_inp_seq, ps_inp_seq_weight,
            inp_img_code, tar_img_code
        )

        ptl_codes = self.get_inp_codes(
            ptl_inp_seq, ptl_inp_seq_weight,
            inp_img_code, tar_img_code
        )

        return ps_codes, ptl_codes
            
    def get_vis_codes(self, batch):
        
        inp_vdata = batch['inp_vdata']
        tar_vdata = batch['vdata']
        inp_img_code = self.encode_img(inp_vdata)
        tar_img_code = self.encode_img(tar_vdata)

        return inp_img_code, tar_img_code
        
    def get_inp_codes(self, inp_seq, inp_seq_weight, inp_img_code, tar_img_code):
                
        inp_seq_code = self.token_net.token_enc_net(inp_seq).view(
            -1, self.inp_ms, self.hd
        )

        mask = inp_seq_weight.unsqueeze(-1)
        
        inp_seq_code *= mask

        codes = torch.cat(
            (inp_img_code, tar_img_code, inp_seq_code), dim=1
        )
        
        return codes

    def sample_edit_ops(self, ptl_codes, plt_inp_seq_weight):
        
        raw_type_preds, raw_loc_preds = self.token_net.infer_eop_tl(ptl_codes)

        type_preds = torch.softmax(raw_type_preds,dim=-1)        

        type_samps = torch.distributions.Categorical(
            type_preds
        ).sample()
                        
        filt_loc_preds = raw_loc_preds[
            torch.arange(raw_loc_preds.shape[0], device=raw_loc_preds.device),
            :,
            type_samps
        ]

        filt_loc_dist = torch.softmax(filt_loc_preds,dim=-1)
        
        valid_loc_dist = filt_loc_dist * plt_inp_seq_weight

        valid_loc_dist /= valid_loc_dist.sum(dim=-1).unsqueeze(-1)

        loc_samps = torch.distributions.Categorical(
            valid_loc_dist
        ).sample()        
        
        edit_ops = []
        edit_lls = []
        
        for i in range(type_samps.shape[0]):
            eot = self.ex.EOI2T[type_samps[i].item()]
            eol = loc_samps[i].item()
            edit_ops.append((eot, eol))
                                                
            ell = 0.
            ell += torch.log(type_preds[i][type_samps[i]] + 1e-8)
            ell += torch.log(valid_loc_dist[i][eol] + 1e-8)
            edit_lls.append(ell.item())
        
        return edit_ops, edit_lls

        
    def make_eotl_preds(self, batch, ptl_codes, res):

        type_preds, loc_preds = self.token_net.infer_eop_tl(ptl_codes)

        eop_type = batch['eop_type_ind']

        type_loss, type_corr, type_total = mu.calc_token_loss(
            type_preds,
            eop_type,
            None
        )
        
        res['type_loss'] = type_loss
        res['type_corr'] = type_corr.item()
        res['type_total'] = type_total
        
        eop_loc = batch['eop_loc_ind']

        typed_loc_preds = loc_preds[torch.arange(eop_type.shape[0], device=loc_preds.device),:,eop_type]

        loc_loss, loc_corr, loc_total = mu.calc_token_loss(
            typed_loc_preds,
            eop_loc,
            None
        )            

        res['loc_loss'] = loc_loss
        res['loc_corr'] = loc_corr.item()
        res['loc_total'] = loc_total

        return res
                
    def make_eops_preds(self, batch, ps_codes, res):
        seq = batch['ps_tar_seq']
        seq_weight = batch['ps_tar_seq_weight']
        
        seq_preds = self.token_net.infer_eop_seq(
            ps_codes,
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

        res['seq_loss'] = loss
        res['seq_corr'] = corr.item()
        res['seq_total'] = total.item()
        
    def model_eval_fn(self, batch):
        
        jb = {'vdata': batch['vdata'].unsqueeze(0)}
            
        eres, einfo = ji.joint_infer(
            self,
            batch['os_net'],
            jb
        )
        
        eval_res = {'errs': 0.}
        
        if eres is None:
            assert einfo is None
            eval_res['errs'] += 1
            
        else:
            for k,v in eres.items():
                if k not in eval_res:
                    eval_res[k] = 0.
                eval_res[k] += v


        name = batch['name']
        
        if einfo is not None and name is not None:
            self.ex.render_group(                
                [
                    einfo['start_exec'],
                    einfo['best_exec'],
                    einfo['tar_exec']
                ],
                name=name,
                rows= 1
            )
                                
        return eval_res
                
    def model_train_batch(self, batch):
        res = {}

        ps_codes, ptl_codes = self.get_train_codes(batch)

        self.make_eotl_preds(batch, ptl_codes, res)
        
        self.make_eops_preds(batch, ps_codes, res)
                
        loss = res['type_loss'] + res['loc_loss'] + res['seq_loss']

        res['loss'] = loss.item()
        res['type_loss'] = res['type_loss'].item()
        res['loc_loss'] = res['loc_loss'].item()
        res['seq_loss'] = res['seq_loss'].item()            
       
        return loss, res
                
    def format_input_progs(self, population):
        inp_seq = torch.zeros(len(population), self.inp_ms).long().to(self.device)
        inp_seq_weight = torch.zeros(len(population), self.inp_ms).float().to(self.device)

        for i, (_,_,iprog,_) in enumerate(population):
            if iprog.shape[0] > self.inp_ms:
                if VERBOSE:
                    print("Saw input prog too big")
                continue
            inp_seq[i,:iprog.shape[0]] = iprog
            inp_seq_weight[:, :iprog.shape[0]] = 1.0
            
        return inp_seq, inp_seq_weight


    def make_edits(
        self, population, tar_pixels, is_fn, is_fn_args, ret_extra=False
    ):
        
        ptl_inp_seq, ptl_inp_seq_weight = self.format_input_progs(population)

        inp_vdata = torch.stack([ivdata for _,_,_,ivdata in population],dim=0)
        tar_vdata = tar_pixels.repeat(len(population),1,1,1)

        inp_img_code, tar_img_code = self.get_vis_codes({
            'inp_vdata': inp_vdata,
            'vdata': tar_vdata,
        })

        ptl_codes = self.get_inp_codes(
            ptl_inp_seq, ptl_inp_seq_weight,
            inp_img_code, tar_img_code
        )

        edit_ops, eo_lls = self.sample_edit_ops(ptl_codes, ptl_inp_seq_weight)

        ps_inp_seq = []

        ps_inp_seq = torch.zeros(len(population), self.inp_ms).long().to(self.device)
        ps_inp_seq_weight = torch.zeros(len(population), self.inp_ms).float().to(self.device)
        
        ps_start_seq = torch.zeros(len(population), self.edit_ms).long().to(self.device)

        ptl_inp_progs = []
        for _,_,iprog,_ in population:
            ptl_inp_progs.append(self.ex.TLang.tensor_to_tokens(iprog,True))
        
        for i,(eo, ptl_is) in enumerate(zip(edit_ops, ptl_inp_progs)):
                        
            ps_is, ps_ts = self.ex.format_tokens_for_edit(
                ptl_is,
                eo[0],
                eo[1],
                []
            )

            ps_ts.pop(-1)

            if len(ps_ts) != 1:
                print("Bad behavior")
                continue

            ps_start_seq[i,:1] = self.ex.TLang.tokens_to_tensor(ps_ts)
            
            ps_is_ts = self.ex.TLang.tokens_to_tensor(ps_is)

            try:
                ps_inp_seq[i,:ps_is_ts.shape[0]] = ps_is_ts
                ps_inp_seq_weight[:, :ps_is_ts.shape[0]] = 1.0
            except Exception as e:
                if VERBOSE:
                    print(f"Failed to format edit input tensor with {e}")
                    
        ps_codes = self.get_inp_codes(
            ps_inp_seq, ps_inp_seq_weight,
            inp_img_code, tar_img_code
        )

        samples = {i: None for i in range(len(population))}

        for edit_types, max_eos_len in self.ET2MEL:
            
            match_inds = torch.tensor([i for i, (eot, _) in enumerate(edit_ops) if eot in edit_types], device=ps_codes.device).long()
            
            sub_info = {
                'batch': match_inds.shape[0],
                'bprefix': ps_codes[match_inds],
                'bseqs': ps_start_seq[match_inds],
                'END_TOKEN_IND': self.ex.TLang.T2I[self.ex.END_TOKEN]
            }
            sub_samples = is_fn(
                self.token_net,
                sub_info,
                max_eos_len,
                is_fn_args
            )

            for i,mi in enumerate(match_inds.tolist()):
                samples[mi] = sub_samples[i]

        for k,v in samples.items():
            assert v is not None, 'NEED TO FIX ET2MEL'
        
        mutations = []
        eerr = 1e-8
        etot = 1e-8

        extra_ret = None
        
        for pop_ind, preds in samples.items():

            if len(preds) == 0:
                mutations.append([])
                continue

            if self.domain.args.infer_is_mode == 'sample':
                if len(preds) != 1:
                    print("Bad behavior")
                    continue

            _mutations = []
            for pred in preds:
                raw_ll, raw_out = pred

                tokens_out = self.ex.TLang.tensor_to_tokens(raw_out, True)
                
                if not ('$' in tokens_out[0] and tokens_out[-1] == self.ex.END_TOKEN):
                    print("Bad behavior")
                    continue
                
                eos = tokens_out[1:-1]
                eot, eol = edit_ops[pop_ind]
                eot = eot[1:]
                                
                start_tokens = ptl_inp_progs[pop_ind]

                edit_info = (eot, eol, eos)
                
                if pop_ind == 0:
                    extra_ret = deepcopy(edit_info)
                    
                etot += 1
                try:
                    tokens = self.prog_diff.make_edit_op(start_tokens, edit_info)
                except Exception as e:
                    eerr += 1
                    if VERBOSE:
                        print(f"Failed edit op with {e}")
                        print(start_tokens)
                        print(edit_info)
                    continue


                expr = ' '.join(tokens)

                exc, mval, info = self.domain.get_vis_metric(
                    expr,
                    tar_pixels[0]
                )

                if exc is None or mval is None:
                    if VERBOSE:
                        print(f"Failed exec {expr}")
                    eerr += 1
                    continue

                edit_tokens = self.prog_diff.normalize_program(tokens)

                try:
                    edit_tensor = self.ex.TLang.tokens_to_tensor(edit_tokens)
                except Exception as e:
                    if VERBOSE:
                        print(f"Failed to get tokens to tensor with {e}")
                    continue
                    
                _mutations.append((                    
                    raw_ll.item() + eo_lls[pop_ind],
                    mval,
                    edit_tensor,
                    exc
                ))

            mutations.append(_mutations)

        if ret_extra:
            return mutations, eerr/ etot, extra_ret
            
        return mutations, eerr / etot

    
class EditTokenNet(nn.Module):
    def __init__(
        self,
        domain,
        num_venc_tokens,
        max_seq_len
    ):
        super(EditTokenNet, self).__init__()
        
        self.domain = domain
        args = domain.args

        self.ex = domain.executor

        self.device = domain.device

        self.img_mp = args.max_prim_enc
        
        # max number of tokens from sequence
        self.ms = max_seq_len
        self.mp = num_venc_tokens
        
        self.nl = args.num_layers
        self.nh = args.num_heads
                
        self.bs = args.batch_size
        self.dropout = args.dropout
        
        # language number of tokens
        self.nt = domain.executor.TLang.get_num_tokens()
        self.neop = len(domain.executor.EOT2I)
        self.hd = args.hidden_dim 
        
        self.token_enc_net = nn.Embedding(self.nt, self.hd)
        
        self.token_head = mu.DMLP(
            self.hd, self.hd, self.hd // 2, self.nt, self.dropout)

        self.op_type_net = mu.DMLP(
            self.hd, self.hd, self.hd // 2, self.neop, self.dropout)
        
        self.op_loc_net = mu.DMLP(
            self.hd, self.hd, self.hd // 2, self.neop, self.dropout)
        
        self.pos_enc = nn.Embedding(self.ms+self.mp, self.hd)
        self.pos_arange = torch.arange(self.ms+self.mp).unsqueeze(0)

        self.ptl_pos_enc = nn.Embedding(self.mp, self.hd)
        self.ptl_pos_arange = torch.arange(self.mp).unsqueeze(0)
        
        self.attn_mask = self.generate_attn_mask()

        
        utils.log_print(f"Loading default arch with {(self.nl, self.nh, self.hd)}", args)
        
        self.attn_layers = nn.ModuleList(
            [mu.AttnLayer(self.nh, self.hd, self.dropout) for _ in range(self.nl)]
        )
            
        
    def generate_attn_mask(self):
        return mu._generate_attn_mask(self)

    def generate_key_mask(self, num):
        return mu._generate_key_mask(self, num)

    # main training function, takes in codes from encoder + sequence
    
    def infer_eop_seq(self, codes, seq):
        return mu._infer_prog(self, codes, seq)

    def infer_eop_tl(self, codes):

        embs = get_transform_embs(self, codes)

        type_preds = self.op_type_net(embs[:,0,:])

        loc_preds = self.op_loc_net(embs)

        return type_preds, loc_preds


    
def get_transform_embs(net, codes):
                                               
    out = codes
    
    out += net.ptl_pos_enc(net.ptl_pos_arange.repeat(codes.shape[0], 1).to(net.device))
        
    attn_mask = net.attn_mask[:net.mp, :net.mp].to(net.device)

    key_mask = net.generate_key_mask(codes.shape[0])[:,:net.mp].to(net.device)
        
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask, key_mask)
        
    return out[:,(2 * net.img_mp):,:]
        
    
    
