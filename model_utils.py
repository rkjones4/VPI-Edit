import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# penalties for violating different constraints
MIN_LL_THRESH = -1000
MIN_LL_PEN = -10000
MASK_LL_PEN = -100000.

        
def load_vis_encoder(domain):
    if domain.name == 'csg3d':
        return load_3d_vis_encoder(domain)
    else:
        return load_2d_vis_encoder(domain)

def load_3d_vis_encoder(domain):
    net = V3DCNN(
        inp_dim = domain.executor.VDIM,
        max_prim_encs = domain.args.max_prim_enc,                
        out_dim = domain.args.venc_hidden_dim,
        drop= domain.args.dropout
    )
    return net

def load_2d_vis_encoder(domain):    
    inp_shape = domain.executor.get_input_shape()        
    assert len(inp_shape) == 3
    enc = V2DCNN(inp_shape[2], domain.args.venc_hidden_dim, domain.args.dropout)
    return enc
    
class TDecNet(nn.Module):
    def __init__(
        self,
        domain,
        num_venc_tokens,
        max_seq_len
    ):
        super(TDecNet, self).__init__()
        
        self.domain = domain
        args = domain.args

        self.ex = domain.executor

        try:
            self.beams = args.beams
        except:
            pass
        
        self.device = domain.device

        # max number of tokens from sequence
        self.ms = max_seq_len
        self.mp = num_venc_tokens
        
        self.nl = args.num_layers
        self.nh = args.num_heads
                
        self.bs = args.batch_size
        self.dropout = args.dropout
        
        # language number of tokens
        self.nt = domain.executor.TLang.get_num_tokens()
        self.hd = args.hidden_dim 
        
        self.token_enc_net = nn.Embedding(self.nt, self.hd)        
        self.token_head = SDMLP(self.hd, self.nt, self.dropout)
        
        self.pos_enc = nn.Embedding(self.ms+self.mp, self.hd)
        self.pos_arange = torch.arange(self.ms+self.mp).unsqueeze(0)
        
        self.attn_mask = self.generate_attn_mask()

        
        utils.log_print(f"Loading default arch with {(self.nl, self.nh, self.hd)}", args)
        
        self.attn_layers = nn.ModuleList(
            [AttnLayer(self.nh, self.hd, self.dropout) for _ in range(self.nl)]
        )                    
        
    def generate_attn_mask(self):
        return _generate_attn_mask(self)

    def generate_key_mask(self, num):
        return _generate_key_mask(self, num)

    # main training function, takes in codes from encoder + sequence
    def infer_prog(self, codes, seq):
        return _infer_prog(self, codes, seq)
 

class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)

class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
                
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    
class SDMLP(nn.Module):
    def __init__(self, ind, odim, DP):
        super(SDMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, odim)
        self.l2 = nn.Linear(odim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        return self.l2(x)

        
# 2D pixel CNN encoder
class V2DCNN(nn.Module):
    def __init__(self, inp_dim, code_size, drop):

        super(V2DCNN, self).__init__()

        self.max_prim_encs = 16
        self.inp_dim = inp_dim
        
        # Encoder architecture
        self.conv1 = nn.Conv2d(
            in_channels=self.inp_dim, out_channels=32, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )
        self.b1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )        
        self.b2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )                                                  
        self.b3 = nn.BatchNorm2d(num_features=128)
        
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )                                                   
        self.b4 = nn.BatchNorm2d(num_features=256)

        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b4,
        )
        
        self.ll = DMLP(256, 256, 256, code_size, drop)
                        
    def forward(self, x):
        x1 = x.view(-1, self.inp_dim, 64, 64)
        x2 = self._encoder(x1)

        x2 = x2.view(-1, 256, self.max_prim_encs)            
        x2 = x2.transpose(1, 2)
                        
        return self.ll(x2)


class V3DCNN(nn.Module):
    def __init__(
            self,
            inp_dim,
            max_prim_encs,
            out_dim,
            drop
    ):

        assert inp_dim == 32
        assert max_prim_encs == 8
        self.max_prim_encs = 8
        
        self.out_dim = out_dim        

        self.inp_dim = inp_dim
        
        super(V3DCNN, self).__init__()
                                                
        # Encoder architecture
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b4 = nn.BatchNorm3d(num_features=256)


        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b4,
        )
        
        self.ll = DMLP(256, 256, 256, self.out_dim, drop)
            
    def forward(self, x):
        x1 = x.view(-1, 1, self.inp_dim, self.inp_dim, self.inp_dim)
        x2 = self._encoder(x1)
        x2 = x2.view(-1, 256, self.max_prim_encs)            
        x2 = x2.transpose(1, 2)        
        x3 = self.ll(x2)
        o = x3.view(-1, self.max_prim_encs, self.out_dim)
        return o
    
######## TRANSFORMER

class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()
        self.nh = nh
        self.hd = hd

        self.self_attn = torch.nn.MultiheadAttention(self.hd, self.nh)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, _src, attn_mask, key_padding_mask):
        
        src = _src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask
        )[0]
        
        
        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(src), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)

        return src.transpose(0, 1)
    
# generate attention mask for transformer auto-regressive training
# first mp spaces have fully connected attention, as they are the priming sequence of visual encoding
def _generate_attn_mask(net):
    sz = net.mp + net.ms
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).T
    mask[:net.mp, :net.mp] = 0.
    return mask

def _generate_encoder_attn_mask(net):
    sz = net.mp + net.ms
    mask = torch.zeros(sz, sz).float()
    return mask

# generate key mask for transformer auto-regressive training
def _generate_key_mask(net, num):
    sz = net.mp + net.ms
    mask = torch.zeros(num, sz).bool()
    return mask

def _generate_encoder_key_mask(net, seq_lens):
    sz = net.mp + net.ms
    mask = torch.zeros(len(seq_lens), sz)
    for i, j in enumerate(seq_lens):
        mask[i,j+1:] = 1.0
        
    return mask.bool()
    
# main forward process of transformer, encode tokens, add PE, run through attention, predict tokens with MLP
def _infer_prog(net, codes, seq, seq_lens=None):
    token_encs = net.token_enc_net(seq).view(-1, net.ms, net.hd)
                                            
    out = torch.cat((codes.view(codes.shape[0], net.mp, net.hd), token_encs), dim = 1)        
    out += net.pos_enc(net.pos_arange.repeat(codes.shape[0], 1).to(net.device))
        
    attn_mask = net.attn_mask.to(net.device)

    if seq_lens is not None:
        key_mask = net.generate_key_mask(seq_lens).to(net.device)
    else:
        key_mask = net.generate_key_mask(codes.shape[0]).to(net.device)
        
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask, key_mask)
        
    seq_out = out[:,net.mp:,:]

    token_out = net.token_head(seq_out)
        
    return token_out

celoss = torch.nn.CrossEntropyLoss(reduction='none')

def calc_token_loss(preds, targets, weights):

    if weights is None:
        weights = 1
        assert len(targets.shape) == 1
        total = targets.shape[0] * 1.

        loss = celoss(preds, targets).mean()
        with torch.no_grad():
            corr = (preds.argmax(dim=1) == targets).float().sum()
            
    else:
        loss = (celoss(preds, targets) * weights).sum() / (weights.sum() + 1e-8)
        with torch.no_grad():
            total = weights.sum()
            corr = ((preds.argmax(dim=1) == targets).float() * weights).sum()
                    
        
    return loss, corr, total

def top_k_top_p_filtering(
    logits,
    top_k = 0,
    top_p = 1.0,
    filter_value = -float("Inf"),
    min_tokens_to_keep = 1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def inner_search_beam_logic(
    net,
    info,
    max_len,
    extra_args
):
    
    batch = info['batch']
    beams = extra_args['beams']    

    raw_bprefix = info['bprefix']
    raw_bseqs = info['bseqs']
        
    bprefix = raw_bprefix.unsqueeze(1).repeat(1,beams,1,1).view(
        batch*beams, raw_bprefix.shape[1], raw_bprefix.shape[2]
    )
    bseqs = raw_bseqs.unsqueeze(1).repeat(1,beams,1).view(
        batch*beams, raw_bseqs.shape[1],
    )
    
    device = bseqs.device

    blls = torch.zeros(batch, beams, device=device)        
    blls[:,1:] += MIN_LL_PEN
    blls = blls.flatten()
    
    if 'bsinds' not in info:
        bsinds = torch.zeros(batch * beams, device=device).long()            
    else:
        raw_bsinds = info['bsinds']
        bsinds = raw_bsinds.unsqueeze(1).repeat(1, beams).flatten()
                            
    fin_progs = {i:[] for i in range(batch)}

    fin_count = torch.zeros(
        batch,
        device=device
    )
        
    break_cond = torch.zeros(batch, device=device).bool()
        
    fin_lls = [[] for _ in range(batch)]
        
    dummy_arange = torch.arange(beams * batch, device=bprefix.device)        

    max_ind = bseqs.shape[1] - 1 

    if 'END_TOKEN_IND' in info:    
        END_TOKEN_IND = info['END_TOKEN_IND']
        bqc = None
        TTNN = None
        _extra = 1
    else:
        END_TOKEN_IND = None
        raw_bqc = info['bqc']
        bqc = raw_bqc.unsqueeze(1).repeat(1, beams).flatten()
        TTNN = info['ttnn']
        if '_extra' in info:
            _extra = info['_extra']
        else:
            _extra = 1
            
        assert bqc is not None
        assert TTNN is not None
    
    for PL in range(max_len-1): 
            
        E_blls = blls.view(batch, beams)
            
        for i in (fin_count >= beams).nonzero().flatten():
            fin_nll = -1 * torch.tensor([
                np.partition(fin_lls[i], beams-1)[beams-1]
            ], device=device)

            if (E_blls[i] < fin_nll).all():
                break_cond[i] = True
            
        if break_cond.all():
            break

        exp_bpreds = net.is_eval_fn(bprefix, bseqs)
        bpreds = exp_bpreds[dummy_arange, bsinds]
                        
        bdist = torch.log(torch.softmax(bpreds, dim = 1) + 1e-8)            
            
        beam_liks, beam_choices = torch.topk(bdist, beams)
            
        next_liks = (beam_liks + blls.view(-1, 1)).view(batch, -1)

        E_ll, E_ranked_beams = torch.sort(next_liks,1,True)

        blls = E_ll[:,:beams].flatten()

        ranked_beams = E_ranked_beams[:,:beams]

        R_beam_choices = beam_choices.view(batch, -1)

        nt = torch.gather(R_beam_choices,1,ranked_beams).flatten()

        old_index = (torch.div(ranked_beams, beams).float().floor().long() + (torch.arange(batch, device=device) * beams).view(-1, 1)).flatten()
            
        bseqs  = bseqs[old_index].clone()
        bsinds = bsinds[old_index].clone() + 1
        bsinds = torch.clamp(bsinds, 0, max_ind)
        bseqs[dummy_arange, bsinds] = nt

        bprefix = bprefix[old_index]

        if END_TOKEN_IND is not None:
            fin_inds = (nt == END_TOKEN_IND).nonzero().flatten().tolist()
        else:        
            bqc = bqc[old_index].clone()
            bqc -= 1            
            bqc += TTNN[nt]            
            fin_inds = (bqc == 0.).nonzero().flatten().tolist()
                            
        for i in fin_inds:
            if blls[i] > MIN_LL_THRESH:
                beam_ind = i // beams
                _ll = blls[i].detach().cpu()
                fin_progs[beam_ind].append((
                    _ll,
                    bseqs[i,:bsinds[i] + _extra]
                ))
                fin_count[beam_ind] += 1
                fin_lls[beam_ind].append(-1 * _ll)                        
                        
            blls[i] += MIN_LL_PEN

    return fin_progs
    
