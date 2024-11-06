import scipy.optimize as sopt
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import math
from copy import deepcopy
import time

edit_ST = '@'

MAX_AP_SAMPLE_NUM = None
MAX_EOP_NUM = None

OKM = {
    'comb_add_after': '$ACA',
    'comb_add_before': '$ACB',
    'comb_rm': '$CR',
    'comb_mod': '$CM',
    'trans_add': '$TA',
    'trans_rm': '$TR',
    'trans_mod': '$TM',
    'param_mod': '$PM',
}

CS_TOKENS = set(['union', 'diff', 'inter'])
TS_TOKENS = set(['reflect'])
OP_ORD = ['move', 'rot', 'scale', 'prim', 'TSI', 'CSI']

NORM_OPS = set(['move', 'scale', 'rot'])

def split_to_sub_progs(tokens):
    
    if tokens[0] == 'START':
        assert tokens[-1] == 'END'
        if len(tokens) == 2:
            return []
        return split_to_sub_progs(tokens[1:-1])

    if tokens[0] == 'POS':
        tpe = 'POS'
    elif tokens[0] == 'NEG':
        tpe = 'NEG'
    else:
        assert False, f'bad prog {tokens}'
           
    rest = [t for t in tokens[1:]]

    cur = []
    
    while len(rest) > 0:
        if rest[0] in ('POS', 'NEG'):
            break

        t = rest.pop(0)        
        cur.append(t)

    if len(rest) == 0:
        pos, neg = [], []
    else:
        pos, neg = split_to_sub_progs(rest)
    if tpe == 'POS':        
        return [cur] + pos, neg
    elif tpe == 'NEG':
        return pos, [cur] + neg
    else:
        assert False

def calc_edit_cost(ex, a, b, INDEX):

    assert not (a is None and b is None)

    if a is None:
        bexp = b.expand(ex)
        return 2 + len(bexp)

    if b is None:
        return 1

    atype = None
    btype = None

    for tt in ('prim', 'CSI', 'TSI'):
        if tt in a.OI:
            assert atype is None
            atype = tt
        if tt in b.OI:
            assert btype is None
            btype = tt

    assert atype is not None and btype is not None
    
    if atype == 'prim' and btype == 'prim':
        return calc_local_edit_cost(a, b)

    elif atype == 'CSI' or btype == 'CSI':
        return calc_csi_edit_cost(ex, a, b, INDEX)
    
    else:
        return calc_tsi_edit_cost(ex, a, b, INDEX)

def calc_csi_edit_cost(ex, a, b, INDEX):

    if 'CSI' in a.OI and 'CSI' in b.OI:
        cost = 0
        cost += get_edit_cost(ex, a.OI['CSI'][1], b.OI['CSI'][1], INDEX)
        cost += get_edit_cost(ex, a.OI['CSI'][2], b.OI['CSI'][2], INDEX)
        cost += calc_local_edit_cost(a, b)

        if a.OI['CSI'][0] != b.OI['CSI'][0]:
            cost += 2
        
        return cost
        
    elif 'CSI' in a.OI:

        c1 = a.OI['CSI'][1]
        c2 = a.OI['CSI'][2]

        a.stash()
        c1.stash()
        c2.stash()

        merge_into_child(ex, a, c1, 'CSI')
        
        cost_rm_2 = get_edit_cost(ex, c1, b, INDEX)

        a.reset_from_stash()
        a.stash()
        
        merge_into_child(ex, a, c2, 'CSI')

        cost_rm_1 = get_edit_cost(ex, c2, b, INDEX)

        a.reset_from_stash()

        cost = 1 + min(cost_rm_1, cost_rm_2)

        c1.reset_from_stash()
        c2.reset_from_stash()

        return cost
        
    elif 'CSI' in b.OI:

        a.stash()
        
        if len(b.OI) == 1:            
            pnode, split_ind, _, cost = simple_split_to_parent(a, b, INDEX, no_op=True)
        else:
            pnode, split_ind, _, cost = split_to_parent(a, b, INDEX, no_op=True)

        assert len(pnode.OI['CSI']) == 2
        cnode = pnode.OI['CSI'][1]

        assert len(b.OI['CSI']) == 3
        by = b.OI['CSI'][1]
        bz = b.OI['CSI'][2]
                
        by_ec = get_edit_cost(ex, cnode, by, INDEX) + get_edit_cost(ex, None, bz, INDEX)
        bz_ec = get_edit_cost(ex, cnode, bz, INDEX) + get_edit_cost(ex, None, by, INDEX)

        byexp = by.expand(ex)
        bzexp = bz.expand(ex)

        if bzexp[0] in CS_TOKENS:
            cost += by_ec
        elif byexp[0] in CS_TOKENS:
            cost += bz_ec
        elif by_ec <= bz_ec:
            cost += by_ec
        else:
            cost += bz_ec

        cost += calc_local_edit_cost(pnode, b)
                
        a.reset_from_stash()
        pnode.reset_from_stash()

        return cost
        
    else:
        assert False
    
def calc_tsi_edit_cost(ex, a, b, INDEX):

    if 'TSI' in a.OI and 'TSI' in b.OI:

        cost = 0
        cost += calc_local_edit_cost(a, b)
        cost += get_edit_cost(ex, a.OI['TSI'][-1], b.OI['TSI'][-1], INDEX)

        atsi = a.OI['TSI'][:-1]
        btsi = b.OI['TSI'][:-1]

        if atsi == btsi:
            pass
        elif atsi[0] == btsi[0]:
            cost += len(btsi) 
        else:
            cost += len(btsi) + 1

        return cost

    elif 'TSI' in a.OI:

        child = a.OI['TSI'][-1]
        
        a.stash()
        child.stash()
        merge_into_child(ex, a, child, "TSI")
        
        cost = get_edit_cost(ex, child, b, INDEX)

        a.reset_from_stash()
        child.reset_from_stash()        
        
        cost += 1
        
        return cost
        

    elif 'TSI' in b.OI:

        btsi = b.OI['TSI'][:-1]

        cost = get_edit_cost(ex, a, b.OI['TSI'][-1], INDEX)

        pl = NormalForm(None, [], {'node':{}})
        
        cost += calc_local_edit_cost(pl, b)

        cost += len(btsi) + 1
        
        return cost                

    else:
        assert False

def calc_local_edit_cost(a, b):
    cost = 0

    amiss = set()
    bmiss = set([bfn for bfn in b.OI if bfn not in ['TSI', 'CSI']])
    
    for afn, aprms in a.OI.items():
        if afn in ['TSI', 'CSI']:
            continue
        
        if afn in bmiss:
            bmiss.remove(afn)
            bprms = b.OI[afn]
            if aprms == bprms:                
                continue
            else:
                # edit op + prms
                cost += 1 + len(bprms[0])
        else:
            amiss.add(afn)

    while len(amiss) > 0 and len(bmiss) > 0:
        afn = amiss.pop()
        bfn = bmiss.pop()
        bprms = b.OI[bfn]
        
        # edit op + fn + prms
        cost += 1 + 1 + len(bprms[0])
    
    for afn in amiss:
        # edit op
        cost += 1 
        
    for bfn in bmiss:
        bprms = b.OI[bfn]
        # edit op + fn +prms
        cost += 1 + 1 + len(bprms[0])

    return cost
        

def get_edit_cost(ex, a, b, INDEX):
    
    if a is None:
        aid = None
    else:    
        aid = a.node_id

    if b is None:
        bid = None
    else:
        bid = b.node_id

    if (aid, bid) not in INDEX['cost']:
        cost = calc_edit_cost(ex, a, b, INDEX)    
        INDEX['cost'][(aid, bid)] = cost
        
    return INDEX['cost'][(aid, bid)]


def norm_op_lang(ex, op, info):

    if ex.name == 'csg2d':
        return norm_op_2d(ex, op, info)
    elif ex.name == 'csg3d':
        return norm_op_3d(ex, op, info)
    else:
        assert False, f'bad ex name {ex.name}'

def round_val(val, ex, fltype):
    tf_vals = ex.TFLT_INFO[fltype]
    f_vals = ex.LFLT_INFO[fltype]
    ind = (tf_vals - val).abs().argmin().item()
    return f_vals[ind]
        
def norm_op_2d(ex, op, info):

    finfo = [[ex.T2F(op,_i) for _i in i] for i in info]

    if op == 'move':
        assert len(info[0]) == 2
        new = [sum([i[0] for i in finfo]), sum([i[1] for i in finfo])]

    elif op == 'scale':
        assert len(info[0]) == 2
        new = [math.prod([i[0] for i in finfo]), math.prod([i[1] for i in finfo])]

    elif op == 'rot':
        assert len(info[0]) == 1
        new = [sum([i[0] for i in finfo])]
    else:
        assert False

    news = [ex.F2T(op, round_val(v, ex, op)) for v in new]            
    return [op] + news


def norm_op_3d(ex, op, info):

    finfo = [[ex.T2F(op,_i) for _i in i] for i in info]

    if op == 'move':
        assert len(info[0]) == 3
        new = [
            sum([i[0] for i in finfo]),
            sum([i[1] for i in finfo]),
            sum([i[2] for i in finfo]),
        ]

    elif op == 'scale':
        assert len(info[0]) == 3
        new = [
            math.prod([i[0] for i in finfo]),
            math.prod([i[1] for i in finfo]),
            math.prod([i[2] for i in finfo]),
        ]

    elif op == 'rot':
        assert len(info[0]) == 3
        new = [
            sum([i[0] for i in finfo]),
            sum([i[1] for i in finfo]),
            sum([i[2] for i in finfo]),
        ]
    else:
        assert False

    news = [ex.F2T(op, round_val(v, ex, op)) for v in new]            
    return [op] + news

def normalize_op(ex, op, info):    
    if op == 'CSI':
        if info[0] == 'dummy':
            assert len(info) == 2
            assert isinstance(info[1], NormalForm)
            return [info[1]]
        else:
            return info

    if op == 'TSI':
        if info[0] == 'dummy':
            assert len(info) == 2
            assert isinstance(info[1], NormalForm)
            return [info[1]]
        else:
            return info

    if len(info) > 1:
        assert op in NORM_OPS, f'op {op} was NOT in NORM OPS unexpectedly {info}'
        return norm_op_lang(ex, op, info)    
    else:
        assert len(info) == 1
        
    return [op] + list(info[0])

class ProgNormalForm:
    def __init__(self, pos_sub_progs, neg_sub_progs):
        self.pos_sub_progs = pos_sub_progs
        self.neg_sub_progs = neg_sub_progs
            
    def convert_to_prog(self, ex):
        pos_sub_progs = []
        neg_sub_progs = []
        
        for node in self.pos_sub_progs:
            pos_sub_progs.append(node.expand(ex))

        for node in self.neg_sub_progs:
            neg_sub_progs.append(node.expand(ex))
            
        tokens = ['START']

        for s in pos_sub_progs:
            if len(s) == 0:
                continue
            tokens += ['POS'] + s

        for s in neg_sub_progs:
            if len(s) == 0:
                continue
            tokens += ['NEG'] + s
            
        return tokens + ['END']

    def convert_to_edit_prog(self, ex, edit_node, edit_fn):
        pos_sub_progs = []
        neg_sub_progs = []
        
        for node in self.pos_sub_progs:
            pos_sub_progs.append(node.edit_expand(ex, edit_node, edit_fn))

        for node in self.neg_sub_progs:
            neg_sub_progs.append(node.edit_expand(ex, edit_node, edit_fn))
            
        tokens = ['START']

        for s in pos_sub_progs:
            if len(s) == 0:
                continue
            tokens += ['POS'] + s

        for s in neg_sub_progs:
            if len(s) == 0:
                continue
            tokens += ['NEG'] + s
            
        return tokens + ['END']
    
    def format_for_edit(self, ex, edit_node, edit_fn):

        if edit_node is None:
            comb = self.convert_to_prog(ex)
            return comb, 0
        
        comb = self.convert_to_edit_prog(ex, edit_node, edit_fn)

        assert ' '.join(comb).count(edit_ST) == 1, 'missing edit loc token'
        
        edit_loc = None
        edit_seq = []
    
        for el, t in enumerate(comb):
            if edit_ST in t:
                assert edit_loc is None
                edit_loc = el
                edit_seq.append(t[:-1])
            else:
                edit_seq.append(t)

        return edit_seq, edit_loc
        
class NormalForm:

    def make_copy(self, o2n):

        C = NormalForm(None, [], None)

        o2n[self] = C
        
        if self.parent is not None:
            C.parent = o2n[self.parent]

        C.deleted = self.deleted
        C.node_id = self.node_id

        C.fn2i = deepcopy(self.fn2i)

        C.OI = {}
        
        for k,V in self.OI.items():
            NV = []

            for v in V:
                if isinstance(v, NormalForm):
                    c = v.make_copy(o2n)
                    NV.append(c)
                else:
                    NV.append(deepcopy(v))    
                        
            C.OI[k] = NV
                        
        assert len(self.ID_stash) == 0
        assert len(self.OI_stash) == 0
        assert len(self.FN2I_stash) == 0

        return C
    
    def __init__(self, ex, sub_prog, mapping):

        if mapping is not None:
            self.node_id = len(mapping['node'])
            mapping['node'][len(mapping['node'])] = self

        self.parent = None
        self.deleted = False
        OI = {
            'CSI': [],
            'TSI': []
        }
        ipc = 1
        fn = None
        while len(sub_prog) > 0:
            if ipc == 0:
                break
            fn = sub_prog.pop(0)
            ipc -= 1
            assert ex.TLang.get_out_type(fn) == ex.BASE_TYPE

            ipts = ex.TLang.get_inp_types(fn)
            
            prms = []
            
            for _i, ipt in enumerate(ipts):
                if ipt == ex.BASE_TYPE:
                    if fn in CS_TOKENS:
                        nf = NormalForm(ex, sub_prog, mapping)
                        nf.parent = self
                        prms.append(nf)
                        
                    elif fn in TS_TOKENS:
                        nf = NormalForm(ex, sub_prog, mapping)
                        nf.parent = self
                        prms.append(nf)
                        
                    else:
                        if fn not in OI:
                            OI[fn] = []
                        OI[fn].append(tuple(prms))                        
                        assert _i + 1 == len(ipts)
                        ipc += 1
                        break
                        
                else:
                    prms.append(sub_prog.pop(0))

                
            if fn in CS_TOKENS:
                assert len(OI['CSI']) == 0
                assert len(OI['TSI']) == 0
                OI['CSI'] = [fn] + prms
                fn = None                

            if fn in TS_TOKENS:
                assert len(OI['CSI']) == 0
                assert len(OI['TSI']) == 0
                OI['TSI'] = [fn] + prms
                fn = None                

                
        if fn is not None:
            if fn not in OI:
                OI[fn] = []
            OI[fn].append(tuple(prms))

        if len(OI['CSI']) == 0:
            OI.pop('CSI')

        if len(OI['TSI']) == 0:
            OI.pop('TSI')
            
        self.OI = OI
        self.fn2i = {}
        cov = set()
        
        for oop in OP_ORD:
            if oop not in self.OI:
                continue
            
            cov.add(oop)

            if mapping is not None:
                self.fn2i[oop] = len(mapping['token'])
                mapping['token'][len(mapping['token'])] = (self, oop)

        assert len(self.OI) == len(cov)

        self.ID_stash = []
        self.OI_stash = []
        self.FN2I_stash = []
        
    def stash(self):
        OI_stash = {}
        FN2I_stash = {}
        
        for k,v in self.OI.items():
            OI_stash[k] = v
            
        for k,v in self.fn2i.items():
            FN2I_stash[k] = v

        self.ID_stash.append(self.node_id)
        self.OI_stash.append(OI_stash)
        self.FN2I_stash.append(FN2I_stash)
            
    def reset_from_stash(self):
        self.node_id = self.ID_stash.pop(-1)
        self.OI = self.OI_stash.pop(-1)
        self.fn2i = self.FN2I_stash.pop(-1)

    def expand(self, ex, e = None):
        if self.deleted:
            return []
        
        if e is None:
            e = {
                'tokens': [],
            }        

        tokens = []
        for oop in OP_ORD:
            if oop not in self.OI:
                continue

            for t in normalize_op(ex, oop, self.OI[oop]):
                tokens.append(t)
                            
        for t in tokens:
            if isinstance(t, NormalForm):                
                t.expand(ex, e)
            else:
                e['tokens'].append(t)

        return e['tokens']

    def edit_expand(self, ex, edit_node, edit_fn, e = None):

        if self.deleted:
            return []
        
        if e is None:
            e = {
                'tokens': [],
            }

        tokens = []
        no_ST = True
        
        for oop in OP_ORD:
            if oop not in self.OI:
                continue

            for i,t in enumerate(normalize_op(ex, oop, self.OI[oop])):
                if (i==0 and self == edit_node and edit_fn == oop) or (
                    i==0 and self == edit_node and edit_fn is None and no_ST
                ):
                    if isinstance(t, NormalForm):
                        edit_node = t
                        tokens.append(t)
                        
                    else:
                        tokens.append(t + edit_ST)
                        no_ST = False
                else:                            
                    tokens.append(t)
                            
        for t in tokens:
            if isinstance(t, NormalForm):
                t.edit_expand(ex, edit_node, edit_fn, e)
            else:
                e['tokens'].append(t)

        return e['tokens']

def merge_values(ex, fn, tvl, bvl):
    assert len(tvl) == 1
    assert len(bvl) == 1
    rop = [norm_op_lang(ex, fn, tvl + bvl)[1:]]
    return rop
    
def split_to_parent(al, bl, INDEX, no_op=False):

    split_op = None
    padd = []

    al.node_id = f'split_bot_{al.node_id}'
    
    for oop in OP_ORD:
        if split_op is not None:
            break
        
        assert oop not in ['CSI']

        if oop in al.OI and oop in bl.OI:
            continue

        elif oop in bl.OI:
            padd.append(oop)            
                
        elif oop in al.OI:            
            split_op = oop
        else:
            continue

    assert split_op is not None
        
    ps_edits = []
    ps_cost = 0

    split_fn = split_op
    sind = al.fn2i[split_fn]

    PSAD = set()
    
    if len(padd) > 0:         
        while len(padd) > 0:
            add_fn = padd.pop(0)
            ps_ei = (al, 'trans_add', tuple([add_fn] + list(bl.OI[add_fn][0])))
            ps_cost += 1 + len(ps_ei[2])
            ps_edits.append(ps_ei)
            PSAD.add(add_fn)
        
    if no_op:
        pl = NormalForm(None, [], {'node':{}})
        pl.OI['CSI'] = ['dummy', al]
        
    else:        
        pl = NormalForm(None, [], INDEX)
        pl.OI['CSI'] = ['dummy', al]
        pl.fn2i['CSI'] = len(INDEX['token'])
        INDEX['token'][len(INDEX['token'])] = (pl, None)
        pl.parent = al.parent
        rectify_parent(pl, al)                
        al.parent = pl

    pl.node_id = f'par{al.node_id}'
    pl.stash()
    
    for oop in OP_ORD:
        assert oop not in ['CSI']
        if oop == split_fn:
            break
        
        if oop in al.OI:
            pl.OI[oop] = al.OI.pop(oop)
            pl.fn2i[oop] = al.fn2i.pop(oop)

        if oop in PSAD:
            pl.OI[oop] = bl.OI[oop]
            if not no_op:
                pl.fn2i[oop] = len(INDEX['token'])
                INDEX['token'][len(INDEX['token'])] = (pl, oop)
                                        
    return pl, sind, ps_edits, ps_cost

def rectify_parent(pl, al):
    if pl.parent is not None:
        if 'CSI' in pl.parent.OI:
            if pl.parent.OI['CSI'][1] == al:
                pl.parent.OI['CSI'][1] = pl
            elif pl.parent.OI['CSI'][2] == al:
                pl.parent.OI['CSI'][2] = pl
            else:
                assert False
        elif 'TSI' in pl.parent.OI:
            assert pl.parent.OI['TSI'][-1] == al
            pl.parent.OI['TSI'][-1] = pl

            
def simple_split_to_parent(al, bl, INDEX, no_op=False):

    assert len(bl.OI) == 1
    assert 'CSI' in bl.OI
        
    for oop in OP_ORD:
        if oop in al.OI:
            split_fn = oop
            break    
        
    sind = al.fn2i[split_fn]

    if no_op:
        pl = NormalForm(None, [], {'node':{}})        
        pl.OI['CSI'] = ['dummy', al]
    else:
        pl = NormalForm(None, [], INDEX)
        pl.OI['CSI'] = ['dummy', al]
        pl.fn2i['CSI'] = len(INDEX['token'])
        INDEX['token'][len(INDEX['token'])] = (pl, None)
        
        pl.parent = al.parent
        rectify_parent(pl, al)        
        al.parent = pl

    pl.node_id = f'par{al.node_id}'
    pl.stash()
                    
    return pl, sind, [], 0

def merge_into_child(ex, P, C, key, mapping=None):
    
    C.node_id = f'{P.node_id}m{C.node_id}'
    
    P.OI.pop(key)
    
    for k,v in P.OI.items():
        if k not in C.OI:
            C.OI[k] = v
            C.fn2i[k] = P.fn2i[k]
        else:
            cv = C.OI[k]
            mv = merge_values(ex, k, v, cv)
            C.OI[k] = mv
            
        if mapping is not None:
            mapping['token'][P.fn2i[k]] = (C, k)
                
    P.OI = {key: ['dummy', C]}
    P.fn2i = {key: P.fn2i[key]}

    if 'TSI' in C.OI and C.OI['TSI'][0] == 'dummy':
        merge_into_child(ex, C, C.OI['TSI'][1], 'TSI', mapping)

    elif 'CSI' in C.OI and C.OI['CSI'][0] == 'dummy':
        merge_into_child(ex, C, C.OI['CSI'][1], 'CSI', mapping)
                
def format_for_edit(ex, spl, m=None):

    nfl = []
    
    for sp in spl:
        nf = NormalForm(ex, deepcopy(sp), m)
        nfl.append(nf)
        
    return nfl

def normalize_program(ex, tokens):
    pos_sprogs, neg_sprogs = split_to_sub_progs(tokens)

    POS_SL = format_for_edit(ex, pos_sprogs, None)
    NEG_SL = format_for_edit(ex, neg_sprogs, None)

    prog_NF = ProgNormalForm(POS_SL, NEG_SL)            

    return prog_NF.convert_to_prog(ex)
    
    

def make_mod_comb_edit(node, cfn, tprms):
    index = node.fn2i[cfn]
    assert len(tprms) == 1
    einfo = (index, 'comb_mod', tuple(tprms[0]))
    ecost = 1 + len(einfo[2])
    return einfo, ecost
    
def make_mod_prm_edit(node, cfn, tprms):
    index = node.fn2i[cfn]    

    assert len(tprms) == 1        
    einfo = (index, 'param_mod', tuple(tprms[0]))
        
    ecost = 1 + len(einfo[2])
    
    return einfo, ecost
    
def make_mod_fn_edit(node, cfn, tfn, tprms):
    index = node.fn2i[cfn]
    assert len(tprms) == 1
    einfo = (index, 'trans_mod', tuple([tfn] + list(tprms[0])))
    ecost = 2 + len(einfo[2])
    return einfo, ecost

def make_rm_fn_edit(node, cfn):
    index = node.fn2i[cfn]
    einfo = (index, 'trans_rm', None)
    ecost = 1
    return einfo, ecost

def make_add_fn_edit(node, tfn, tprms):
    index = node
    assert len(tprms) == 1
    einfo = (index, 'trans_add', tuple([tfn] + list(tprms[0])))
    ecost = 2 + len(einfo[2])
    return einfo, ecost

def find_tsi_edit(anode, atsi, btsi, INDEX):
    index = anode.fn2i['TSI']

    if atsi[:-1] == btsi[:-1]:
        return [], 0    
    elif atsi[0] == btsi[0]:        
        einfo = (index, 'param_mod', tuple(btsi[1:-1]))
    else:
        einfo = (index, 'trans_mod', tuple(btsi[0:-1]))

    ecost = 1 + len(einfo[2])
    return [(einfo, None)], ecost

def find_comb_best_edits(ex, al, bl, INDEX):

    if 'CSI' in al.OI and 'CSI' in bl.OI:

        assert len(al.OI['CSI']) == 3
        assert len(bl.OI['CSI']) == 3
                
        ne, nc = find_local_best_edits(al, bl, INDEX)
                
        c1, l1e, l1c = find_best_edits(ex, al.OI['CSI'][1], bl.OI['CSI'][1], INDEX)
        c2, l2e, l2c = find_best_edits(ex, al.OI['CSI'][2], bl.OI['CSI'][2], INDEX)
        
        al.OI['CSI'][1] = c1
        al.OI['CSI'][2] = c2
        comb_edits = ne + l1e + l2e
        comb_cost = nc + l1c + l2c
        
        if al.OI['CSI'][0] != bl.OI['CSI'][0]:
            ce, cc = make_mod_comb_edit(al, 'CSI', [[bl.OI['CSI'][0]]])        
            comb_edits += [(ce,None)]
            comb_cost += cc
                        
        return al, comb_edits, comb_cost
        
    elif 'CSI' in al.OI:
        
        assert len(al.OI['CSI']) == 3

        c1 = al.OI['CSI'][1]
        c2 = al.OI['CSI'][2]

        al.stash()
        c1.stash()
        c2.stash()

        merge_into_child(ex, al, c1, 'CSI')
        
        cost_rm_2 = get_edit_cost(ex, c1, bl, INDEX)

        al.reset_from_stash()
        al.stash()
        
        merge_into_child(ex, al, c2, 'CSI')

        cost_rm_1 = get_edit_cost(ex, c2, bl, INDEX)

        al.reset_from_stash()

        if cost_rm_1 <= cost_rm_2:
            rm_child = c1
            kp_child = c2
            kp_ind = 2
        else:
            rm_child = c2
            kp_child = c1
            kp_ind = 1
            
        # calc edits

        rm_edit = (rm_child, 'comb_rm', None)
        comb_edits = [
            (rm_edit, None)
        ]
        comb_cost = 1
        
        kp_child, kce, kcc = find_best_edits(ex, kp_child, bl, INDEX)
        
        al.OI['CSI'][kp_ind] = kp_child
        
        for eop, econd in kce:
            if econd is not None:
                ncond = econd + [rm_edit]
            else:
                ncond = [rm_edit]
            comb_edits.append((eop, ncond))
        
        comb_cost += kcc
            
        c1.reset_from_stash()
        c2.reset_from_stash()
        
        return al, comb_edits, comb_cost
        
    elif 'CSI' in bl.OI:
        
        al.stash()
        
        if len(bl.OI) == 1:
            pnode, split_ind, ps_edits, ps_cost = simple_split_to_parent(al, bl, INDEX)
        else:
            pnode, split_ind, ps_edits, ps_cost = split_to_parent(al, bl, INDEX)
            
        assert len(pnode.OI['CSI']) == 2
        cnode = pnode.OI['CSI'][1]

        assert len(bl.OI['CSI']) == 3
        by = bl.OI['CSI'][1]
        bz = bl.OI['CSI'][2]
        
        by_ec = get_edit_cost(ex, cnode, by, INDEX) + get_edit_cost(ex, None, bz, INDEX)
        bz_ec = get_edit_cost(ex, cnode, bz, INDEX) + get_edit_cost(ex, None, by, INDEX)

        byexp = by.expand(ex)
        bzexp = bz.expand(ex)

        comb_edits = [(p, None) for p in ps_edits]
        comb_cost = ps_cost

        
        if bzexp[0] in CS_TOKENS:
            sm = 'after'
        elif byexp[0] in CS_TOKENS:
            sm = 'before'
        elif by_ec <= bz_ec:
            sm = 'after'
        else:
            sm = 'before'

        cm2a = bl.OI['CSI'][0]
        assert cm2a in CS_TOKENS
        
        if sm == 'after':
            ac_op = (split_ind, 'comb_add_after', tuple([cm2a] + bz.expand(ex)))
            kp_child, kce, kcc = find_best_edits(ex, cnode, by, INDEX)
                        
        elif sm == 'before':            
            ac_op = (split_ind, 'comb_add_before', tuple([cm2a] + by.expand(ex)))  
            kp_child, kce, kcc = find_best_edits(ex, cnode, bz, INDEX)
            
        else:
            assert False
            
        le, lc = find_local_best_edits(pnode, bl, INDEX)
        
        al.reset_from_stash()
        pnode.reset_from_stash()
        
        comb_edits.append((ac_op, ps_edits))
        comb_cost += 1 + len(ac_op[2])
        
        for eop, econd in kce + le:
            if econd is not None:
                ncond = econd + [ac_op]
            else:
                ncond = [ac_op]

                
            comb_edits.append((eop, ncond))

        comb_cost += kcc
        comb_cost += lc
            
        pnode.OI['CSI'][1] = kp_child

        return pnode, comb_edits, comb_cost
        
    else:
        assert False, 'one should have CSI'

def find_hier_best_edits(ex, al, bl, INDEX):
    
    if 'CSI' in al.OI or 'CSI' in bl.OI:
        return find_comb_best_edits(ex, al, bl, INDEX)

    # assume if there was combinator we handled it above
    
    if 'TSI' in al.OI and 'TSI' in bl.OI:        
        te, tc = find_tsi_edit(al, al.OI['TSI'], bl.OI['TSI'], INDEX)
        
        ne, nc = find_local_best_edits(al, bl, INDEX)
                
        tsi_child, le, lc = find_best_edits(ex, al.OI['TSI'][-1], bl.OI['TSI'][-1], INDEX)
        
        al.OI['TSI'][-1] = tsi_child
        
        comb_edits = te + ne + le
        comb_cost = tc + nc + lc
        
        return al, comb_edits, comb_cost
        
    elif 'TSI' in al.OI:

        comb_edits = []
        comb_cost = 1
        
        tsi_rm_edit = (al.fn2i['TSI'], 'trans_rm', None)

        comb_edits.append((tsi_rm_edit, None))

        child = al.OI['TSI'][-1]
        
        al.stash()
        child.stash()

        # Take all info from al and move it into child
        merge_into_child(ex, al, child, "TSI")

        assert len(al.OI) == 1
        assert 'TSI' in al.OI
        
        _, le, lc = find_best_edits(ex, child, bl, INDEX)
        
        al.reset_from_stash()
        child.reset_from_stash()        
        
        for eop, econd in le:
            if econd is not None:
                ncond = econd + [tsi_rm_edit]
            else:
                ncond = [tsi_rm_edit]
            comb_edits.append((eop, ncond))

        comb_cost += lc
        
        return al, comb_edits, comb_cost
        
    elif 'TSI' in bl.OI:
        
        bchild = bl.OI['TSI'][-1]
        pchild, le, lc = find_best_edits(ex, al, bchild, INDEX)

        pl = NormalForm(None, [], INDEX)
        pl.parent = pchild.parent
        rectify_parent(pl, pchild)
        pchild.parent = pl
        
        pl.OI['TSI'] = ['dummy', pchild]
        pl.fn2i['TSI'] = len(INDEX['token'])
        INDEX['token'][len(INDEX['token'])] = (pl, None)
        
        tsi_add_edit = (pl.fn2i['TSI'], 'trans_add', tuple(bl.OI['TSI'][:-1]))
        tc = 1 + len(tsi_add_edit[2])
        ne, nc = find_local_best_edits(pl, bl, INDEX)
        
        comb_edits = le
        comb_edits += [(tsi_add_edit, None)]

        for eop, econd in ne:
            assert econd is None
            comb_edits.append((eop, [tsi_add_edit]))
                
        comb_cost = tc + nc + lc 
        
        return pl, comb_edits, comb_cost
        
    else:
        assert False

def find_best_edits(ex, al, bl, INDEX):
    
    atype = None
    btype = None

    for tt in ('prim', 'CSI', 'TSI'):
        if tt in al.OI:
            assert atype is None
            atype = tt
        if tt in bl.OI:
            assert btype is None
            btype = tt

    assert atype is not None and btype is not None
            
    if atype == 'prim' and btype == 'prim':
        e, c = find_local_best_edits(al, bl, INDEX)
        return al, e, c
    else:
        rn, e, c = find_hier_best_edits(ex, al, bl, INDEX)
        return rn, e, c

def find_local_best_edits(al, bl, INDEX):
    
    cost = 0
    edits = []
            
    amiss = set()
    bmiss = set([bfn for bfn in bl.OI if bfn not in ['TSI', 'CSI']])

    for afn, aprms in al.OI.items():
        if afn in ['TSI', 'CSI']:
            continue
        
        if afn in bmiss:
            bmiss.remove(afn)
            bprms = bl.OI[afn]
            if aprms == bprms:
                # match prms
                continue
            else:
                # mod prm
                e, c = make_mod_prm_edit(al, afn, bprms)
                edits.append((e, None))
                cost += c            
        else:
            amiss.add(afn)
            
    while len(amiss) > 0 and len(bmiss) > 0:
        afn = amiss.pop()
        bfn = bmiss.pop()
        bprms = bl.OI[bfn]

        e, c = make_mod_fn_edit(al, afn, bfn, bprms)
        edits.append((e, None))
        cost += c
    
    for afn in amiss:
        e,c = make_rm_fn_edit(al, afn)
        edits.append((e, None))
        cost += c
        
    for bfn in bmiss:
        bprms = bl.OI[bfn]
        e,c = make_add_fn_edit(al, bfn, bprms)
        edits.append((e, None))
        cost += c

    return edits, cost



def find_best_pairing(ex, POS_AL, NEG_AL, POS_BL, NEG_BL, INDEX):
    
    def_POS_AL, def_POS_BL, pos_cost = opt_pairing(ex, POS_AL, POS_BL, INDEX)
    def_NEG_AL, def_NEG_BL, neg_cost = opt_pairing(ex, NEG_AL, NEG_BL, INDEX)

    def_cost = pos_cost + neg_cost
    
    return def_POS_AL, def_NEG_AL, def_POS_BL, def_NEG_BL, def_cost


def opt_pairing(ex, AL, BL, INDEX):
    ML = max(len(AL), len(BL))
    D = np.zeros((ML, ML)) - 1

    E_AL = [AL[i] if i < len(AL) else None for i in range(ML)]
    E_BL = [BL[i] if i < len(BL) else None for i in range(ML)]
    
    for i in range(ML):
        for j in range(ML):
            ai = E_AL[i]
            bj = E_BL[j]
                
            lcost = get_edit_cost(ex, ai, bj, INDEX)

            D[i,j] = lcost

    assignment = sopt.linear_sum_assignment(D)
                
    O_AL = []
    O_BL = []
    O_cost = 0.
    
    for i,j in zip(assignment[0], assignment[1]):
        O_AL.append(E_AL[i])
        O_BL.append(E_BL[j])
        O_cost += D[i,j]
    
    return O_AL, O_BL, O_cost
        

def get_edit_ops(ex, POS_AL, NEG_AL, POS_BL, NEG_BL, INDEX):
    edits = []
    cost = 0    

    POS_SL = []
    NEG_SL = []    

    for STOKEN, AL, BL, SL in [
        ('POS', POS_AL, POS_BL, POS_SL),
        ('NEG', NEG_AL, NEG_BL, NEG_SL)    
    ]:
        assert len(AL) == len(BL)
        
        for al, bl in zip(AL, BL):        
            sl = None
            
            if al is None:
                qe = tuple([STOKEN] + bl.expand(ex))
                local_edits = [(('first_node', 'comb_add_after', qe), None)]
                local_cost = len(qe) + 1
            
            elif bl is None:
                local_edits = [((al, 'comb_rm', None), None)]
                local_cost = 1
                sl = al
            else:
                sl, local_edits, local_cost = find_best_edits(ex, al, bl, INDEX)
                
                assert sl is not None
                                                                                
            edits += local_edits
            cost += local_cost
            if sl is not None:
                SL.append(sl)

    prog_NF = ProgNormalForm(POS_SL, NEG_SL)            
    return prog_NF, cost, edits


def make_edit_op(ex, start_tokens, edit_info):
    raw_edit_prog = _make_edit_op(ex, start_tokens, edit_info)
    norm_prog = normalize_program(ex, raw_edit_prog)
    return norm_prog
    
def _make_edit_op(ex, start_tokens, edit_info):
    q = [(t, None) for t in start_tokens]

    eot, eol, eos = edit_info

    q[eol] = (q[eol][0], (eot, eos))

    tokens = []

    while len(q) > 0:
        t,ei = q.pop(0)

        if ei is None:
            tokens.append(t)
            continue

        et, es = ei
        
        if et == 'ACA':

            if t == ex.START_TOKEN:
                rest = []
                last_neg = None
                while len(q) > 0:
                    assert q[0][1] is None
                    qt = q.pop(0)[0]
                    if qt == 'NEG':
                        last_neg = len(rest)
                    rest.append(qt)

                assert rest[-1] == 'END'
                assert len(tokens) == 0

                if last_neg is None:
                    tokens = [t] + rest[:-1] + es + ['END']
                else:
                    assert rest[last_neg] == 'NEG'
                    tokens = [t] + rest[:last_neg] + es + rest[last_neg:]
                
            else:
                se = [t]
                ipc = ex.TLang.get_num_inp(t)
                while ipc > 0:
                    nt,nei = q.pop(0)
                    assert nei is None
                    se.append(nt)
                    ipc += ex.TLang.get_num_inp(nt) - 1 
                tokens += [es[0]] + se + es[1:]
                
        elif et == 'ACB':
            
            if tokens[-1] in CS_TOKENS and es[0] in CS_TOKENS and es[0] == tokens[-1]:                
                ct = tokens.pop(-1)
                se = [ct, t]            
            else:
                se = [t]            
            
            ipc = ex.TLang.get_num_inp(t)
            while ipc > 0:
                nt,nei = q.pop(0)
                assert nei is None
                se.append(nt)
                ipc += ex.TLang.get_num_inp(nt) - 1

            tokens += es + se
            
        elif et == 'CR':
            
            se = [t]
            ipc = ex.TLang.get_num_inp(t)
            while ipc > 0:
                nt,nei = q.pop(0)
                assert nei is None
                se.append(nt)
                ipc += ex.TLang.get_num_inp(nt) - 1

                
            if tokens[-1] in ('POS', 'NEG'):
                tokens.pop(-1)
                
            elif tokens[-1] in CS_TOKENS:
                assert q[0][0] != 'END'
                tokens.pop(-1)

            elif q[0][0] == 'END':

                scont = True
                for ct in CS_TOKENS:
                    if ct in tokens:
                        scont = False
                        break

                if scont:
                    continue

                luis = [(i, x) for i,x in enumerate(tokens)]

                c = 0

                while len(luis) > 0:
                    ti, tt = luis.pop(-1)
                    if tt == 'prim':
                        c -= 1
                        continue

                    if tt in CS_TOKENS:
                        c += 1
                        
                        if c==0:
                            lui = ti
                            break
                                                    
                assert tokens[lui] in CS_TOKENS                
                tokens.pop(lui)
                
            else:

                luis = [i for i,x in enumerate(tokens) if x in CS_TOKENS]

                assert len(luis) > 0
                
                lui = luis[-1]                
                lbi = tokens[lui:].count('prim')                

                if lbi == 1:                    
                    rm_ind = lui
                elif lbi == 2:
                    assert len(luis) > 1
                    rm_ind = luis[-2]
                else:
                    assert False, 'dont know how to handle this case'
                    
                assert tokens[rm_ind] in CS_TOKENS                
                tokens.pop(rm_ind)                
                

        elif et == 'TA':
            se = [t]
            ipc = ex.TLang.get_num_inp(t)
            while ipc > 0:
                nt,nei = q.pop(0)
                assert nei is None
                se.append(nt)
                ipc += ex.TLang.get_num_inp(nt) - 1

            fn = es.pop(0)
            tokens.append(fn)

            for inp in ex.TLang.get_inp_types(fn):
                if inp == ex.BASE_TYPE:
                    assert se is not None
                    tokens += se
                    se = None
                else:
                    tokens.append(es.pop(0))
                    
            assert se is None
            assert len(es) == 0
            
        elif et == 'TR':

            ofn = t
            for inp in ex.TLang.get_inp_types(ofn):
                if inp == ex.BASE_TYPE:

                    ifn = q.pop(0)[0]
                    tokens.append(ifn)
                    ipc = ex.TLang.get_num_inp(ifn)
                    
                    while ipc > 0:
                        nt,nei = q.pop(0)
                        assert nei is None
                        tokens.append(nt)
                        ipc += ex.TLang.get_num_inp(nt) - 1                            
                    

                else:
                    q.pop(0)

        elif et in ('TM', 'PM'):

            ofn = t
            se = []
            for inp in ex.TLang.get_inp_types(ofn):
                if inp == ex.BASE_TYPE:

                    ifn = q.pop(0)[0]
                    se.append(ifn)
                    ipc = ex.TLang.get_num_inp(ifn)
                    
                    while ipc > 0:
                        nt,nei = q.pop(0)
                        assert nei is None
                        se.append(nt)
                        ipc += ex.TLang.get_num_inp(nt) - 1                               

                else:
                    q.pop(0)

            if et == 'TM':
                fn = es.pop(0)
            elif et == 'PM':
                fn = ofn
                
            tokens.append(fn)

            for inp in ex.TLang.get_inp_types(fn):
                if inp == ex.BASE_TYPE:
                    assert se is not None
                    tokens += se
                    se = None
                else:
                    tokens.append(es.pop(0))
                    
            assert se is None or len(se) == 0
            assert len(es) == 0            

        elif et in ('CM'):
            assert t in CS_TOKENS
            assert len(es) == 1
            assert es[0] in CS_TOKENS
            tokens.append(es.pop(0))
            
        else:
            assert False

    return tokens


def make_NF_edit(ex, NL, edit_op, INDEX, ret_early=False):

    fn_ind, edit_type, edit_prms = edit_op

    if isinstance(fn_ind, NormalForm):
        edit_node = fn_ind
        edit_fn = None
    elif fn_ind == 'first_node':
        edit_node = None
        edit_fn = None
    else:
        edit_node, edit_fn = INDEX['token'][fn_ind] 

        if edit_node is None:
            pass        
        elif edit_fn is not None and edit_fn not in edit_node.fn2i:
            assert False, 'bad 1'
        elif edit_fn is not None and edit_node.fn2i[edit_fn] != fn_ind:
            assert False, 'bad 2'

    prog, edit_loc = NL.format_for_edit(ex, edit_node, edit_fn)

    if edit_prms is None:
        edit_prms = []
    
    edit_info = (OKM[edit_type][1:], edit_loc, list(edit_prms))
    
    if ret_early:
        return prog, edit_info
    
    def help_add_new_fn(n, f, p):
        n.OI[f] = p
        n.fn2i[f] = len(INDEX['token'])
        INDEX['token'][len(INDEX['token'])] = (n, f)
        
    if edit_fn == 'TSI' and 'comb_add' not in edit_type:

        child = edit_node.OI['TSI'][-1]
        
        if edit_type == 'param_mod':
            edit_node.OI['TSI'] = tuple([edit_node.OI['TSI'][0]] + list(edit_prms) + [child])
        elif edit_type == 'trans_mod':
            edit_node.OI['TSI'] = tuple(list(edit_prms) + [child])
        elif edit_type == 'trans_rm':            
            merge_into_child(ex, edit_node, child, 'TSI', INDEX)
        else:            
            assert False, f'unexpected {edit_type}'
            
    # FOR LOCAL TRANSFORM OPS
    elif edit_type == 'param_mod':        
        edit_node.OI.pop(edit_fn)
        edit_node.OI[edit_fn] = [edit_prms]
            
    elif edit_type == 'trans_mod':
        edit_node.OI.pop(edit_fn)
        help_add_new_fn(edit_node, edit_prms[0], [edit_prms[1:]])
        
    elif edit_type == 'trans_rm':
        edit_node.OI.pop(edit_fn)
    
    elif edit_type == 'trans_add':
        assert edit_fn not in edit_node.OI
        if edit_prms[0] in TS_TOKENS:
                        
            assert len(edit_node.OI) == 1
            assert edit_node.OI['TSI'][0] == 'dummy'

            pchild = edit_node.OI['TSI'][1]
            
            edit_node.OI.pop('TSI')
            edit_node.OI['TSI'] = list(edit_prms) + [pchild]
            
        else:
            help_add_new_fn(edit_node, edit_prms[0], [edit_prms[1:]])

    elif edit_type == 'comb_mod':

        assert edit_fn == 'CSI'
        assert len(edit_prms) == 1

        assert edit_prms[0] in CS_TOKENS
        
        edit_node.OI[edit_fn][0] = edit_prms[0]        
            
    elif edit_type == 'comb_rm':

        pnode = edit_node.parent
        if pnode is not None:

            assert 'CSI' in pnode.OI
            assert len(pnode.OI['CSI']) == 3

            kp_child = None

            if pnode.OI['CSI'][1] == edit_node:
                kp_child = pnode.OI['CSI'][2]

            if pnode.OI['CSI'][2] == edit_node:
                kp_child = pnode.OI['CSI'][1]

            assert kp_child is not None

            merge_into_child(ex, pnode, kp_child, 'CSI', INDEX)
        else:            
            edit_node.deleted = True

    
    elif 'comb_add' in edit_type:

        if edit_node is None:            
            spt = edit_prms[0]
            
            ad = NormalForm(ex, list(edit_prms[1:]), INDEX)
            assert 'after' in edit_type  
            
            if spt == 'POS':
                NL.pos_sub_progs.append(ad)                    
            elif spt == 'NEG':
                NL.neg_sub_progs.append(ad)                
            else:
                assert False, f'bad spt {spt}'
            
        
        elif edit_node.parent is not None:
                        
            pnode = edit_node.parent
            orig_child = edit_node

            # find a dummy node
            while 'CSI' not in pnode.OI or \
                  pnode.OI['CSI'][0] != 'dummy':
                orig_child = pnode
                pnode = pnode.parent                

            # once we get to a dummy node, continue up to the highest dummy node
            while pnode.parent is not None and 'CSI' in pnode.parent.OI and pnode.parent.OI['CSI'][0] == 'dummy':
                orig_child = pnode
                pnode = pnode.parent
                
            try:
                assert len(pnode.OI) == 1
                assert pnode.OI['CSI'][0] == 'dummy'
                assert pnode.OI['CSI'][1] == orig_child
            except:
                raise Exception('double_add')
                
            ctkn = edit_prms[0]
            assert ctkn in CS_TOKENS
            
            new_child = NormalForm(ex, list(edit_prms[1:]), INDEX)
            new_child.parent = pnode
            
            if 'before' in edit_type:
                pnode.OI['CSI'] = [ctkn, new_child, orig_child]
            elif 'after' in edit_type:
                pnode.OI['CSI'] = [ctkn, orig_child, new_child]

            for oop in OP_ORD:
                assert oop not in ['CSI']

                if oop == edit_fn:
                    break
                
                if oop in edit_node.OI:
                    pnode.OI[oop] = edit_node.OI.pop(oop)
                    pnode.fn2i[oop] = edit_node.fn2i.pop(oop)
                    INDEX['token'][
                        pnode.fn2i[oop]
                    ] = (pnode, oop)
                                
        else:
            assert False, 'bad parsing of add comb'                  
        
    else:
        assert False, f'need to impl {edit_type}'
        
    return prog, edit_info

def make_NF_copy(O_PNF):

    PNF = ProgNormalForm([], [])
    o2n = {}
    
    for O in O_PNF.pos_sub_progs:
        N = O.make_copy(o2n)
        PNF.pos_sub_progs.append(N)

    for O in O_PNF.neg_sub_progs:
        N = O.make_copy(o2n)
        PNF.neg_sub_progs.append(N)        
        
    return PNF, o2n


def super_copy_info(O_PNF, O_EI, O_PE, O_NE):

    N_PNF, o2n = make_NF_copy(O_PNF)

    N_EI = {
        'node': {},
        'token': {},
    }

    for k,v in O_EI['node'].items():
        N_EI['node'][k] = o2n[v]

    for k,v in O_EI['token'].items():
        assert len(v) == 2
        N_EI['token'][k] = (o2n[v[0]], v[1])

    N_PE = []
    for eop in O_PE:
        if isinstance(eop[0], NormalForm):
            N_PE.append(tuple(
                [o2n[eop[0]]] + [e for e in eop[1:]]
            ))
        else:
            N_PE.append(tuple(list(eop)))


    N_NE = []

    for one in O_NE:
        if isinstance(one[0], NormalForm):
            nne = tuple(
                [o2n[one[0]]] + [e for e in one[1:]]
            )
        else:
            nne = tuple(list(one))
        N_NE.append(nne)
                
    return N_PNF, N_EI, N_PE, N_NE

def AP_super_edit_ops(all_edit_ops):

    valid_order = []

    seen_edits = set()
    while len(all_edit_ops) > 0:
        edit_op, edit_prereqs = all_edit_ops.pop(0)
        should_skip = False
        if edit_prereqs is not None:
            for epr in edit_prereqs:
                if epr not in seen_edits:
                    all_edit_ops.append((edit_op, edit_prereqs))
                    should_skip = True
                    break
        if not should_skip:
            seen_edits.add(edit_op)
            valid_order.append((edit_op, edit_prereqs))
                
    eops = list(range(len(valid_order)))
    
    all_pairs = []
    
    for i in range(0, len(eops)):

        if len(eops) > MAX_EOP_NUM:
            sub_eops = random.sample(eops, MAX_EOP_NUM)
            all_combs = list(itertools.combinations(sub_eops, i))
        else:
            all_combs = list(itertools.combinations(eops, i))

        if len(all_combs) < MAX_AP_SAMPLE_NUM:
            sampled_combs = all_combs
        else:
            sampled_combs = random.sample(all_combs, MAX_AP_SAMPLE_NUM)
        
        for ecomb in sampled_combs:
            prev_eops = []
            prev_req = set()
                
            for e in ecomb:
                prev_eops.append(valid_order[e][0])
                if valid_order[e][1] is not None:
                    for prq in valid_order[e][1]:
                        prev_req.add(prq)

            sprev_eops = set(prev_eops)

            targets = [e for e in eops if e not in ecomb]
                                
            if len(prev_req - sprev_eops) > 0:
                continue

            info = {
                'prev_eops': prev_eops,
                'new_eops': []
            }
            
            for t in targets:
                new_eop, new_preq = valid_order[t]

                if new_preq is not None and (
                        len(set(new_preq) - sprev_eops) > 0
                ):
                    continue

                info['new_eops'].append(new_eop)

            all_pairs.append(info)

    return all_pairs

def AP_conv_to_data_super(ex, start_prog, end_prog, PNF, EI, edit_ops):

    all_edit_ops = []
    for eop, epr in edit_ops:
        all_edit_ops.append((tuple(eop), epr if epr is None else tuple(epr)))

    AP_edit_info = AP_super_edit_ops(all_edit_ops)
    
    ret = []    
    
    for AP_EI in AP_edit_info:

        prev_edits, new_edits = AP_EI['prev_eops'], AP_EI['new_eops']
        
        super_PNF, super_EI, super_prev_edits, super_new_edits = super_copy_info(
            PNF, EI, prev_edits, new_edits
        )

        for super_pe in super_prev_edits:
            make_NF_edit(ex, super_PNF, super_pe, super_EI)
                    
        ocp = super_PNF.convert_to_prog(ex)
        
        api = {
            'corr_tokens': ocp,
            'eoi': [] 
        }
        for super_ne in super_new_edits:
            next_corr_prog, next_edit_info = make_NF_edit(
                ex, super_PNF, super_ne, super_EI, ret_early=True
            )

            if ocp != next_corr_prog:
                assert False
                
            api['eoi'].append(next_edit_info)

        ret.append(api)

    return ret

def calc_true_edit_cost(EO):
    cost = 0
    for eo in EO:
        cost += 1
        if eo[0][2] is not None:
            cost += len(eo[0][2])

    return cost

def get_edit_info(ex, corr_tokens, tar_tokens):
    
    corr_pos_sprogs, corr_neg_sprogs = split_to_sub_progs(corr_tokens)    
    tar_pos_sprogs, tar_neg_sprogs = split_to_sub_progs(tar_tokens)

    INDEX = {
        'token': {},
        'node': {},
        'cost': {}
    }
    DINDEX = {
        'token': {},
        'node': {}
    }
    
    UP_POS_AL = format_for_edit(ex, corr_pos_sprogs, INDEX)
    UP_NEG_AL = format_for_edit(ex, corr_neg_sprogs, INDEX)
    
    UP_POS_BL = format_for_edit(ex, tar_pos_sprogs, DINDEX)
    UP_NEG_BL = format_for_edit(ex, tar_neg_sprogs, DINDEX)
    
    POS_AL, NEG_AL, POS_BL, NEG_BL, pair_cost = find_best_pairing(
        ex, UP_POS_AL, UP_NEG_AL, UP_POS_BL, UP_NEG_BL, INDEX,
    )
           
    prog_NF, _, edit_ops = get_edit_ops(ex, POS_AL, NEG_AL, POS_BL, NEG_BL, INDEX)
                
    return (prog_NF, INDEX, edit_ops)





####### VERIFICATION ########

def verify(
    ex,
    start_prog,
    end_prog,
    NL, INDEX, edit_set
):
        
    start_img = ex.execute(' '.join(start_prog))
    end_img = ex.execute(' '.join(end_prog))

    corr_prog = NL.convert_to_prog(ex)
    corr_img = ex.execute(' '.join(corr_prog))

    edit_imgs = [start_img]
    
    # check start state    
    assert (start_img == corr_img).all()

    q = []
    for eop, epr in edit_set:
        q.append((tuple(eop), epr if epr is None else tuple(epr)))

    seen_edits = set()

    assert len(q) > 0
    
    while len(q) > 0:

        edit_op, edit_prereqs = q.pop(0)
                
        should_skip = False
        if edit_prereqs is not None:
            for epr in edit_prereqs:
                if epr not in seen_edits:
                    q.append((edit_op, edit_prereqs))
                    should_skip = True
                    break
                
        if should_skip:
            continue
                
        seen_edits.add(edit_op)
        
        next_corr_prog, next_edit_info = make_NF_edit(ex, NL, edit_op, INDEX)
        
        # check start == prev        
        assert (ex.execute(' '.join(next_corr_prog)) == \
            ex.execute(' '.join(corr_prog))).all()

        edit_prog = make_edit_op(ex, next_corr_prog, deepcopy(next_edit_info))        

        nf_edit_prog = NL.convert_to_prog(ex)

        edit_img = ex.execute(' '.join(edit_prog))        

        nf_edit_img = ex.execute(' '.join(nf_edit_prog))
        
        if not (nf_edit_img == edit_img).all():            
            assert False, 'something went wrong'
        
        corr_prog = edit_prog

        edit_imgs.append(edit_img)

    edit_imgs.append(end_img)
        
    if not (edit_img == end_img).all():
        assert False, 'something went wrong'
                
    return edit_imgs


class CSGProgDiff:
    def __init__(self, ex):
        self.ex = ex
        self.set_gparams()
        
    def set_gparams(self, A = 10, B = 20):
        global MAX_AP_SAMPLE_NUM
        global MAX_EOP_NUM
        MAX_AP_SAMPLE_NUM = A
        MAX_EOP_NUM = B
        
    def normalize_program(self, tokens):
        return normalize_program(self.ex, tokens)

    def make_edit_op(self, start_tokens, edit_info):
        return make_edit_op(self.ex, start_tokens, edit_info)

    def conv_to_data(self, dcm, start_prog, tar_prog, PNF, EI, edit_ops):

        if dcm == 'super_ap':                    
            return AP_conv_to_data_super(
                self.ex, start_prog, tar_prog, PNF, EI, edit_ops
            )
        
        else:
            assert False

    def get_edit_info(self, corr_tokens, tar_tokens):
        return get_edit_info(self.ex, corr_tokens, tar_tokens)
    
    def calc_true_edit_cost(self, EO):
        return calc_true_edit_cost(EO)

    def verify(self, start_prog, end_prog, PNF, EI, edit_ops):
        return verify(self.ex, start_prog, end_prog, PNF, EI, edit_ops)
    
