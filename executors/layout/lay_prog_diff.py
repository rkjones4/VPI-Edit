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
    'trans_add': '$TA',
    'trans_rm': '$TR',
    'trans_mod': '$TM',
    'param_mod': '$PM',
}

CS_TOKENS = set(['union'])
TS_TOKENS = set(['symReflect', 'symTranslate', 'symRotate'])
OP_ORD = ['color', 'move', 'scale', 'prim', 'TSI', 'CSI']

def split_by_union(tokens, ex):
    
    if tokens[0] == 'START':
        assert tokens[-1] == 'END'
        if len(tokens) == 2:
            return []
        return split_by_union(tokens[1:-1], ex)

    if tokens[0] == 'union':
        cur = []
        rest = [t for t in tokens[1:]]
        ipc = 1
        while len(rest) > 0:
            n = rest.pop(0)
            ipc -= 1
            ipc += ex.TLang.get_num_inp(n)

            cur.append(n)
            
            if ipc == 0:
                break
            
        assert ipc == 0
        
        return [cur] + split_by_union(rest, ex)

    else:
        return [tokens]
    
def calc_edit_cost(a, b, INDEX):
    
    assert not (a is None and b is None)

    if a is None:
        bexp = b.expand()        
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
        return calc_csi_edit_cost(a, b, INDEX)
    
    else:
        return calc_tsi_edit_cost(a, b, INDEX)

def calc_csi_edit_cost(a, b, INDEX):

    if 'CSI' in a.OI and 'CSI' in b.OI:
        cost = 0
        cost += get_edit_cost(a.OI['CSI'][1], b.OI['CSI'][1], INDEX)
        cost += get_edit_cost(a.OI['CSI'][2], b.OI['CSI'][2], INDEX)
        cost += calc_local_edit_cost(a, b)
        return cost
        
    elif 'CSI' in a.OI:

        c1 = a.OI['CSI'][1]
        c2 = a.OI['CSI'][2]

        a.stash()
        c1.stash()
        c2.stash()

        merge_into_child(a, c1, 'CSI')
        
        cost_rm_2 = get_edit_cost(c1, b, INDEX)

        a.reset_from_stash()
        a.stash()
        
        merge_into_child(a, c2, 'CSI')

        cost_rm_1 = get_edit_cost(c2, b, INDEX)

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
                
        by_ec = get_edit_cost(cnode, by, INDEX) + get_edit_cost(None, bz, INDEX)
        bz_ec = get_edit_cost(cnode, bz, INDEX) + get_edit_cost(None, by, INDEX)

        byexp = by.expand()
        bzexp = bz.expand()

        if bzexp[0] == 'union':
            cost += by_ec
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
    

def calc_tsi_edit_cost(a, b, INDEX):

    if 'TSI' in a.OI and 'TSI' in b.OI:

        cost = 0
        cost += calc_local_edit_cost(a, b)
        cost += get_edit_cost(a.OI['TSI'][-1], b.OI['TSI'][-1], INDEX)

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
        merge_into_child(a, child, "TSI")
        
        cost = get_edit_cost(child, b, INDEX)

        a.reset_from_stash()
        child.reset_from_stash()        
        
        cost += 1
        
        return cost
        

    elif 'TSI' in b.OI:

        btsi = b.OI['TSI'][:-1]

        cost = get_edit_cost(a, b.OI['TSI'][-1], INDEX)

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
        
    
def get_edit_cost(a, b, INDEX):
    
    if a is None:
        aid = None
    else:    
        aid = a.node_id

    if b is None:
        bid = None
    else:
        bid = b.node_id

    if (aid, bid) not in INDEX['cost']:
        cost = calc_edit_cost(a, b, INDEX)    
        INDEX['cost'][(aid, bid)] = cost
        
    return INDEX['cost'][(aid, bid)]

    
        
    
def normalize_op(op, info):
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

        if op == 'move':        
            new = tuple([sum([float(i[0]) for i in info]), sum([float(i[1]) for i in info])])
            news = [str(round(new[0],3)), str(round(new[1],3))]
            return [op] + news

        if op == 'scale':
            new = tuple([math.prod([float(i[0]) for i in info]), math.prod([float(i[1]) for i in info])])
            news = [str(round(new[0],3)), str(round(new[1],3))]
            return [op] + news

        if op == 'color':
            rv= [op] + list(info[0])
            return rv

        assert False, f'unexpected op {op}'
        
    return [op] + list(info[0])

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
            
    def expand(self, e = None):
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

            for t in normalize_op(oop, self.OI[oop]):
                tokens.append(t)
                            
        for t in tokens:
            if isinstance(t, NormalForm):                
                t.expand(e)
            else:
                e['tokens'].append(t)

        return e['tokens']

    def edit_expand(self, edit_node, edit_fn, e = None):

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

            for i,t in enumerate(normalize_op(oop, self.OI[oop])):
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
                t.edit_expand(edit_node, edit_fn, e)
            else:
                e['tokens'].append(t)

        return e['tokens']

def merge_values(fn, tvl, bvl):
    assert len(tvl) == 1
    assert len(bvl) == 1
    
    if fn == 'scale':
        return [[
            str(float(tvl[0][0]) * float(bvl[0][0])),
            str(float(tvl[0][1]) * float(bvl[0][1])),
        ]]
    elif fn == 'move':
        return [[
            str(float(tvl[0][0]) + float(bvl[0][0])),
            str(float(tvl[0][1]) + float(bvl[0][1])),
        ]]
    elif fn == 'color':
        return tvl

    else:
        assert False, f'unexpected fn {fn}'

def split_to_parent(al, bl, INDEX, no_op=False):

    split_op = None
    has_above = False
    padd = []

    al.node_id = f'split_bot_{al.node_id}'
    
    for oop in OP_ORD:

        if split_op is not None:
            break
        
        assert oop not in ['CSI']

        if oop in al.OI and oop in bl.OI:
            has_above = True

        elif oop in bl.OI:
            if not has_above and len(padd) == 0:
                padd.append(oop)
                has_above = True
                
        elif oop in al.OI:
            if has_above:
                split_op = oop
            else:
                has_above = True

        else:
            continue

    assert has_above
    assert split_op is not None
        
    ps_edits = []
    ps_cost = 0

    split_fn = split_op
    sind = al.fn2i[split_fn]
    
    if len(padd) > 0: 
        
        assert len(padd) == 1

        add_fn = padd.pop(0)
        ps_ei = (al, 'trans_add', tuple([add_fn] + list(bl.OI[add_fn][0])))
        ps_cost += 1 + len(ps_ei[2])
        ps_edits.append(ps_ei)

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
                                                                
def merge_into_child(P, C, key, mapping=None):
    
    C.node_id = f'{P.node_id}m{C.node_id}'
    
    P.OI.pop(key)
    
    for k,v in P.OI.items():
        if k not in C.OI:
            C.OI[k] = v
            C.fn2i[k] = P.fn2i[k]
        else:
            cv = C.OI[k]
            mv = merge_values(k, v, cv)
            C.OI[k] = mv
            
        if mapping is not None:
            mapping['token'][P.fn2i[k]] = (C, k)
                
    P.OI = {key: ['dummy', C]}
    P.fn2i = {key: P.fn2i[key]}

    if 'TSI' in C.OI and C.OI['TSI'][0] == 'dummy':
        merge_into_child(C, C.OI['TSI'][1], 'TSI', mapping)

    elif 'CSI' in C.OI and C.OI['CSI'][0] == 'dummy':
        merge_into_child(C, C.OI['CSI'][1], 'CSI', mapping)
                
def format_for_edit(ex, spl, m=None):

    nfl = []
    
    for sp in spl:
                
        nf = NormalForm(ex, deepcopy(sp), m)
        nfl.append(nf)
        
    return nfl

def normalize_program(ex, tokens, debug=False):
    sub_progs = split_by_union(tokens, ex)

    NFL = format_for_edit(ex, sub_progs)

    norm_prog = conv_NFL_to_prog(ex, NFL)

    if debug:
        o_img = ex.execute(' '.join(tokens))
        n_img = ex.execute(' '.join(norm_prog))

        assert (o_img == n_img).all()        
    
    return norm_prog
    
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

def find_comb_best_edits(al, bl, INDEX):

    if 'CSI' in al.OI and 'CSI' in bl.OI:

        assert len(al.OI['CSI']) == 3
        assert len(bl.OI['CSI']) == 3
        
        ne, nc = find_local_best_edits(al, bl, INDEX)

        c1, l1e, l1c = find_best_edits(al.OI['CSI'][1], bl.OI['CSI'][1], INDEX)
        c2, l2e, l2c = find_best_edits(al.OI['CSI'][2], bl.OI['CSI'][2], INDEX)
        
        al.OI['CSI'][1] = c1
        al.OI['CSI'][2] = c2

        comb_edits = ne + l1e + l2e
        comb_cost = nc + l1c + l2c
        
        return al, comb_edits, comb_cost
        
    elif 'CSI' in al.OI:
        
        assert len(al.OI['CSI']) == 3

        c1 = al.OI['CSI'][1]
        c2 = al.OI['CSI'][2]

        al.stash()
        c1.stash()
        c2.stash()

        merge_into_child(al, c1, 'CSI')
        
        cost_rm_2 = get_edit_cost(c1, bl, INDEX)

        al.reset_from_stash()
        al.stash()
        
        merge_into_child(al, c2, 'CSI')

        cost_rm_1 = get_edit_cost(c2, bl, INDEX)

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
        
        kp_child, kce, kcc = find_best_edits(kp_child, bl, INDEX)
        
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
                
        by_ec = get_edit_cost(cnode, by, INDEX) + get_edit_cost(None, bz, INDEX)
        bz_ec = get_edit_cost(cnode, bz, INDEX) + get_edit_cost(None, by, INDEX)

        byexp = by.expand()
        bzexp = bz.expand()

        comb_edits = [(p, None) for p in ps_edits]
        comb_cost = ps_cost

        if bzexp[0] == 'union':
            sm = 'after'
        
        elif by_ec <= bz_ec:
            sm = 'after'
        else:
            sm = 'before'            
        
        if sm == 'after':

            ac_op = (split_ind, 'comb_add_after', tuple(['union'] + bz.expand()))
            kp_child, kce, kcc = find_best_edits(cnode, by, INDEX)
                        
        elif sm == 'before':
            
            ac_op = (split_ind, 'comb_add_before', tuple(['union'] + by.expand()))        
            kp_child, kce, kcc = find_best_edits(cnode, bz, INDEX)
            
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

def find_hier_best_edits(al, bl, INDEX):
    
    if 'CSI' in al.OI or 'CSI' in bl.OI:
        return find_comb_best_edits(al, bl, INDEX)
                
    # assume if there was combinator we handled it above
    
    if 'TSI' in al.OI and 'TSI' in bl.OI:        
        te, tc = find_tsi_edit(al, al.OI['TSI'], bl.OI['TSI'], INDEX)
        
        ne, nc = find_local_best_edits(al, bl, INDEX)
                
        tsi_child, le, lc = find_best_edits(al.OI['TSI'][-1], bl.OI['TSI'][-1], INDEX)
        
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
        merge_into_child(al, child, "TSI")

        assert len(al.OI) == 1
        assert 'TSI' in al.OI
        
        _, le, lc = find_best_edits(child, bl, INDEX)
        
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

        pchild, le, lc = find_best_edits(al, bchild, INDEX)

        pl = NormalForm(None, [], INDEX)
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

def find_best_edits(al, bl, INDEX):

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
        rn, e, c = find_hier_best_edits(al, bl, INDEX)
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

def find_naive_pairing(AL, BL, INDEX):
        
    pairing = []

    while len(AL) > 0 or len(BL) > 0:
        al = None
        bl = None
        if len(AL) > 0:
            al = AL.pop(0)

        if len(BL) > 0:
            bl = BL.pop(0)

        pair_cost = get_edit_cost(al, bl, INDEX)
            
        pairing.append((al, bl, pair_cost))

        
    BP_AL = [i[0] for i in pairing]
    BP_BL = [i[1] for i in pairing]
    BP_COST = sum([i[2] for i in pairing])
    
    return BP_AL, BP_BL, BP_COST


def find_best_pairing(AL, BL, INDEX, use_naive):

    def_AL, def_BL, def_cost = find_naive_pairing(
        [a for a in AL],
        [b for b in BL],
        INDEX        
    )

    if use_naive:
        best_cost = None
    else:    
        best_AL, best_BL, best_cost = dp_fbp(AL, BL, INDEX, def_cost, [], [], 0)

    if best_cost is not None:
        return best_AL, best_BL, best_cost
    else:
        return def_AL, def_BL, def_cost

def dp_fbp(AL, BL, INDEX, cost_max, cA, cB, ccost):

    if ccost >= cost_max:
        return None, None, None
    
    if len(AL) == 0 and len(BL) == 0:
        return cA, cB, ccost

    best_cost = 10000000
    best_opt = None
    
    if len(AL) > 0 and len(BL) > 0:
        o1_pc = get_edit_cost(AL[0], BL[0], INDEX)
        o1_A, o1_B, o1_cost = dp_fbp(AL[1:], BL[1:], INDEX, cost_max, cA + [AL[0]], cB + [BL[0]], ccost + o1_pc)

        if o1_cost is not None and o1_cost < best_cost:
            best_cost = o1_cost
            best_opt = (o1_A, o1_B, o1_cost)
        
    if len(BL) > 0:        
        o2_pc = get_edit_cost(None, BL[0], INDEX)
        o2_A, o2_B, o2_cost = dp_fbp(AL, BL[1:], INDEX, cost_max, cA + [None], cB + [BL[0]], ccost + o2_pc)

        if o2_cost is not None and o2_cost < best_cost:
            best_cost = o2_cost
            best_opt = (o2_A, o2_B, o2_cost)
        
    if len(AL) > 0:
        o3_pc = get_edit_cost(AL[0], None, INDEX)
        o3_A, o3_B, o3_cost = dp_fbp(AL[1:], BL, INDEX, cost_max, cA + [AL[0]], cB + [None], ccost + o3_pc)

        if o3_cost is not None and o3_cost < best_cost:
            best_cost = o3_cost
            best_opt = (o3_A, o3_B, o3_cost)
    
    if best_opt is None:
        return None, None, None
    
    return best_opt
        

    

def get_edit_ops(AL, BL, INDEX):
    edits = []
    cost = 0    
    SL = []

    assert len(AL) == len(BL)

    q_edits = []
    
    for ind, (al, bl) in enumerate(zip(AL, BL)):        
        
        if al is None:
            q_edits.append(tuple(['union'] + bl.expand()))
            continue        
            
        elif bl is None:
            local_edits = [((al, 'comb_rm', None), None)]
            local_cost = 1
            sl = al
        else:
            sl, local_edits, local_cost = find_best_edits(al, bl, INDEX)

            assert sl is not None

            add_cond = []
            
            while len(q_edits) > 0:
                qe = q_edits.pop(0)
                new_add_edit = (sl, 'comb_add_before', qe)
                local_edits += [
                    (new_add_edit, [a for a in add_cond])
                ]
                local_cost += len(qe) + 1
                add_cond.append(new_add_edit)
                                                                    
        edits += local_edits
        cost += local_cost
        if sl is not None:
            SL.append(sl)            

    add_cond = []
    while len(q_edits) > 0:
        qe = q_edits.pop(0)
        new_add_edit = ('last_node', 'comb_add_after', qe)
        edits += [
            (new_add_edit, [a for a in add_cond])            
        ]
        cost += len(qe) + 1
        add_cond.append(new_add_edit)

    return SL, cost, edits


def make_edit_op(ex, start_tokens, edit_info):
    
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
            se = [t]
            ipc = ex.TLang.get_num_inp(t)
            while ipc > 0:
                nt,nei = q.pop(0)
                assert nei is None
                se.append(nt)
                ipc += ex.TLang.get_num_inp(nt) - 1 

            tokens += [es[0]] + se + es[1:]
                
        elif et == 'ACB':
            
            if tokens[-1] == 'union' and es[0] == 'union':                
                tokens.pop(-1)
                se = ['union', t]            
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
                
            if tokens[-1] == 'union':
                assert q[0][0] != 'END'
                tokens.pop(-1)

            elif q[0][0] == 'END':
                if 'union' not in tokens:
                    continue

                luis = [(i, x) for i,x in enumerate(tokens)]

                c = 0

                while len(luis) > 0:
                    ti, tt = luis.pop(-1)
                    if tt == 'prim':
                        c -= 1
                        continue

                    if tt == 'union':
                        c += 1
                        
                        if c==0:
                            lui = ti
                            break
                                                    
                assert tokens[lui] == 'union'
                
                tokens.pop(lui)
            else:
                assert 'union' in tokens
                luis = [i for i,x in enumerate(tokens) if x == 'union']

                lui = luis[-1]                
                lbi = tokens[lui:].count('prim')                
                assert lbi == 1

                assert tokens[lui] == 'union'
                
                tokens.pop(lui)                
                

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
                    
        else:
            assert False

    return tokens

def conv_NFL_to_prog(ex, NL):

    sub_progs = []

    for node in NL:
        tokens = node.expand()        
        sub_progs.append(tokens)

    sub_progs = [s for s in sub_progs if len(s) > 0]
        
    comb = ex.comb_sub_progs(sub_progs)

    return comb

def format_NFL_for_edit(ex, NL, edit_node, edit_fn):
    
    sub_progs = []
        
    for node in NL:
        tokens = node.edit_expand(edit_node, edit_fn)        
        sub_progs.append(tokens)

    sub_progs = [s for s in sub_progs if len(s) > 0]
    comb = ex.comb_sub_progs(sub_progs)

    
    assert ' '.join(comb).count(edit_ST) == 1

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
    
# This edits NL in place -> 
def make_NF_edit(ex, NL, edit_op, INDEX, ret_early=False):
    
    fn_ind, edit_type, edit_prms = edit_op

    if isinstance(fn_ind, NormalForm):
        edit_node = fn_ind
        edit_fn = None
    elif fn_ind == 'last_node':
        edit_node = NL[-1]
        edit_fn = None
    else:
        edit_node, edit_fn = INDEX['token'][fn_ind] 
        
        if edit_fn is not None and edit_fn not in edit_node.fn2i:
            assert False, 'something went wrong'

        if edit_fn is not None and edit_node.fn2i[edit_fn] != fn_ind:
            assert False, 'something went wrong'
        
    prog, edit_loc = format_NFL_for_edit(ex, NL, edit_node, edit_fn)

    if edit_prms is None:
        edit_prms = []
    
    edit_info = (OKM[edit_type][1:], edit_loc,  list(edit_prms))

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
            merge_into_child(edit_node, child, 'TSI', INDEX)
            
        else:            
            assert False, f'unexpected {edit_type}'
            
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
            
            merge_into_child(pnode, kp_child, 'CSI', INDEX)
        else:            
            edit_node.deleted = True
    
    elif 'comb_add' in edit_type:

        if edit_node.parent is not None:
            
            assert len(edit_node.parent.OI) == 1
            assert edit_node.parent.OI['CSI'][0] == 'dummy'
            assert edit_node.parent.OI['CSI'][1] == edit_node

            assert edit_prms[0] == 'union'

            new_child = NormalForm(ex, list(edit_prms[1:]), INDEX)
            new_child.parent = edit_node.parent
            
            if 'before' in edit_type:
                edit_node.parent.OI['CSI'] = ['union', new_child, edit_node]
            elif 'after' in edit_type:
                edit_node.parent.OI['CSI'] = ['union', edit_node, new_child]

            for oop in OP_ORD:
                assert oop not in ['CSI']

                if oop == edit_fn:
                    break
                
                if oop in edit_node.OI:
                    edit_node.parent.OI[oop] = edit_node.OI.pop(oop)
                    edit_node.parent.fn2i[oop] = edit_node.fn2i.pop(oop)
                    INDEX['token'][
                        edit_node.parent.fn2i[oop]
                    ] = (edit_node.parent, oop)
                    
            
        else:
        
            add_ind = None
            for ind, node in enumerate(NL):
                if node == edit_node:
                    add_ind = ind
                    break
                    
            assert add_ind is not None
            
            ad = NormalForm(ex, list(edit_prms[1:]), INDEX)

            if 'before' in edit_type:        
                NL.insert(add_ind, ad)
            else:
                NL.insert(add_ind+1,ad)
        
    else:
        assert False, f'need to impl {edit_type}'
        
    return prog, edit_info


    

def make_NF_copy(OL):
    o2n = {}
    NL = []

    for O in OL:
        N = O.make_copy(o2n)
        NL.append(N)
        
    return NL, o2n
        
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

    # Find one valid way of ordering the edits (w.r.t dependencies)
    
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

    # This is the set of possible edits
    eops = list(range(len(valid_order)))
    
    all_pairs = []

    # i is the number of previous edits
    for i in range(0, len(eops)):
               
        # for i number of previous edits, find all ways we could have applied i edits
        all_combs = list(itertools.combinations(eops, i))

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
            # ecomb is the set of edits we assume has been applied
            
            # we know need to know if ecomb violates any of the dependencies
            prev_eops = []
            # set of edits that should have been made
            prev_req = set()
                
            for e in ecomb:
                prev_eops.append(valid_order[e][0])
                if valid_order[e][1] is not None:
                    for prq in valid_order[e][1]:
                        prev_req.add(prq)

            # made ops at this point
            sprev_eops = set(prev_eops)

            # set of edits that could be targets at this point
            targets = [e for e in eops if e not in ecomb]

            # if the set of edits we have made hasn't covered the set of edits we need to make, don't use this starting point
            if len(prev_req - sprev_eops) > 0:
                continue

            # For this "state" (having made prev_ops) we want to find all next ops we should predict
            info = {
                'prev_eops': prev_eops,
                'new_eops': []
            }

            # Find the edits in target set that are valid
            for t in targets:
                new_eop, new_preq = valid_order[t]

                # if we haven't made all the edits required for t, don't include it as a valid target
                if new_preq is not None and (
                    len(set(new_preq) - sprev_eops) > 0
                ):
                    continue

                info['new_eops'].append(new_eop)

            # info will contribute one "training example"
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
                    
        ocp = conv_NFL_to_prog(ex, super_PNF)
        
        api = {
            'corr_tokens': ocp,
            'eoi': [],
        }
        
        for super_ne in super_new_edits:
            next_corr_prog, next_edit_info = make_NF_edit(
                ex, super_PNF, super_ne, super_EI, ret_early=True
            )

            if ocp != next_corr_prog:
                assert False, 'something went wrong'
                
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

def get_edit_info(ex, corr_tokens, tar_tokens, use_naive=False):
    
    corr_sub_progs = split_by_union(corr_tokens, ex)
    tar_sub_progs = split_by_union(tar_tokens, ex)

    INDEX = {
        'token': {},
        'node': {},
        'cost': {}
    }
    DINDEX = {
        'token': {},
        'node': {}
    }
    
    UP_AL = format_for_edit(ex, corr_sub_progs, INDEX)
    UP_BL = format_for_edit(ex, tar_sub_progs, DINDEX)
    
    AL, BL, pair_cost = find_best_pairing(UP_AL, UP_BL, INDEX, use_naive)
                
    prog_NF, _, edit_ops = get_edit_ops(AL, BL, INDEX)
    
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

    corr_prog = conv_NFL_to_prog(ex, NL)
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
        
        edit_prog = make_edit_op(ex, next_corr_prog, next_edit_info)        
        
        nf_edit_prog = conv_NFL_to_prog(ex, NL)
        
        edit_img = ex.execute(' '.join(edit_prog))        

        nf_edit_img = ex.execute(' '.join(nf_edit_prog))

        if not (nf_edit_img == edit_img).all():            
            assert False, 'something went wrong'

        edit_imgs.append(edit_img)
            
        corr_prog = edit_prog

    edit_imgs.append(end_img)
        
    return edit_imgs


class LayProgDiff:
    def __init__(self, ex):
        self.ex = ex
        self.set_gparams()

    def split_by_union(self, tokens):
        return split_by_union(tokens, self.ex)
        
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
    
