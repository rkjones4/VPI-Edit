import random
import torch
import numpy as np
MAX_TRIES = 10

colors = [
    (31, 119, 180),
    (174, 199, 232),
    (255,127,14),
    (255, 187, 120),
    (44,160,44),
    (152,223,138),
    (214,39,40),
    (255,152,150),
    (148, 103, 189),
    (192,176,213),
    (140,86,75),
    (196,156,148),
    (227,119,194),
    (247,182,210),
    (127,127,127),
    (199,199,199),
    (188,188,34),
    (219,219,141),
    (23,190,207),
    (158,218,229)
]
CMAP = {
    i:(torch.tensor(colors[i]) / 256.)
    for i in range(len(colors))
}

def norm_np(L):
    return np.array(L) / np.array(L).sum()

def _wrap(txt, B=150):
    s = []
    while len(txt) > 0:
        s += txt[:B]
        txt = txt[B:]
        s += '\n'
    return ''.join(s)

def norm_sample(vals, mean, std, mi, ma):
    v = None

    if mi == ma:
        return mi
    
    for _ in range(MAX_TRIES):
        v = mean + (np.random.randn() * std)
        if v >= mi and v <= ma:
            val = round_val(v, vals)
            if val >= mi and v <= ma:
                return val

    return mi


def round_val(sval, tvals):
    bv = None
    be = 1e8
    for t in tvals:
        err = abs(t-sval)
        if err < be:
            bv = t
            be = err

    return bv


def make_flt_map(
    spec_vals,
    min_val,
    max_val,
    num_tokens,
    fprec=2
):

    m = []
    sl = list(spec_vals)

    count = 0
    
    for sv in sl:
        count += 1
        m.append(round(sv, fprec))

    num_tokens -= count

    diff = (max_val - min_val) 
    
    for _i in range(1, num_tokens+1):
        val = min_val + ((_i / (num_tokens+1.)) * diff)
        rval = round(val,fprec)
        if rval not in m:        
            m.append(rval)

    m.sort()    
    return m

def sample_dist(X):
    A, B = X
    return np.random.choice(A,p=B)
