AT INFER_PROGS

#  Learning to Edit Visual Programs with Self-Supervision 
R. Kenny Jones, Renhao Zhang, Aditya Ganeshan, and Daniel Ritchie 

Neurips 2024

Arxiv: https://arxiv.org/pdf/2406.02383

# Inference

python3 main.py -en {EXP_NAME} -mm eval -dn {DOMAIN} -os_lmp {OS_NET} -lmp {EDIT_NET} -ets {EVAL_NUM} -inf_ps {INF_POP_SIZE} -inf_rs {INF_ROUNDS}

## Trained models

trained_models/pretrain
train_models/finetune

# Finetuning

python3 main.py -en {EXP_NAME} -mm finetune -dn {DOMAIN} -os_lmp {OS_NET} -lmp {EDIT_NET}

-ts {TS} -evs {EVS} -ws {WS} -inf_ps {INF_POP_SIZE} -inf_rs {INF_ROUNDS}

# Pretraining

## How to pretrain one-shot

```python3 main.py -en {EXP_NAME} -mm os_pretrain -dn {DOMAIN} -pm os```

## How to pretrain edit

```python3 main.py -en {EXP_NAME} -mm edit_pretrain -dn {DOMAIN} -pm edit -spcp {CACHE_PATH} -logp 100 -os_lmp {OS_NET}```

Explain about CACHE --> either need to make or use

# Exploring synthetic data generators

```python3 executors/common/test.py {DOMAIN} {DATA_TYPE} {NUM} {MODE}```

DOMAIN -> csg2d / csg3d / lay
DATA_TYPE -> os / edit
MODE - stats / vis
