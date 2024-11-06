# Learning to Edit Visual Programs with Self-Supervision

By [R. Kenny Jones](https://rkjones4.github.io/), [Renhao Zhang](https://renhaoz.github.io/), [Aditya Ganeshan](https://bardofcodes.github.io/), and [Daniel Ritchie](https://dritchie.github.io/)

![Teaser](https://rkjones4.github.io/img/vpi_edit/pp_vpi_edit_teaser.gif)
 
## About the paper

[Paper](https://rkjones4.github.io/pdf/vpi_edit.pdf)

[Project Page](https://rkjones4.github.io/vpi_edit.html)

Presented at [NeurIPS 2024](https://neurips.cc/).

## Bibtex
```
@inproceedings{jones2024VPIEdit,
  title= {Learning to Edit Visual Programs with Self-Supervision},
  author= {Jones, R. Kenny and Zhang, Renhao and Ganeshan, Aditya and Ritchie, Daniel},
  booktitle = {Advances in Neural Information Processing Systems},
  year= {2024}
}
```

# General Info

This repository contains code and data for the experiments in the above paper.

Each main.py call requires setting an experiment name and domain name (EXP_NAME and DOMAIN).

EXP_NAME can be any string (results will be saved to model_output/{EXP_NAME})

DOMAIN can be one of: layout / csg2d / csg3d (see main.py)

# Data and Pretrained Models

Google drive download links for:

[Trained Models](https://drive.google.com/file/d/1e8qH4-a4hkx914nrkXGRxyl-W9hX4Ipj/view?usp=sharing)

[Target Data](https://drive.google.com/file/d/1e8qH4-a4hkx914nrkXGRxyl-W9hX4Ipj/view?usp=sharing)

Please unzip each of these files from the root of this directory.

For each domain, we include the following trained models:
- trained_mode/finetune/{DOMAIN}_edit_ft.pt --> edit network finetuned on target data
- trained_mode/finetune/{DOMAIN}_os_inf_ft.pt --> os network for inference finetuned on target data
- trained_mode/finetune/{DOMAIN}_os_gen_ft.pt --> os network for wake-sleep generation finetuned on target data
  
- trained_mode/pretrain/{DOMAIN}_edit_pre.pt --> edit network pretrained on synthetic data
- trained_mode/pretrain/{DOMAIN}_os_pre.pt --> oneshot network pretrained on synthetic data

# Joint Inference 

To run our joint inference algorithm using both one-shot and edit networks to perform VPI tasks, you can use the following command:

```python3 main.py -en {EXP_NAME} -mm eval -dn {DOMAIN} -os_lmp {OS_NET} -lmp {EDIT_NET} -ets {EVAL_NUM}```

OS_NET should be a path to a trained one-shot network (e.g. trained_mode/finetune/{DOMAIN}_os_inf_ft.pt)

EDIT_NET should be a path to a trained edit network (e.g. trained_mode/finetune/{DOMAIN}_edit_ft.pt)

Results of this search procedure over EVAL_NUM number of examples will be save to model_output/EXP_NAME

*Notes*:

To control the population size and number of rounds in this search setting, you can use the following arguments:

``` -inf_ps {INF_POP_SIZE} -inf_rs {INF_ROUNDS} ```

When visualizing csg3d results, rendering voxel fields with matplotlib can take a prohibitively long time. You can turn off this feature by setting the (-nw / --num_write) argument to 0

# Finetuning 

To start a new joint finetuning run, use a command like:

```python3 main.py -en {EXP_NAME} -mm finetune -dn {DOMAIN} -os_lmp {OS_NET} -lmp {EDIT_NET}```

You can use the following arguments to vary the number of training/val target shapes and the number of wake-sleep generations used:

``` -ts {TS} -evs {EVS} -ws {WS} ```

# Pretraining

**One-shot network** 

To start a new pretraining run for a one-shot network, use a command like:

```python3 main.py -en {EXP_NAME} -mm os_pretrain -dn {DOMAIN} -pm os```

**Edit network**

To start a new pretraining run for an edit network, use a command like:

```python3 main.py -en {EXP_NAME} -mm edit_pretrain -dn {DOMAIN} -pm edit -spcp {CACHE_PATH} -logp 100 -os_lmp {OS_NET}```

*Notes*:

Edit network pretraining requires using a pretrained one-shot network to produce program samples when conditioned on randomly sampled synthetic scenes. These (start, end) program pairs will be saved to CACHE_PATH (e.g. some file name in cache/).

If the CACHE_PATH file exists, then the saved pairs are used. If this file does not exist, then creating this paired data can take a bit of time before the edit network begins training.  

Our experiments use 100,000 training program pairs and 100 validation program pairs for edit network pretraining, which can be recreated by setting: 

``` -ts 100000 -evs 100 ```

Once the cache is saved from the first run of the above command, please rerun the command to begin edit network training.

# Sampling synthetic scenes

To sample synthetic scenes from our visual programming domains, use a command like:

```python3 executors/common/test.py {DOMAIN} {PRED_MODE} {NUM} {MODE}```

PRED_MODE can be either os or edit

NUM is the number of synthetic scenes

MODE can be either stats or vis 

# Code Structure

**Folders**:

*data* --> target set data 

*domains* --> visual programming domains

*executors* --> execution logic for each domain. Also includes logic for findEdits algorithm.

*model_output* --> where experiment logic is saved

**Files**:

*edit_models.py* --> edit model architecture

*edit_pretrain.py* --> logic for edit network pretraining

*infer_progs.py* --> wrapper logic for running joint inference over sets 

*joint_finetune.py* --> high-level logic for finetuning

*joint_infer.py* --> inference algorithm using one-shot and edit network

*joint_plad.py* --> finetuning training logic

*main.py* --> main training/eval entrypoint

*model_utils.py* --> network utility functions

*os_models.py* --> one-shot model architecture

*os_pretrain.py* --> logic for one-shot network pretraining

*train_utils.py* --> general training related utility functions

*utils.py* --> general utility functions

*wake_sleep.py* --> logic for training and sampling generative network during finetuning

# Related Repos

[PLAD: Learning to Infer Shape Programs with Pseudo-Labels and Approximate Distributions](https://github.com/rkjones4/PLAD)
By [R. Kenny Jones](https://rkjones4.github.io/), [Homer Walke](https://homerwalke.com/), and [Daniel Ritchie](https://dritchie.github.io/). **CVPR 2022**

[Improving Unsupervised Visual Program Inference with Code Rewriting Families](https://github.com/bardofcodes/coref/)
By [Aditya Ganeshan](https://bardofcodes.github.io/), [R. Kenny Jones](https://rkjones4.github.io/), and [Daniel Ritchie](https://dritchie.github.io/). **ICCV 2023 (Oral)**

[Learning to Infer Generative Template Programs for Visual Concepts](https://github.com/rkjones4/TemplatePrograms)
By [R. Kenny Jones](https://rkjones4.github.io/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), and [Daniel Ritchie](https://dritchie.github.io/). **ICML 2024**

# Dependencies

This code was tested on Ubuntu 20.04, an NVIDIA 3090 GPU, python 3.9.7, pytorch 1.9.1, and cuda 11.1

The environment this code was developed in can be found in env.yml

# Acknowledgments

We would like to thank the anonymous reviewers for their helpful suggestions. Renderings of 3D shapes were produced using the Blender Cycles renderer. This work was funded in parts by NSF award #1941808 and a Brown University Presidential Fellowship. Daniel Ritchie is an advisor to Geopipe and owns equity in the company. Geopipe is a start-up that is developing 3D technology to build immersive virtual copies of the real world with applications in various fields, including games and architecture. 
