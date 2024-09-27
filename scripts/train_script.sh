#!/bin/bash

CONFIG_DIR=conf/
SAVE_DIR=/data2/yoongi/dacruns

EXPNAME=${1}
GPU=${2}
# RESUME=${2}

CUDA_VISIBLE_DEVICES=${GPU} \
taskset -c $((8*${GPU}))-$((8*${GPU}+7)) \
python scripts/train_vrvq.py \
--args.load ${CONFIG_DIR}/${EXPNAME}.yml \
--save_path ${SAVE_DIR}/${EXPNAME}

# torchrun --nproc_per_node gpu scripts/train_imp_grad.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} --resume