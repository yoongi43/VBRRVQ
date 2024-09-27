#!/bin/bash

EXPNAME=$1
GPU=$2
TYPE=$3

MODEL_DIR=/data2/yoongi/dacruns

CUDA_VISIBLE_DEVICES=${GPU} \
python scripts/eval_vrvq.py \
--args.load conf/${EXPNAME}.yml \
--save_path ${MODEL_DIR}/${EXPNAME} \
--save_result_dir results \
--data_type ${TYPE} \
--no_visqol True