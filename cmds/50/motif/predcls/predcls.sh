#!/bin/bash

SLURM_JOB_NAME=motif_none_visual_predcls_mini_1e3

export PROJECT_DIR=/home/zhanwen/ietrans
source ${PROJECT_DIR}/scripts/shared_functions/utils.sh
SLURM_JOB_ID=$(timestamp)
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
export DATASETS_DIR=${HOME}/datasets

export PREDICTOR=MotifPredictor
export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_motif.yaml
export USE_GRAFT=True
export GRAFT_ALPHA=0.5
export USE_SEMANTIC=True
export STRATEGY='cooccurrence-pred_cov'
export BOTTOM_K=30
export NUM2AUG=4
export MAX_BATCHSIZE_AUG=16


export WITH_CLEAN_CLASSIFIER=False
export WITH_TRANSFER_CLASSIFIER=False
export USE_GT_BOX=True
export USE_GT_OBJECT_LABEL=True
export PRE_VAL=False

if [ "${USE_SEMANTIC}" = True ]; then
    export BATCH_SIZE_PER_GPU=$((${MAX_BATCHSIZE_AUG} / 2))
else
    export BATCH_SIZE_PER_GPU=${MAX_BATCHSIZE_AUG}
fi

export CUDA_VISIBLE_DEVICES=0,1

export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
export BATCH_SIZE=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))
export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# specify current directory as SG, e.g.:
export SG=${HOME}/ietrans
# specify experiment directory, e.g.:
export EXP=${HOME}/experiments
export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl


# train a supervised model
bash cmds/50/motif/predcls/sup/train.sh
# # conduct internal transfer
# bash cmds/50/motif/predcls/lt/internal/relabel.sh
# # conduct external transfer
# bash cmds/50/motif/predcls/lt/external/relabel.sh
# # combine internal and external transferred data
# bash cmds/50/motif/predcls/lt/combine/combine.sh
# # train a new model
# #bash cmds/50/motif/predcls/lt/combine/train.sh
# # train a new model using rwt
# bash cmds/50/motif/predcls/lt/combine/train_rwt.sh
