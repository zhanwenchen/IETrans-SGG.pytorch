OUTPATH=$EXP/1800/motif/predcls/sup/sup
mkdir -p $OUTPATH

torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
  tools/relation_train_net.py --config-file "configs/sup-1000.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH ${NUM_GPUS} DTYPE "float16" \
  SOLVER.MAX_ITER 40000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ${DATASETS_DIR}/glove MODEL.PRETRAINED_DETECTOR_CKPT ${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  TEST.METRIC "R"
