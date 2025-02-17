OUTPATH=$EXP/50/motif/predcls/sup/sup
mkdir -p $OUTPATH

#CUDA_LAUNCH_BLOCKING=1
torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
  tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
  MODEL.ROI_RELATION_HEAD.PREDICTOR ${PREDICTOR} \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_OBJECT_LABEL} \
  MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER ${WITH_CLEAN_CLASSIFIER} \
  MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER ${WITH_TRANSFER_CLASSIFIER}  \
  SOLVER.AUGMENTATION.USE_SEMANTIC ${USE_SEMANTIC} \
  SOLVER.AUGMENTATION.USE_GRAFT ${USE_GRAFT} \
  SOLVER.AUGMENTATION.GRAFT_ALPHA ${GRAFT_ALPHA} \
  SOLVER.AUGMENTATION.NUM2AUG ${NUM2AUG} \
  SOLVER.AUGMENTATION.MAX_BATCHSIZE_AUG ${MAX_BATCHSIZE_AUG} \
  SOLVER.AUGMENTATION.STRATEGY ${STRATEGY} \
  SOLVER.AUGMENTATION.BOTTOM_K ${BOTTOM_K} \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH ${BATCH_SIZE} TEST.IMS_PER_BATCH ${NUM_GPUS} DTYPE "float16" \
  SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ${DATASETS_DIR}/glove \MODEL.PRETRAINED_DETECTOR_CKPT ${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  TEST.METRIC "R"
