OUTPATH=$EXP/1800/motif/predcls/lt/external/relabel
mkdir -p $OUTPATH
cp $EXP/1800/motif/predcls/sup/sup/last_checkpoint $OUTPATH/last_checkpoint

torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
  tools/external_relabel.py --config-file "configs/wsup-1000.yaml" \
  DATASETS.TRAIN \(\"1000DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH ${NUM_GPUS} \
  DTYPE "float16" SOLVER.MAX_ITER 1000000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ${DATASETS_DIR}/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  SOLVER.PRE_VAL False \
  WSUPERVISE.DATASET ExTransDataset  EM.MODE E  WSUPERVISE.SPECIFIED_DATA_FILE  datasets/vg/1000/vg_clip_logits.pk


# cut top k% and merge with external information
cp tools/ietrans/external_cut.py $OUTPATH
cp tools/ietrans/na_score_rank.py $OUTPATH
cd $OUTPATH
python na_score_rank.py
python external_cut.py 1.0 # use all relabeled DS data
cd $SG
