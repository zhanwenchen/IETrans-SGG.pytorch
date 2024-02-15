OUTPATH=$EXP/50/vctree/predcls/lt/combine/rwt

#cd $SG
torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
        tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
        TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR ${DATASETS_DIR}/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH OUTPUT_DIR $OUTPATH   \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
