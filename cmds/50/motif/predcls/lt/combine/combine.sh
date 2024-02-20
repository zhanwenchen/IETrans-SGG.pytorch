OUTPATH=$EXP/50/motif/predcls/lt/combine/relabel
mkdir -p $OUTPATH
cd $OUTPATH
cp ${PROJECT_DIR}/tools/ietrans/combine.py ./
python combine.py motif
