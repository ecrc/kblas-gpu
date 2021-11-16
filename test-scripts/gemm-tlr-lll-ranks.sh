#!/bin/bash

KBLAS_HOME=.
#--------------------
START_DIM=5120
STOP_DIM=61440
STEP_DIM=5120

START_RANK=16
STOP_RANK=256
STEP_RANK=2
NB=1024
#--------------------
GPU=0
if [ -z $CUDA_VISIBLE_DEVICES ]; then
  NGPUS=$(nvidia-smi -L | wc -l)
else
  aux=${CUDA_VISIBLE_DEVICES/,,/,} # replace double commas
  aux=${aux/%,/} # remove trailing comma
  aux=${aux/#,/} # remove leading comma
  NGPUS=$(echo "$aux," |  awk 'BEGIN{FS=","} {print NF?NF-1:0}')
fi
#--------------------
cd $KBLAS_HOME
# cd testing/bin
RESDIR=./kblas-test-log
mkdir -p ${RESDIR}

if [ $NGPUS -lt "1" ]; then
  echo "KBLAS: No GPUs detected to test on! Exiting..."
  exit
fi

FILE=${RESDIR}/gemm-tlr-lll-ranks.txt
for (( r=$START_RANK; r <= $STOP_RANK; r*=$STEP_RANK ))
do
  echo "testing TLR-GEMM-LLL with rank $r..."
  CMD="./testing/bin/test_dgemm_tlr --range $START_DIM:$STOP_DIM+$STEP_DIM --rank $r --nb $NB --batch 1000 --svd 2 -t -w --LR t --magma"
  echo ${CMD} >> ${FILE}
  ${CMD} >> ${FILE}
  echo >> ${FILE}
done
echo " done, data dumped in ${FILE}"

