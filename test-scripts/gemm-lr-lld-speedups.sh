#!/bin/bash

KBLAS_HOME=.
#--------------------
START_DIM=32
STOP_DIM=800
STEP_DIM=32

START_RANK=16
STOP_RANK=128
STEP_RANK=2
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

FILE=${RESDIR}/gemm-lr-lld-speedups.txt
for (( r=$START_RANK; r <= $STOP_RANK; r*=$STEP_RANK ))
do
  echo "testing LR-GEMM-LLD with rank $r..."
  START_DIM_valid=$START_DIM
  if [ "$START_DIM_valid" -lt "$r" ]; then
    START_DIM_valid=$r
  fi
  CMD="./testing/bin/test_dgemm_lr --range $START_DIM_valid:$STOP_DIM+$STEP_DIM --rank $r --batch 1000 --svd 2 -t -w --LR b --magma"
  echo ${CMD} >> ${FILE}
  ${CMD} >> ${FILE}
  echo >> ${FILE}
done
echo " done, data dumped in ${FILE}"

