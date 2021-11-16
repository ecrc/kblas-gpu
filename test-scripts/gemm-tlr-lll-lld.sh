#!/bin/bash

KBLAS_HOME=.
#--------------------
START_DIM=5120
STOP_DIM=40960
STEP_DIM=5120
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

echo -n "testing TLR-GEMM-LLD..."
FILE=${RESDIR}/gemm-tlr-lld.txt

CMD="./testing/bin/test_dgemm_tlr --range $START_DIM:25600+$STEP_DIM --rank 16 --nb $NB --batch 1000 --svd 2  -t -w --LR b --magma"
echo ${CMD} >> ${FILE}
${CMD} >> ${FILE}

CMD="./testing/bin/test_dgemm_tlr --range $START_DIM:$STOP_DIM+$STEP_DIM --rank 16 --nb $NB --batch 1000 --svd 2 -w --LR b --magma"
echo ${CMD} >> ${FILE}
${CMD} >> ${FILE}
echo " done, data dumped in ${FILE}"

echo -n "testing TLR-GEMM-LLL..."
FILE=${RESDIR}/gemm-tlr-lll.txt
CMD="./testing/bin/test_dgemm_tlr --range $START_DIM:61440+$STEP_DIM --rank 32 --nb $NB --batch 1000 --svd 2  -t -w --LR t --magma"
echo ${CMD} >> ${FILE}
${CMD} >> ${FILE}
echo " done, data dumped in ${FILE}"
