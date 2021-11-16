# To be sourced in the jenkins script. Currenly for bash only!
# Define usable GPUs
# If node has mixed gpus (example: k20 +k40) make sure it runs on specified gpu

if [[ $NODE_LABELS == *"mixedgpu"* ]]
then
    CUDA_VISIBLE_DEVICES=`nvidia-smi -L | tac | grep -i "$gpuarch" | cut -d : -f 1 |cut -c 5- | tr '\n' ,`
else
    if [ -z $CUDA_VISIBLE_DEVICES ]
    then
        CUDA_VISIBLE_DEVICES=`nvidia-smi -L | tac | grep -i "$gpuarch" | cut -d : -f 1 |cut -c 5- | tr '\n' ,`
    fi
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/%,/}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES," |  awk 'BEGIN{FS=","} {print NF?NF-1:0}')
echo "NUMBER OF USABLE GPUS: $NGPUS [$CUDA_VISIBLE_DEVICES]"
