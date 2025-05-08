CHECKPOINT="deepseek-ai/DeepSeek-Prover-V2-7B"
echo $CHECKPOINT
vllm serve ${CHECKPOINT} --tensor-parallel-size 2 --port 8000