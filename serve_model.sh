CHECKPOINT="deepseek-ai/DeepSeek-Prover-V2-7B"
echo $CHECKPOINT
vllm serve ${CHECKPOINT} \
  --tensor-parallel-size 2 \
  --port 8000 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \