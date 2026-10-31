#!/usr/bin/env bash
# Pod-side experiment: serve Qwen3.8-27B-FP8 with vLLM (TP=4), run the direct-answer
# solver over public ARC-AGI-2 eval tasks, relay results back. Runs under rp's
# dead-man's-switch; every network step is wrapped in `timeout`.
# MIT No Attribution (MIT-0).
set -u
cd /workspace
echo "PHASE=start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
df -h /workspace | tail -1

export DEBIAN_FRONTEND=noninteractive
timeout 300 apt-get update -qq && timeout 600 apt-get install -y -qq build-essential git >/dev/null
echo "PHASE=apt_done $(date -u +%FT%TZ)"

timeout 120 git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git data || echo "CLONE_FAILED"
python3 prep_data.py data/data/evaluation challenges.json solutions.json

timeout 120 pip install -q uv
timeout 1200 uv venv /workspace/venv --python 3.12 && \
  timeout 1500 uv pip install -q --python /workspace/venv/bin/python "vllm==${VLLM_VERSION:-0.19.0}" "transformers==4.57.6"
echo "PHASE=vllm_installed $(date -u +%FT%TZ)"
/workspace/venv/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.cuda.device_count(),'gpus')"

export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p out

# Configs tried in order: "<MTP tokens> <kv dtype> <attention backend>". SM89 (L4 and RTX 4090) needs
# Triton for an FP8 KV cache, and the MTP drafter must be pinned to it (VLLM_FORCE_ATTN_BACKEND).
# A config is accepted only after a burst of real generation requests succeeds: MTP configs have
# passed /health and then died on the first requests.
up=0; CHOSEN=""
IFS='|' read -ra CFGS <<< "${CONFIGS:-2 fp8 TRITON_ATTN|0 fp8 TRITON_ATTN|0 auto FLASH_ATTN}"
for cfg in "${CFGS[@]}"; do
  read -r mtp kv attn <<< "$cfg"
  name="mtp${mtp}_${kv}_${attn}"
  echo "TRYING config $name $(date -u +%FT%TZ)"
  SPEC=()
  [ "$mtp" -gt 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${mtp}}")
  VLLM_FORCE_ATTN_BACKEND="$attn" nohup /workspace/venv/bin/vllm serve "${MODEL:-Qwen/Qwen3.8-27B-FP8}" \
    --served-model-name qwen --tensor-parallel-size "${TP:-4}" --max-model-len "${MAX_LEN:-65536}" \
    --gpu-memory-utilization 0.94 --max-num-seqs "${NSEQ:-16}" --port 8000 --trust-remote-code \
    --kv-cache-dtype "$kv" --attention-backend "$attn" --no-async-scheduling \
    --reasoning-parser qwen3 "${SPEC[@]}" > "/workspace/vllm_$name.log" 2>&1 &
  VLLM_PID=$!
  ok=0
  for i in $(seq 1 180); do   # 30 min: cold compile on a slow host exceeded 20
    if curl -sf --max-time 5 http://127.0.0.1:8000/health >/dev/null; then ok=1; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "VLLM_DIED $name"; break; }
    [ $((i % 6)) -eq 0 ] && { echo "waiting for vllm $name ($((i * 10))s)"; tail -n 1 "/workspace/vllm_$name.log" | cut -c1-200; }
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    python3 smoke.py && { up=1; CHOSEN="$name"; ln -sf "/workspace/vllm_$name.log" /workspace/vllm.log; break; }
    echo "SMOKE_FAILED $name"
  fi
  grep -n -E "Error|Traceback|illegal|assert|Exception|CUDA" "/workspace/vllm_$name.log" | grep -v "POST /v1" | head -n 40 > "out/crash_$name.log"
  kill "$VLLM_PID" 2>/dev/null
  # Workers outlive the API server and keep GPU memory; the next config fails to start unless they die.
  pkill -9 -f "vllm" 2>/dev/null; pkill -9 -f "VLLM" 2>/dev/null; pkill -9 -f "multiprocessing" 2>/dev/null
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-99999}" -lt 1500 ] && break
    sleep 3
  done
  echo "gpu memory used after cleanup: ${used:-?} MiB"
done
echo "CHOSEN_CONFIG=$CHOSEN" | tee out/config.txt
echo "PHASE=vllm_up=$up $(date -u +%FT%TZ)"

if [ "$up" = 1 ]; then
  ( while true; do sleep 120; grep -E "generation throughput|SpecDecoding" /workspace/vllm.log | tail -n 2 | cut -c1-260; done ) &
  STATS_PID=$!
  if [ "${MODE:-direct}" = synth ]; then
    python3 synth.py --challenges challenges.json --solutions solutions.json --out out/submission.json \
      --log-dir out/synth_logs --limit "${LIMIT:-24}" --n-parallel "${NPAR:-16}" \
      --max-tokens "${MAX_TOKENS:-32768}" --deadline-min "${DEADLINE_MIN:-60}" --effort "${EFFORT:-medium}" \
      ${SYNTH_ARGS:-}
  else
    python3 llm_solve.py --challenges challenges.json --solutions solutions.json --out out/submission.json \
      --log-dir out/llm_logs --limit "${LIMIT:-24}" --n-parallel "${NPAR:-16}" \
      --max-tokens "${MAX_TOKENS:-24000}" --deadline-min "${DEADLINE_MIN:-60}" --effort "${EFFORT:-medium}"
  fi
  kill "$STATS_PID" 2>/dev/null
fi
echo "PHASE=solve_done $(date -u +%FT%TZ)"
tail -n 60 /workspace/vllm.log > out/vllm_tail.log 2>/dev/null; cp /workspace/vllm_mtp*.log out/ 2>/dev/null; for f in out/vllm_mtp*.log; do tail -n 300 "$f" > "$f.tail"; rm "$f"; done
grep -E "generation throughput|SpecDecoding" /workspace/vllm.log | tail -n 40 > out/vllm_throughput.log 2>/dev/null
tar czf /workspace/result.tar.gz -C /workspace out
base="arc2exp-$RANDOM"
runpodctl send --code "$base" /workspace/result.tar.gz >/tmp/send.log 2>&1 &
sp=$!
code=""
for _ in $(seq 1 40); do code=$(grep -oE "${base}[-0-9a-z]+" /tmp/send.log | head -1); [ -n "$code" ] && break; sleep 0.5; done
echo "RP_ARTIFACT_CODE=${code:-send-failed}"
wait "$sp"
