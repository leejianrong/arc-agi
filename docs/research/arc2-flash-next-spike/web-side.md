# Flash-Next web spike (2026-10-31)

PRIMARY
- Model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next  (125B total/6B active MoE + 51B n-gram embedding + 4B MTP = 180B on Hub; 48 layers = 12 x (3 GatedDeltaNet + 1 Qwen Sparse Attention), 512 experts, 10 routed + 1 shared; ctx 262,144 native (YaRN to 1M); 1 MTP layer; vision encoder; license qwen-community-1.0; thinking sampling T=1.0 top_p=0.95 top_k=20; reasoning effort xhigh(default)/medium/low; suggested output limit 262,144 reasoning)
- FP8 card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 (block-128 FP8, no MTP cmd, no size)
- License: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE (commercial OK; name display only >100M MAU/$20M monthly revenue; MaaS / AI-work-assistant businesses need separate licence; redistribution needs notice; no competing-model clause seen; fetched via summariser, read full text before relying)
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next (stable vLLM >=0.30.0, tuned on nightly; FP8 172.78 GiB, BF16 335 GiB; engram CPU offload default, needs >=51 GB host RAM; flags --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --speculative-config {"method":"mtp","num_speculative_tokens":3} --max-num-seqs 256 --reasoning-parser qwen3 --tool-call-parser qwen3_coder; FP8 TP8 fails, use TEP; no engram PP)
- ARC Prize page for sibling Qwen3.8-27B: https://arcprize.org/results/alibaba-qwen3-8-27b (ARC-AGI-2 xhigh 42.4%, medium 13.2%, low 22.8%, none 1.5%; via Baseten API)
- Duck harness: https://github.com/Tufalabs/duck-harness and https://tufalabs.ai/research/duck-harness/ (ARC-AGI-3 harness; Qwen3.6-27B FP8; Python REPL, observations as variables, image+ASCII, evict oldest messages; vLLM local or OpenRouter; licence not seen)
- Kaggle ARC-AGI-2 leaderboard: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard (Tufa Labs 88.06, ~50% of test data)
- Poetiq repo: https://github.com/poetiq-ai/poetiq-arc-agi-solver (API-model harness; licence unclear in sources)

SECONDARY / HEARSAY
- 4x3090 W4A16 recipe https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe (patched vLLM PR head; weights 18.5 GiB/card TP4 + 1.2 GiB MTP; 107-117 tok/s single stream with MTP k=4; KV only 76k tokens at 65k ctx, 142k tokens with patched fp8 KV at 128k ctx; ~97 GiB host RAM for PLE table in BF16; needs ~110 GB free host RAM; mem util 0.96)
- https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory (KV ~24 KB/token BF16 derived: 12 QSA layers x 2 KV heads x 256; GGUF Q4 ~111 GB)
- Metirai blog https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026 (ARC-AGI-3 M2 winners run Duck on Qwen3.8-Flash-Next; scores 27.89/23.84/22.53 per search snippet)
- https://github.com/davidlueng986-alt/arc-agi-2-kaggle (says Tufa 88.06 unpublished)

NOT FOUND: any ARC-AGI-2 score for Flash-Next; any Tufa ARC-AGI-2 writeup; any Ada/SM89 report; Duck on ARC-AGI-2.
