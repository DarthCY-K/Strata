#!/bin/bash
# deploy Strata NVFP4 GPTQ (fork engine) on the 5090, single GPU; stops the miner first
set -u
T=/root/strata/Strata-nvfp4
M=/root/strata/Strata-data/models/qwen-nvfp4-gptq
CFG=$T/strata-nvfp4-gptq.json
RUN=$T/run-nvfp4-gptq.sh
LOG=$T/strata-nvfp4-gptq.log
echo "== stop miner $(date -Is)"
pkill -x wildrig-multi; sleep 4; pgrep -x wildrig-multi >/dev/null && { pkill -9 -x wildrig-multi; sleep 2; }
pgrep -af wildrig | head -3
bash /root/miners/gpu-mode.sh model || true
echo "== stop any strata"
python3 /root/strata/stop_strata.py >/dev/null 2>&1; sleep 2
mkdir -p /root/strata/switch-nvfp4-20261019
KEY=$(python3 -c 'import json;print(json.load(open("/root/strata/Strata-145/strata-sc117-iq3_s.json"))["api_key"])')
python3 - "$KEY" <<'EOF'
import json, sys
T = "/root/strata/Strata-nvfp4"; M = "/root/strata/Strata-data/models/qwen-nvfp4-gptq"
key = sys.argv[1]
d = {
 "exe": f"{T}/engine/strata",
 "args": ["--pack", f"{M}/pack", "--native", f"{M}/qwen-nvfp4-gptq-dense.gguf",
          "--native-dense-gguf", f"{M}/qwen-nvfp4-gptq-dense.gguf",
          "--ple-gguf", f"{M}/ple-fp8.gguf", "--embd-gguf", f"{M}/token-embd-bf16.gguf",
          "--expert-profile", f"{T}/data/expert-profile.bin",
          "--expert-cache", "auto", "--prefill", "auto",
          "--spec", "6", "--spec-min-p", "0.7", "--mtp", f"{M}/mtp",
          "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
          "--vram-reserve-mib", "1000"],
 "cwd": T, "tokenizer": f"{M}/pack/tokenizer", "model_name": "Qwen3.8-Flash-Next",
 "log": f"{T}/strata-nvfp4-gptq.log", "port": 8080, "gpu": [0],
 "gpus_asked": True, "open_browser": False,
 "env": {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
 "uncensored": True, "reasoning_budget_tokens": 32768,
 "sampling": {"presence_penalty": 0},
 "api_key": key,
}
json.dump(d, open(f"{T}/strata-nvfp4-gptq.json", "w"), indent=1, ensure_ascii=False)
print("config written:", " ".join(d["args"]))
EOF
cat > $RUN <<EOF
#!/bin/sh
cd "$T"
exec "/root/strata/Strata-145/.venv/bin/python" "$T/serve/server.py" --engine strata --config "$CFG" --port 8080
EOF
chmod +x $RUN
echo "== start"
setsid nohup bash $RUN > /root/strata/switch-nvfp4-20261019/serve.log 2>&1 < /dev/null &
for i in $(seq 1 160); do
  sleep 3
  grep -q "^ready:" /root/strata/switch-nvfp4-20261019/serve.log && break
  pgrep -f "Strata-nvfp4/serve/server.py" >/dev/null || { echo "SERVER_EXITED"; break; }
done
tail -6 /root/strata/switch-nvfp4-20261019/serve.log | cut -c1-300
echo "== engine log highlights"
grep -iE "nvfp4|type 40|w4a8|w4a4|experts loaded|filling the GPU|VRAM free|uncensor|error" $LOG | tail -12 | cut -c1-260
free -g | sed -n 2p
nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader
