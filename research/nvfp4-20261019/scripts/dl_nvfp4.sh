#!/bin/bash
# download Qwen3.8-Flash-Next NVFP4 GPTQ (Strata) via hf-mirror, per-file aria2c, then verify sha256
set -u
D=/root/strata/Strata-data/models/qwen-nvfp4-gptq
REPO=Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata
REV=379ff0b8a39f
BASE="https://hf-mirror.com/${REPO}/resolve/${REV}"
LOG=/root/strata/dl-nvfp4.log
mkdir -p $D/mtp $D/pack/tokenizer
echo "== start $(date -Is)" > $LOG
FILES="LICENSE SHA256SUMS SIZES.txt README.md mtp/dense.bin mtp/dense.txt mtp/draft_vocab.bin mtp/experts.bin \
pack/dense.bin pack/experts.bin pack/index.txt pack/native_experts.txt pack/tokenizer/chat_template.jinja \
pack/tokenizer/merges.txt pack/tokenizer/token_type.json pack/tokenizer/tokenizer.json pack/tokenizer/vocab.json \
ple-fp8.gguf qwen-nvfp4-gptq-dense.gguf token-embd-bf16.gguf"
for f in $FILES; do
  case "$f" in *.bin|*.gguf) X="-x16 -s16 -k8M";; *) X="-x4 -s4";; esac
  d=$(dirname "$D/$f"); mkdir -p "$d"
  for try in 1 2 3 4 5; do
    ionice -c2 -n7 nice -n10 aria2c -q --allow-overwrite=true --auto-file-renaming=false -c $X -U "Mozilla/5.0" \
      -d "$d" -o "$(basename $f)" "$BASE/$f" && break
    echo "retry $try $f $(date +%H:%M:%S)" >> $LOG
    sleep 10
  done
  echo "$(date +%H:%M:%S) done $f $(stat -c %s "$D/$f" 2>/dev/null)" >> $LOG
done
echo "== verifying" >> $LOG
cd $D && sha256sum -c SHA256SUMS >> $LOG 2>&1
echo "VERIFY_EXIT=$?" >> $LOG
du -sh $D >> $LOG
echo DL_NVFP4_DONE >> $LOG
