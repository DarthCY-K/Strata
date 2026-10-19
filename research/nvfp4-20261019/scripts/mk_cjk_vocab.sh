#!/bin/bash
# build an NVFP4 MTP folder variant with CJK added to the draft vocabulary (originals untouched)
set -u
T=/root/strata/Strata-nvfp4
M=/root/strata/Strata-data/models/qwen-nvfp4-gptq
NEW=$M/mtp-cjk
PY=/root/strata/Strata-145/.venv/bin/python
mkdir -p $NEW
for f in dense.bin dense.txt experts.bin; do ln -f $M/mtp/$f $NEW/$f 2>/dev/null || cp -p $M/mtp/$f $NEW/$f; done
cd $T
echo "== stats of shipped subset"
$PY tools/draft_vocab.py --gguf $M/qwen-nvfp4-gptq-dense.gguf --base $M/mtp/draft_vocab.bin --stats 2>&1 | tail -12
echo "== add cjk"
$PY tools/draft_vocab.py --gguf $M/qwen-nvfp4-gptq-dense.gguf --base $M/mtp/draft_vocab.bin --add cjk --out $NEW/draft_vocab.bin --stats 2>&1 | tail -14
ls -la $NEW
python3 -c "import os;print('ids', os.path.getsize('$NEW/draft_vocab.bin')//4)"
echo "== compare with SC117's subset size"
python3 -c "import os;print('sc117 ids', os.path.getsize('/root/strata/Strata-data/mtp/rt-sc117/draft_vocab.bin')//4)"
