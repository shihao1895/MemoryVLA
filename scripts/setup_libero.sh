#!/usr/bin/env bash
# MemoryVLA LIBERO eval env — installs into container-local /root/envs/libero
# Usage: `WORKSPACE=/path/to/MemoryVLA bash scripts/setup_libero.sh`
# Originally written for a PAI air-gapped container; also works on any Linux box
# with HF/pypi egress as long as $WORKSPACE is the MemoryVLA repo root.
# Idempotent. Safe to re-run. Logs to $WORKSPACE/logs/libero/setup.log.

set -e
JOB=libero
WORKSPACE=${WORKSPACE:-$(pwd)}
VENV=/root/envs/$JOB
LOG=$WORKSPACE/logs/$JOB/setup.log

# HF/ckpts on 3fs (large files OK); pip cache + TMPDIR on local /root (small files, avoid 3fs RIO)
export HF_HOME=$WORKSPACE/cache/hf
export HF_HUB_CACHE=$WORKSPACE/cache/hf/hub
export TRANSFORMERS_CACHE=$WORKSPACE/cache/hf/transformers
export TORCH_HOME=$WORKSPACE/cache/torch
export PIP_CACHE_DIR=/root/.cache/pip
export XDG_CACHE_HOME=/root/.cache
export TMPDIR=/root/tmp
export PYTHONNOUSERSITE=1
export DEBIAN_FRONTEND=noninteractive

mkdir -p $WORKSPACE/logs/$JOB $WORKSPACE/ckpts/$JOB $WORKSPACE/third_libs \
  $WORKSPACE/cache/hf/hub $WORKSPACE/cache/hf/transformers $WORKSPACE/cache/torch \
  /root/envs /root/.cache/pip /root/tmp

step() { echo "=== [$(date -Is)] $*" | tee -a $LOG; }
trap 'echo "FAILED at line $LINENO" | tee -a $LOG' ERR

step "$JOB setup start"

step "step 1: apt deps (idempotent)"
apt-get update -qq >> $LOG 2>&1 || true
apt-get install -y git git-lfs curl wget ca-certificates build-essential pkg-config \
  python3 python3-pip python3-venv python3.10-venv ffmpeg \
  libegl1-mesa libgl1-mesa-dev libgles2-mesa-dev libgl1 libglib2.0-0 \
  libglx-mesa0 libopengl0 libglu1-mesa mesa-utils libvulkan1 vulkan-tools \
  libosmesa6 libosmesa6-dev \
  lsof >> $LOG 2>&1
git lfs install --skip-repo >> $LOG 2>&1 || true

step "step 2: venv at $VENV"
torch_ok=0
if [ -x $VENV/bin/python ]; then
  if $VENV/bin/python -c "import torch; assert torch.__version__.startswith('2.2.0'), torch.__version__; import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" 2>>$LOG; then
    torch_ok=1; step "  venv exists and torch/numpy verified"
  else
    step "  venv exists but torch or numpy broken — wiping"
    rm -rf $VENV
  fi
fi
[ -x $VENV/bin/python ] || python3 -m venv $VENV >> $LOG 2>&1
source $VENV/bin/activate
python -V >> $LOG 2>&1

step "step 3: pip upgrade + build deps + pin numpy<2 BEFORE torch"
pip install --upgrade pip setuptools wheel >> $LOG 2>&1
pip install psutil packaging ninja >> $LOG 2>&1
pip install "numpy==1.26.4" >> $LOG 2>&1

step "step 4: torch 2.2.0 (retry up to 5× against 3fs RIO)"
if [ $torch_ok -ne 1 ]; then
  ok=0
  for i in 1 2 3 4 5; do
    if pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 >> $LOG 2>&1 \
       && pip install "numpy==1.26.4" --force-reinstall >> $LOG 2>&1 \
       && python -c "import torch; assert torch.__version__.startswith('2.2.0'), torch.__version__; import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" 2>>$LOG; then
      step "  torch+numpy verified on attempt $i"; ok=1; break
    fi
    step "  attempt $i broken, wipe + retry"
    pip uninstall -y torch torchvision torchaudio 2>>$LOG || true
    rm -rf $VENV/lib/python3.10/site-packages/torch* 2>>$LOG
    sleep 5
  done
  [ $ok -eq 1 ] || { step "torch install gave up after 5 retries"; exit 1; }
fi

step "step 5: ensure vla/__init__.py has dlimp-optional guard (needed while real dlimp not staged)"
if ! grep -q "sec-test dlimp workaround" $WORKSPACE/vla/__init__.py; then
  cp $WORKSPACE/vla/__init__.py $WORKSPACE/vla/__init__.py.bak
  cat > $WORKSPACE/vla/__init__.py <<'PY'
# sec-test dlimp workaround: materialize.py pulls dlimp (training data loading);
# eval-only (deploy.py -> load_vla) doesn't need datasets. Guard the import so
# load_vla works without dlimp. Restore full import once real dlimp is staged.
from .memory_vla import MemoryVLA
from .load import available_model_names, available_models, get_model_description, load, load_vla
try:
    from .materialize import get_vla_dataset_and_collator
except ModuleNotFoundError as _e:
    import warnings as _w
    _w.warn(f"vla.materialize not importable ({_e.name} missing) — eval still works, training disabled")
    def get_vla_dataset_and_collator(*args, **kwargs):
        raise ImportError(
            "get_vla_dataset_and_collator requires dlimp; not available in this env. "
            "Install real dlimp (moojink/dlimp_openvla) for training."
        )
PY
fi

step "step 6: pip install -e memvla (pyproject.toml is patched: requires-python, flash_attn commented, dlimp commented)"
cd $WORKSPACE
for i in 1 2 3; do
  if pip install -e . >> $LOG 2>&1; then
    step "  memvla install OK on attempt $i"; break
  fi
  step "  memvla install attempt $i failed, retry"; sleep 5
done

step "step 7: SUPPLY-CHAIN CHECK — make sure no sec-test dlimp slipped in"
# Internal pypi has a dlimp==0.1.0 impersonator (author sectest@example.com, only print stmt).
# If it's installed, rip it out immediately.
if pip show dlimp 2>/dev/null | grep -qiE "sectest|sec-test|yourusername|ipablepytorch"; then
  step "  ⚠️  sec-test dlimp DETECTED — uninstalling"
  pip uninstall -y dlimp >> $LOG 2>&1 || true
  rm -rf $VENV/lib/python3.10/site-packages/pyav
fi

step "step 8: transforms3d (used by evaluation/simpler_env/vla_policy.py; legit pypi package)"
pip install transforms3d >> $LOG 2>&1

step "step 9a: LLaMA-2-7b config+tokenizer mirror (gated meta-llama unreachable; using NousResearch mirror — identical content)"
LLAMA_LOCAL=$WORKSPACE/cache/llama2-7b-local
if [ ! -f $LLAMA_LOCAL/config.json ] || [ ! -f $LLAMA_LOCAL/tokenizer.model ]; then
  ML_SNAP=$WORKSPACE/cache/hf/hub/models--meta-llama--Llama-2-7b-hf/snapshots
  if [ -d $ML_SNAP ]; then
    SHA=$(ls $ML_SNAP | head -1)
    mkdir -p $LLAMA_LOCAL
    for f in config.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json generation_config.json; do
      [ -e $ML_SNAP/$SHA/$f ] && cp -L $ML_SNAP/$SHA/$f $LLAMA_LOCAL/
    done
    step "  llama2-7b-local materialized at $LLAMA_LOCAL ($(ls $LLAMA_LOCAL | wc -l) files)"
  else
    step "  WARN: $ML_SNAP missing — supervisor must pre-stage HF cache for meta-llama or NousResearch/Llama-2-7b-hf"
  fi
fi
# Workaround: prismatic/models/backbones/llm/llama2.py picks up MEMVLA_LLAMA2_7B_LOCAL_PATH
# when set (env var), bypassing the buggy transformers cache lookup for `meta-llama/Llama-2-7b-hf`.
echo "export MEMVLA_LLAMA2_7B_LOCAL_PATH=$LLAMA_LOCAL" >> /root/.bashrc 2>/dev/null || true

step "step 9b: LIBERO third_lib — build per-item symlink dir (avoid shizhao.sun's LIBERO/LIBERO self-loop) and register via .pth"
SRC=/mnt/3fs/data/shizhao.sun/eval_workspace/xyz_vla/evaluation/LIBERO_EVAL/LIBERO
DST=$WORKSPACE/third_libs/LIBERO
if [ -d $SRC ]; then
  # Nuke any prior symlink/dir and rebuild fresh
  if [ -L $DST ]; then rm -f $DST; fi
  if [ -d $DST ] && [ ! -L $DST/setup.py ]; then :; fi  # already a dir with symlinks, keep
  if [ ! -d $DST ]; then mkdir -p $DST; fi
  # populate per-item symlinks, skipping the self-loop `LIBERO` and stale `libero.egg-info`
  for name in $(ls -A $SRC); do
    case "$name" in
      LIBERO|libero.egg-info) continue ;;
    esac
    [ -e $DST/$name ] || ln -s $SRC/$name $DST/$name
  done
  # LIBERO's `libero/` has no __init__.py — namespace package. Use a .pth file to put
  # the repo root on sys.path so `from libero.libero import ...` resolves.
  echo "$DST" > $VENV/lib/python3.10/site-packages/libero-repo.pth
  step "  LIBERO dir built at $DST (per-item symlinks) and registered via libero-repo.pth"
else
  step "  WARN: LIBERO source not found at $SRC — skipping"
fi

step "step 10: smoke (imports + LIBERO benchmark enumeration, no ckpt load)"
cd $WORKSPACE
find vla -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
# IMPORTANT: pass env vars via separate `export` lines (single-line `export A=$X B=$X/y` won't expand $X)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMVLA_LLAMA2_7B_LOCAL_PATH=$WORKSPACE/cache/llama2-7b-local
SMOKE=$WORKSPACE/scripts/_smoke_libero_inline.py
cat > $SMOKE <<'PY'
import sys, os
print("Python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available(),
      "dev count:", torch.cuda.device_count())
import transformers, numpy
print("transformers:", transformers.__version__, "numpy:", numpy.__version__)
try:
    import flash_attn; print("flash_attn:", flash_attn.__version__)
except ModuleNotFoundError:
    print("flash_attn: not installed (training-only per README)")
from vla import load_vla
print("vla.load_vla: OK")
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()
print("LIBERO suites:", list(bm.keys()))
s = bm["libero_spatial"](); print(f"libero_spatial: {s.n_tasks} tasks; task[0]: {s.get_task(0).name}")
from libero.libero.envs import OffScreenRenderEnv
print("libero.libero.envs.OffScreenRenderEnv: OK")
print("=== smoke OK ===")
PY
python $SMOKE 2>&1 | tee -a $LOG
rm -f $SMOKE

step "$JOB setup DONE — venv: $VENV, repo: $WORKSPACE"
echo "  pip list installed: $(pip list 2>/dev/null | wc -l) packages" | tee -a $LOG
echo "  du -sh $VENV: $(du -sh $VENV 2>/dev/null | cut -f1)" | tee -a $LOG
echo "  du -sh $WORKSPACE/third_libs/LIBERO: $(du -sh -L $WORKSPACE/third_libs/LIBERO 2>/dev/null | cut -f1) (via symlinks)" | tee -a $LOG
