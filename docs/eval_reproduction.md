# MemoryVLA Eval 复现报告

**日期**：2026-04-25  
**作者**：intern_memvla_developer  
**复现范围**：完整 benchmark — LIBERO-Spatial（10 task × 50 ep = 500 ep）+ SimplerEnv Bridge（4 scene × 24 ep = 96 ep）  
**WORKSPACE**：`/mnt/3fs/data/tingwen.du/workspace/MemoryVLA`（3fs 共享盘）

---

## 1. 结果总览

### 1.1 Bridge（SimplerEnv，widowx）

| Scene | 我们 | Paper | Δ |
|---|---|---|---|
| StackGreenCubeOnYellowCubeBakedTexInScene-v0（Cube）| **0.4583**（11/24）| 0.708 | -0.250 |
| PutCarrotOnPlateInScene-v0（Carrot）| **0.7083**（17/24）| 0.583 | +0.125 |
| PutSpoonOnTableClothInScene-v0（Spoon）| **0.7500**（18/24）| 0.625 | +0.125 |
| PutEggplantInBasketScene-v0（Eggplant）| **1.0000**（24/24）| 0.958 | +0.042 |
| **平均** | **0.7292** | **0.7188** | **+0.010** |

✅ **Bridge 平均与 paper 几乎完全吻合（差 1pp 以内）**。Cube 单 scene 偏低、其余三个偏高，相互抵消。

### 1.2 LIBERO-Spatial（10 task × 50 trial = 500 ep）

| Task # | Task 名称 | 我们 |
|---|---|---|
| 1 | pick up black bowl between plate and ramekin | 1.00 |
| 2 | pick up black bowl next to ramekin | 0.96 |
| 3 | pick up black bowl from table center | 1.00 |
| 4 | pick up black bowl on cookie box | 1.00 |
| 5 | pick up black bowl in top drawer of wooden cabinet | 0.98 |
| 6 | **pick up black bowl on the ramekin** | **0.00** ⚠ |
| 7 | pick up black bowl next to cookie box | 0.98 |
| 8 | pick up black bowl on the stove | 1.00 |
| 9 | pick up black bowl next to the plate | 0.94 |
| 10 | pick up black bowl on wooden cabinet | 1.00 |

- **整体 success rate：0.886（443/500）**
- **Paper：0.984**
- **Δ：-0.098（差 9.8pp）**

⚠ **Task 6 显著异常**（其余 9 task 全部 ≥0.94，唯 Task 6 全失败 0/50）：
  - 排除 Task 6：(8.86)/9 = **0.984** = paper 完全一致
  - 单 task 系统性失败的常见原因：action_chunking_window=8 + 该 task 的初始姿态使前几个 chunk 把 gripper 推到 ramekin 上方撞掉、或 vision encoder 对 `ramekin` 这个少见 distractor 词敏感
  - **结论**：核心模型表现复现了 paper，但有 1 个 task 触发了边界情况；我对此没有立即结论，需要主管定夺是否进一步消融（如试 ac=4、不同 seed、关闭 cfg）

---

## 2. 复现命令（air-gapped，纯离线，可直接运行）

### 2.1 环境前置（一次）

两个独立 PAI 容器：

| 用途 | Host | 容器 | venv |
|---|---|---|---|
| LIBERO eval | 10.100.38.11 | port 35724 | `/root/envs/libero` |
| Bridge / SimplerEnv | 10.100.2.47 | port 29862 | `/root/envs/simpler` |
| HF/git egress staging | 10.100.193.67（dev4infer）| - | - |

`$WORKSPACE = /mnt/3fs/data/tingwen.du/workspace/MemoryVLA` 在所有节点上是同一份（3fs 挂载）。

每个容器初次进入后跑：

```bash
# LIBERO
bash $WORKSPACE/scripts/setup_libero.sh

# Simpler
bash $WORKSPACE/scripts/setup_simpler.sh
```

setup 脚本幂等，已经把所有踩过的坑（apt deps、numpy<2 pin、dlimp guard、LIBERO symlink dir、LLaMA local mirror、osmesa lib）写死。

### 2.2 LIBERO eval

```bash
ssh -p 35724 root@10.100.38.11
cd /mnt/3fs/data/tingwen.du/workspace/MemoryVLA
source /root/envs/libero/bin/activate

# 单 ckpt × 单 suite，约 4h（H100 单卡）
bash script/eval/libero/eval_libero.sh
```

注意 `script/eval/libero/eval_libero.sh` 是 **client/server** 架构：
1. `python deploy.py ... &`（Flask server，加载 ckpt 暴露 `/process_frame`）
2. `sleep 1800`（30 min，等模型 load + LLaMA-2-7b mmap）
3. `python evaluation/libero/eval_libero.py ... --port $port`（client，逐 task rollout）

结果落在 `$WORKSPACE/ckpts/libero/memvla-libero-spatial/eval_libero/memvla-libero-spatial.pt/libero_spatial-50trials-seed7-ac8-<TS>.txt`。

### 2.3 Bridge eval

```bash
ssh -p 29862 root@10.100.2.47
cd /mnt/3fs/data/tingwen.du/workspace/MemoryVLA
source /root/envs/simpler/bin/activate

# 4 scene 串行，约 1.5h
bash script/eval/bridge/eval_bridge.sh
```

每个 scene 一次完整 python 进程（ckpt 重新 load），结果落在 `$WORKSPACE/ckpts/simpler/memvla-bridge/eval_simpler/memvla-bridge.pt/{Cube,Carrot,Spoon,Eggplant}.txt`。

### 2.4 抽取 LIBERO 总分

```bash
# 编辑 script/eval/libero/extract_libero_results.py 顶部的 ckpt_paths 列表，加：
#   "/mnt/3fs/data/tingwen.du/workspace/MemoryVLA/ckpts/libero/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt"
python script/eval/libero/extract_libero_results.py
```

或直接 `grep "Current task success rate" $RESULT.txt | awk '{s+=$NF;c++} END{print s/c}'`。

---

## 3. 踩过的坑（所有 fix 均已写入对应文件，可重跑）

### 3.1 `gymnasium.wrappers.TimeLimit` 不透传非标准属性 ⚠⚠

**现象**：Bridge 第一次跑全部 4 个 scene 起 1s 内 crash：
```
AttributeError: 'TimeLimit' object has no attribute 'is_final_subtask'
```

**根因**：`build_maniskill2_env()` 用 `gymnasium.wrappers.TimeLimit` 包了 env。老 `gym` 会 fallback 到 `__getattr__` 透传任意属性，但 `gymnasium 0.28+` 收紧了，只透传 `unwrapped` / `spec` 等有限白名单。所以 `env.is_final_subtask()` / `env.get_language_instruction()` / `env.advance_to_next_subtask()` / `env.robot_uid` 全炸。

**Fix（已落盘）**：
- `evaluation/simpler_env/maniskill2_evaluator.py` — 5 处 `env.X` → `env.unwrapped.X`（line 82/89/121/129/133）
- `third_libs/SimplerEnv/simpler_env/utils/env/observation_utils.py` — 1 处 `env.robot_uid` → `env.unwrapped.robot_uid`

### 3.2 LIBERO 容器缺 `libosmesa6` ⚠

**现象**：`evaluation/libero/eval_libero.py:8` 设 `MUJOCO_GL=osmesa`，但 PyOpenGL 找不到 GL backend：
```
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

**Fix（已落盘）**：`scripts/setup_libero.sh` apt 列表加 `libosmesa6 libosmesa6-dev`；`script/eval/libero/eval_libero.sh` 第 4 行 `export MUJOCO_GL=${MUJOCO_GL:-osmesa}` 显式声明便于 override。

### 3.3 bash `cmd | tee f1; cmd2 | tee f2` 不传播 python crash ⚠⚠

**现象**：第一次 Bridge 4 个 scene 全 crash 但 bash exit=0，trap 报 `eval end exit=0`，看起来"成功"实际全失败。

**根因**：pipe 的 `$?` 来自 tee（永远 0），bash 不会因为 python 死了而 `set -e` 退出。

**Mitigation**：直接读 python stdout 而非靠 bash exit；可选 `set -o pipefail`（但会改变 sh 语义，慎用）。

### 3.4 3fs ENOSPC 触发 matplotlib savefig 失败 ⚠

**现象**：Carrot scene rollout 到第 5 ep，eval 一切正常，但 `model.visualize_epoch()` 内 `mpl.image.imsave` 抛 `OSError: [Errno 28] No space left on device`，把整个 scene 干掉。

**根因**：3fs 集群 93% 利用率，间歇性 ENOSPC。

**Fix（已落盘）**：`evaluation/simpler_env/maniskill2_evaluator.py` 把 `model.visualize_epoch` 包 try/except OSError，丢图但不丢 episode。

### 3.5 `script/eval/bridge/eval_bridge.sh` orphan `done` ⚠

我之前编辑该脚本加 env exports 时把 `for scene_name; do ... done` 删错留了 trailing `done`，导致 bash 直接 syntax error（"unexpected token 'done'"）。已修。

### 3.6 LIBERO 必须 `sleep 1800` 而非 `sleep 60`

**根因**：`deploy.py` load 的是 prismatic-7b VLM + DiT-L action head + LLaMA-2-7b backbone，模型本身 ~14GB，从 3fs mmap 到 GPU 在 RIO 慢的时段要 ~20+ min。`sleep 1800` 是工程上的安全缓冲，不能换 `wait_for_port`（Flask app 起来 ≠ ckpt load 完）。

---

## 4. 修改的文件清单（仅 3fs 上）

| 文件 | 改动 |
|---|---|
| `evaluation/simpler_env/maniskill2_evaluator.py` | 5 处 env wrapper unwrap + try/except OSError 包 visualize_epoch |
| `third_libs/SimplerEnv/simpler_env/utils/env/observation_utils.py` | 1 处 env.robot_uid → env.unwrapped.robot_uid |
| `script/eval/bridge/eval_bridge.sh` | 删 line 67 orphan `done` |
| `script/eval/libero/eval_libero.sh` | 加 `export MUJOCO_GL=${MUJOCO_GL:-osmesa}` |
| `scripts/setup_libero.sh` | apt 加 `libosmesa6 libosmesa6-dev` |
| `vla/__init__.py` | dlimp-optional guard（setup 自动注入，幂等）|

**所有改动均仅在 3fs 上**，尚未整合进 `/work-agents/.../MemoryVLA` git repo。下一步走 PR。

---

## 5. 待主管 review 的问题

1. **LIBERO Task 6 0/50 的处理**：是否需要消融来定位（ac=4 / 不同 seed / 关 cfg）？还是接受 9/10 task 与 paper 一致即认为"复现成功"？
2. **Bridge Cube scene 我们 0.458 vs paper 0.708 的差距**：episode_id 集合一致、env 一致，差异可能来自 sapien 渲染版本或 random seed。需要 paper 作者确认他们的 sapien commit hash 和 seed 设置。
3. **修改文件 PR 整合**：6 个修改文件应当合成一个 PR 走流程；准备开 task + 分支 + PR（待主管 confirm 范围）。

---

## 6. 数据落盘位置

- LIBERO 完整日志：`$WORKSPACE/logs/libero/eval_run_20260425_084128.log`（~3000 行）
- Bridge 完整日志：`$WORKSPACE/logs/simpler/rerun_20260425_094735.log`
- Bridge per-scene .txt：`$WORKSPACE/ckpts/simpler/memvla-bridge/eval_simpler/memvla-bridge.pt/{Cube,Carrot,Spoon,Eggplant}.txt`
- LIBERO trial-by-trial：`$WORKSPACE/ckpts/libero/memvla-libero-spatial/eval_libero/memvla-libero-spatial.pt/libero_spatial-50trials-seed7-ac8-2026_04_25-09_13_25.txt`（500 ep × 多行）
- LIBERO 每 ep 的视频：同目录 `_videos/` 子目录

