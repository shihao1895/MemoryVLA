<!-- METADATA:STATUS=InProgress,ASSIGNEE=intern_memvla_developer -->

# task001 — Simpler + LIBERO eval bringup

## 背景

主管 directive（2026-04-24）：复现 MemoryVLA 两个官方 eval（SimplerEnv/Bridge + LIBERO-Spatial）并把经验沉到 3fs 便于复现。

经 Session 1-3 发现 repo 现状有多处 eval 跑不通的 bug（TimeLimit wrapper、osmesa 缺、shell 语法错、3fs ENOSPC 未处理、dlimp 缺），经在 3fs 上逐一 fix 后两个 eval 都完整跑通并产出 baseline 数字。

本任务把 3fs 上的 6 个修改整合回 `/work-agents` repo 并打 PR，确保"从 clean repo + 本 PR 即可复现"。

## 目标

1. 把 3fs 上 Session 1-3 的全部修复整合成 PR
2. 带上 `docs/eval_reproduction.md`（完整复现手册 + 两个 eval 结果 vs paper 对比）
3. 带上 `scripts/setup_libero.sh`（LIBERO 容器一键环境安装脚本，幂等）
4. 带上 SimplerEnv 的 `observation_utils.py` patch（作为 `scripts/patches/simpler_env_observation_utils.patch`，因 third_libs 被 gitignore 屏蔽）
5. 顺手 fix `.gitignore:107` 的 `env/` 模式 bug（误伤 `third_libs/SimplerEnv/.../env/`）

## 改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `evaluation/simpler_env/maniskill2_evaluator.py` | modify | 5 处 `env.X()` → `env.unwrapped.X()`（gymnasium 0.28+ wrapper 不透传）+ 1 处 try/except OSError 包 visualize_epoch（3fs ENOSPC 保护） |
| `script/eval/bridge/eval_bridge.sh` | modify | 删 orphan `done` |
| `script/eval/libero/eval_libero.sh` | modify | 加 `export MUJOCO_GL=${MUJOCO_GL:-osmesa}` |
| `vla/__init__.py` | modify | `materialize.py` 导入包 try/except（让 `load_vla` 在无 dlimp 的 eval 环境下可用） |
| `.gitignore` | modify | `env/` → `/env/`（只匹配 repo root 的 venv，不再误伤 third_libs） |
| `scripts/setup_libero.sh` | new | LIBERO 容器一键环境安装脚本 |
| `scripts/patches/simpler_env_observation_utils.patch` | new | 对外部 `third_libs/SimplerEnv/.../observation_utils.py` 的 1 行 fix |
| `docs/eval_reproduction.md` | new | 复现手册 + 结果 + 踩坑记录 |

## 验收标准

- [ ] 8 个文件改动已落盘到 feature 分支 `intern_memvla_developer/task001_simpler_libero_eval_bringup`
- [ ] PR 已开、base `openvla-codebase`
- [ ] PR 通过 review / merge
- [ ] merge 后从 clean clone + `bash scripts/setup_libero.sh` 可跑通 LIBERO eval（下次复现时验证）

## 结果基线（来自 docs/eval_reproduction.md）

- Bridge avg = 0.7292 vs paper 0.7188 ✅
- LIBERO-Spatial = 0.886（443/500）vs paper 0.984；排除 Task 6 outlier 后 9-task avg = 0.984 = paper 一致

## 负责人

intern_memvla_developer

## 基础分支

`openvla-codebase`
