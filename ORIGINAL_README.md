> Historical source documentation. See the repository root README and PUBLICATION_NOTES.md for current distribution limits.

# 双独立 8DoF KAN-Transformer 监督式闭环轨迹生成

> 当前递送实验入口（2026-09-10）：`scripts/run_smooth_delivery.ps1`；固定端点为 `results/smooth_kan_v1/payload_delivery_tasks_local.json`。采用冻结 KAN-Transformer + 几何局部修复 + C3 平滑执行，右臂固定为零。场景仅修正了两个待递送零件的位置。参见 `results/smooth_kan_v1/EXPERIMENT_PLAN.md` 和 `results/smooth_kan_v1/DELIVERY_REPORT.md`。以下原始流水线说明保留为历史记录，不代表当前全部实验状态。

本项目在原始 `finally.usd` 上完成了一条可运行、可复现的 **supervised imitation learning** 流水线。左右机械臂各自拥有独立的 8D policy、KAN tokenizer、Transformer、optimizer、checkpoint 和 action-history FIFO；仅在同一个 PhysX 场景中同步采集与执行。工程未使用 PPO/SAC/TD3/DDPG，也未修改或保存原始 USD。

## 可执行 Workflow

固定环境：

```powershell
$py = 'D:\conda_envs\env_isaaclab\python.exe'
Set-Location 'D:\work_space\isaacsim_lab_zbot_6s_deepseek\v6.3'
```

按验证顺序运行：

```powershell
& $py scripts/environment_report.py
& $py scripts/inspect_scene.py --headless
& $py scripts/test_coordinates.py --headless
& $py scripts/test_robot_control.py --headless
& $py scripts/test_collisions.py --headless
& $py -m pytest tests -q

& $py scripts/generate_stage1_data.py --headless --episodes 3 --seed 43
& $py scripts/generate_tiny_data.py --headless
& $py scripts/tiny_overfit.py
& $py scripts/closed_loop_smoke.py --headless

& $py scripts/train_parallel_h64.py --epochs 30
& $py scripts/generate_stage2_data.py --headless --episodes 1 --seed 43
& $py scripts/generate_stage2_batch.py --headless --seed-start 1000 --seed-count 100
& $py scripts/generate_collision_counterfactual.py --headless --positives-per-kind 8
& $py scripts/train_stage2_h64.py --epochs 15
& $py scripts/collect_dagger_randomized.py --headless --tasks 20 --steps-per-task 32
& $py scripts/run_dagger_h64.py --iteration 5 --rollout-steps 180 --epochs 30
& $py scripts/train_h128.py --epochs 20
& $py scripts/evaluate.py --headless --max-steps 180
```

TensorBoard：

```powershell
& $py -m tensorboard.main --logdir logs/tensorboard
```

`scripts/train_left_h64.py` 与 `scripts/train_right_h64.py` 可单独训练；主实验使用 `train_parallel_h64.py`，但仍执行两个独立 backward/optimizer step，绝不拼成 16D joint policy。

## Observation / State / Action / Timing

每侧 observation 由 8 个 semantic task tokens、1 个对侧几何 token、`H` 个本侧历史 action tokens 和 1 个 `CONTROL_QUERY` 构成：

- `q_start/q_current/q_goal [B,8]`，KAN 前采用 sin/cos 后为 `[B,16]`。
- `delta_q_goal [B,8]` 使用 periodic shortest-angle error。
- `x_start/x_current/x_goal [B,9]` = position `[3]` + rotation-6D `[6]`，全部在本侧真实 `a1` Base Frame。
- `delta_x_goal [B,6]` = position error `[3]` + SO(3) log error `[3]`。
- `other_arm_geometry [B,9,3]`，为对侧 `a1..a8,b8` 在本侧 Base Frame 中的位置。
- `history_action [B,H,8]` 与 `history_mask [B,H]`；H=64 或 128。在当前运动学模式中保存本侧周期关节增量 `periodic_delta(q_t, q_{t-1})`。
- action 为 `Delta q [B,8]`，逐关节以 `5 deg/control-step` 的 `tanh` 硬限幅。

系统默认使用运动学状态转移：`q_next = wrap_to_pi(q_current + clip(delta_q))`。PhysX 的 `physics_dt=1/120 s` 仅用于刷新 FK 和执行 self/inter-arm contact query；控制频率保持 30 Hz，不再使用 PD、惯性或 actuator tracking 产生状态。每周期重新读取 q/TCP/goal/对侧几何/history，模型只输出下一步 action。`GoalManager` 支持实时 goal 替换；大跳变时 `start <- current` 并清空 FIFO，同时产生 `goal_reset_event`。

## 模型与 Loss

KAN 使用可训练 cubic B-spline edge functions，不是把普通 MLP 重命名为 KAN。Transformer 内部 FFN 保持标准结构。默认目标：

```text
L = L_SmoothL1(action)
  + 1e-3 L_velocity
  + 1e-4 L_acceleration
  + 1e-5 L_jerk
  + 0.05 L_self_BCE
  + 0.05 L_inter_BCE
```

PhysX boolean contact 从不参与 `backward()`；碰撞通过安全 expert、DAgger 和 weighted BCE auxiliary head 学习。Collision Head 显式接收 `candidate_action`：expert 数据查询 expert action，DAgger 查询实际 policy action，counterfactual 数据查询跨越 PhysX 碰撞边界的 action。left/right self 与 inter-arm 均具有正负配对样本，避免用虚高 accuracy 代替 precision/recall/F1/AUROC。`DifferentiableCollisionCost` 仅预留接口，在没有真实 SDF/distance gradient 前会主动报错。

## Scene Mapping 与碰撞后端

- 全系统实际为 `/World/unitree_h1/pelvis` 下的单一 35-DOF PhysX articulation；左右 8DoF 通过完整 DOF path 动态解析。
- 左臂：`/World/Base_01/zbot_8s_new_01`；右臂：`/World/Base_02/zbot_8s_new_02`。
- Base Frame：各自的 `.../a1` rigid-body pose。右安装节点存在 determinant=-1 reflection，不能直接转换为 quaternion，故不作为模型 base。
- TCP：左侧 `/World/cixi_01` rigid-body origin 已按半球最小 extent 轴的小面积端帽重定位到 physical contact apex；live `b8 → TCP=0.111308 m`。右侧 `/World/cixi_00` 保持不变，后续将删除。标定与迁移证据见 `LEFT_TCP_COORDINATE_FIX_REPORT.md`。
- 权威碰撞源为 pair-filtered PhysX tensor contact view，输出 collider pair、contact point、impulse 与 `left_self/right_self/inter_arm` 分类。
- Stage-2 teacher 为本地实现的 periodic bidirectional RRT-Connect；它在 coupled 16D configuration space 做安全搜索，然后拆分为左右独立 8D expert labels。cuRobo/OMPL 未安装，Lula 因缺少该机器人的 robot-description 配置未被冒充为可用后端。

## 数据、Checkpoint 与原始结果

```text
datasets/{left,right}/        Stage-1、Stage-2、DAgger，左右标签始终 [N,8]
datasets/tiny/{left,right}/   debug-only tiny overfit 数据，不混入正式随机分布
checkpoints/{left,right}/     bc_h64、stage2_h64、dagger_h64、h128
logs/tensorboard/             左右独立 TensorBoard events
results/                      JSON、原始 CSV、自动生成图表
```

关键实测结论见 `results/progress.md`。最终 H64/H128 comparison 保存于 `results/evaluation/`，图表由这些 CSV 自动生成到 `results/plots/`。tiny closed-loop gate 已 dual-success；最终 10 个 frozen hold-out tasks 上 H64/H128 均达到 70% dual-success，但右臂仍发生少量 self-collision safety-stop，因此不能视为可部署模型。

## Core Assumptions

- USD 没有可信的有限 joint limits/velocity/force/gains；实验以 `[-pi,pi]` 为周期采样域，并在 session runtime 写入保守 actuator limits/gains。
- 任务是 proprioceptive joint/TCP control，不包含 camera observation；因此 headless run 未创建 RGB/depth/segmentation render product，Omniverse synthetic recorder 保持关闭。状态轨迹使用原始 CSV，训练使用 TensorBoard。
- 正式 Stage-1 endpoint sampler 是 joint-space uniform `U(-pi,pi)^8`，不是 Cartesian-workspace uniform。
- 当前正式数据为 9562 samples/side、379 episode groups；左侧全部 Cartesian 字段已迁移到 physical contact TCP，关节、动作、历史和碰撞标签保持不变。
- 数据划分以完整 `source-file + seed + episode` group 为单位执行 70/15/15 train/validation/test；同一 trajectory 的相邻 timestep 不会跨 split。DAgger 只查询 train-split tasks，并强制写入 train split。
- normalization 只在 train split 拟合且左右独立；joint angles 保持周期 sin/cos，position、goal errors、other-arm geometry 与有效 history actions 使用各侧统计量。统计量随各侧 checkpoint config 保存。
- 本机 IsaacLab 2.3.2 源码已确认不存在旧版 `RLTask`（现有的是 `DirectRLEnv/ManagerBasedRLEnv`）。本工程是单场景 supervised data/BC/DAgger，而非 RL，因此采用原生 `SimulationContext + Articulation`；没有为满足旧 API 名称而引入伪 RL loop。
- 本机只有一张 NVIDIA GPU，实际运行采用 `cuda:0`；未伪造 multi-GPU benchmark。扩展到多 GPU 时应使用每侧独立 DDP process group/checkpoint，并保持左右模型无参数共享。

## Risk Points（Sim-to-Real 边界）

- action 已限幅 5°/step，但真机还必须增加硬件 joint/torque/velocity limit、watchdog 和独立物理 E-stop；软件 stop 不能替代安全回路。
- runtime 在 forbidden self/inter contact 或 `|qd|>6 rad/s` 时触发 safety stop；真机必须在接触前用距离阈值停止，不能依赖碰撞后检测。
- USD 原始关节参数为无限/零增益，当前 gains 仅供仿真；未经系统辨识不可直接映射到电机。
- 当前为运动学仿真，不建模质量、惯性、摩擦、驱动跟踪误差、结构柔顺性，也未建模 sim-real transport jitter、packet loss 与 actuator latency。真机前必须另建动力学验证层，注入实测 delay/noise distribution，并执行限幅、碰撞预测与 worst-case watchdog test。
- 当前 sensor noise=0，domain randomization 与多 GPU 大规模训练尚未执行；这限制 sim-to-real 覆盖范围。
- TCP 是根据现有 tool rigid body 选定；装配偏差、柔顺性、线缆与未建模附件会改变 collision geometry。
- H64/H128 在 10-task hold-out 上最终 dual-success=70%，但右臂 self-collision rate 仍约 0.8%，严禁直接进入硬件验证。必须先把所有独立 safety seeds 降至零碰撞，再通过 shadow-mode 与低速笼内测试。

## Verification Pipeline

完成门槛依次为：环境版本 → scene/reference/DOF/TCP/base audit → coordinate identity/determinant tests → 左/右/同时运动学控制 → 实际制造三类碰撞 → 25 unit tests → tiny overfit → kinematic closed-loop smoke with PhysX contacts → Stage-1 BC → Stage-2 RRT expert → episode split/normalization → collision counterfactual → randomized DAgger → H128 warm start → frozen hold-out raw-CSV comparison。任何前置 gate 失败时，不应继续下游训练。
