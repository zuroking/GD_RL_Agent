# GD RL Agent — 安装与使用指南

一个通过屏幕捕获和键盘输入注入来学习玩 Geometry Dash（Steam 版）的强化学习智能体。无需 RL 框架——全部基于 PyTorch 从零实现。

## 系统要求

- Windows 10/11（pydirectinput 需要 Windows）
- Python 3.12 或更高版本
- Geometry Dash（Steam 版本 2.2）
- 无需 GPU，仅 CPU 即可运行

## 安装

```bash
git clone <repo-url>
cd gd-rl-agent
pip install -e .
```

## 第一步 — 校准

以**窗口模式**启动 Geometry Dash，在**练习模式（practice mode）**中打开 Stereo Madness 关卡。

```bash
python cli.py calibrate
```

该命令将：
1. 自动找到 GD 窗口
2. 测量截图帧率和输入延迟
3. 截取一张屏幕截图——打开 `calibration_frame.png`，按提示输入像素坐标
4. 将配置保存到 `configs/agent_config.json`

**需要输入的值：**
- `player_right_x` — 方块精灵右边缘的 X 像素坐标
- `ground_y` — 地面砖块顶部的 Y 像素坐标
- `cube_height_px` — 方块的像素高度（在截图中测量）

## 第二步 — 验证视觉管道

```bash
python cli.py debug --duration 10
```

观察输出的特征向量，并检查 `debug_frame.png`。绿色矩形是扫描 ROI 区域，红色竖线标记检测到的障碍物，橙色标记坑洞。如果检测不正常，需要在 `vision/obstacle_detector.py` 中通过实验调整 HSV 阈值。

## 第三步 — 训练

在 GD 设置中启用 **Auto-Retry**。在练习模式下启动 Stereo Madness，并在关卡最开始处放置检查点（按 `Z` 键）。然后运行：

```bash
# 标准 DQN
python cli.py train --episodes 500 --run-name vanilla

# Double DQN
python cli.py train --episodes 500 --double --run-name double

# Dueling DQN
python cli.py train --episodes 500 --dueling --run-name dueling
```

训练日志保存在 `runs/<run-name>/train_log.csv`，每 50 个 episode 保存一次检查点。

**重要提示：** 第一次训练后，检查日志中的实际帧率，如果与校准时的值差异较大，请在 `configs/agent_config.json` 中重新校准 `max_episode_steps`。

## 第四步 — 观察与评估

```bash
# 观察智能体游玩（不更新权重）
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 5

# 评估平均奖励和成功率
python cli.py eval --checkpoint runs/vanilla/ckpt_ep0500.pt --episodes 20
```

## 配置参数说明

所有设置保存在 `configs/agent_config.json`（由 `calibrate` 命令创建）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `episode.max_episode_steps` | 250 | 每轮步数——需要在 DQN 运行时重新校准 |
| `episode.auto_retry` | true | 若 GD 已启用 Auto-Retry，设为 true |
| `reward.r_step` | 0.01 | 每存活一步的奖励 |
| `reward.r_death` | -1.0 | 死亡惩罚 |
| `reward.r_completion` | 5.0 | 到达片段终点的奖励 |
| `reward.penalize_useless_jumps` | true | 惩罚前方无障碍时的跳跃 |
| `reward.useless_jump_threshold` | 0.35 | "无用跳跃"的距离阈值 |
| `hp.use_double_dqn` | false | 启用 Double DQN |
| `hp.use_dueling_dqn` | false | 启用 Dueling DQN |
| `hp.use_obstacle_lookahead` | false | 在特征向量中添加第二个障碍物 |

## 项目结构

```
gd-rl-agent/
├── capture/          # mss 屏幕捕获，窗口检测
├── input/            # 通过 pydirectinput 注入跳跃
├── vision/           # HSV 障碍物/坑洞/死亡检测，状态提取器
├── env/              # 自定义环境循环（reset/step/close）
├── agent/            # QNetwork、ReplayBuffer、Trainer
├── configs/          # Pydantic 配置模型
├── cli.py            # Typer CLI（calibrate/debug/train/watch/eval）
└── tests/            # agent/ 的单元测试（无需真实游戏）
```

## 运行测试

```bash
python -m pytest tests/ -v
```
