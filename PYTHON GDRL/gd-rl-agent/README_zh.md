# GD RL Agent

一个通过屏幕捕获和键盘输入注入来学习玩 Geometry Dash（Steam 版）的强化学习智能体。教学项目——无需 RL 框架，全部基于 PyTorch 从零实现。

## 核心约束

- 仅 CPU（Intel i7-1225U，无需 GPU）
- 实时环境——无法加速模拟，无法并行化
- 二元动作空间：跳跃 / 不跳跃（仅方块模式）
- 在 Stereo Madness 0–29% 训练（方块段，无传送门）

## 快速开始

```bash
pip install -e .
python cli.py calibrate        # 查找窗口、测量延迟、保存配置
python cli.py debug            # 可视化验证视觉管道
python cli.py train --episodes 500 --run-name vanilla
python cli.py watch --checkpoint runs/vanilla/ckpt_ep0500.pt
```

## DQN 变体

```bash
python cli.py train --no-double --no-dueling --run-name vanilla
python cli.py train --double    --no-dueling --run-name double
python cli.py train --no-double --dueling    --run-name dueling
```

## 项目结构

```
gd-rl-agent/
├── capture/       # mss 屏幕捕获，窗口检测
├── input/         # pydirectinput 跳跃注入，延迟测量
├── vision/        # HSV 障碍物/坑洞/死亡检测，状态提取器
├── env/           # 自定义环境循环：reset() / step() / close()
├── agent/         # QNetwork、ReplayBuffer、Trainer
├── configs/       # Pydantic v2 配置模型
├── cli.py         # Typer CLI：calibrate / debug / train / watch / eval
└── tests/         # agent/ 的单元测试（无需真实游戏）
```

## 奖励公式

```
r_t = +0.01   每存活一步
    + (-1.0)  死亡
    + (+5.0)  完成片段
    + (-0.02) 无用跳跃（已落地，前方无障碍物）
γ = 0.99
```

## 特征向量（4 个浮点数）

| 特征 | 范围 | 哨兵值 |
|---|---|---|
| `dist_obstacle` | [0, 1] | 1.0 |
| `dist_pit` | [0, 1] | 1.0 |
| `obstacle_height` | [0, 1] | 0.0 |
| `is_grounded` | {0, 1} | — |

## 运行测试

```bash
python -m pytest tests/ -v
```

完整安装与校准指南见 `INSTRUCTION_zh.md`。
