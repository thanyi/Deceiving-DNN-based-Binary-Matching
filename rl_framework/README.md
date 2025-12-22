# PPO-Based Binary Code Perturbation Framework

基于 PPO 的二进制代码变异强化学习框架

## 架构设计

```
┌───────────────────────────────────────────────────────────────┐
│                  PPO Training Loop (Python 3)                  │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ PPO Agent  │───▶│ Policy Net   │───▶│  Action      │      │
│  │ (Actor)    │    │ (Actor-Critic)│    │  Selection   │      │
│  └────────────┘    └──────────────┘    └──────┬───────┘      │
│         ▲                                      │               │
│         │                                      ▼               │
│  ┌────────────┐                      ┌─────────────┐         │
│  │  Update    │◀─────────────────────│   Reward    │         │
│  │  Policy    │                      │ Computation │         │
│  └────────────┘                      └─────────────┘         │
└────────────────────────┬──────────────────▲───────────────────┘
                         │ Function Call     │ Return Value
                         ▼                   │
┌───────────────────────────────────────────────────────────────┐
│              Environment Wrapper (Python 3)                    │
│  ┌───────────────┐  ┌───────────────────┐  ┌──────────────┐ │
│  │  Apply Action │─▶│  Call Mutation    │─▶│  Evaluate    │ │
│  │               │  │  (subprocess)     │  │  (run_one)   │ │
│  └───────────────┘  └────────┬──────────┘  └──────────────┘ │
│                              │                                │
│                              ▼                                │
│         ┌───────────────────────────────────┐                │
│         │  python2 uroboros_automate.py     │  ◀─ Python 2!  │
│         │  (唯一需要 Python 2 的部分)        │                │
│         └───────────────────────────────────┘                │
└───────────────────────────────────────────────────────────────┘
```

## 文件说明

### 1. `ppo_agent.py` (Python 3)
**PPO 智能体核心实现**

- `PolicyNetwork`: Actor-Critic 神经网络
  - Actor: 输出动作概率分布
  - Critic: 估计状态价值
  
- `PPOAgent`: PPO 算法实现
  - `select_action()`: 根据策略选择动作
  - `update()`: PPO 裁剪目标更新
  - `compute_returns()`: GAE 优势估计
  
- `RewardShaper`: 奖励塑形
  - 成功奖励（score < 0.40）
  - 进步奖励（相比历史最优）
  - 梯度引导奖励
  - 步数惩罚

### 2. `env_wrapper.py` (Python 3)
**环境包装器 - 与现有代码集成**

- `BinaryPerturbationEnv`: 变异环境
  - `apply_mutation()`: 调用 uroboros 执行变异
  - `evaluate()`: 调用 run_one 评估相似度
  - `extract_features()`: 提取状态特征
  - `run_loop()`: 监听 PPO 指令

- 通信机制：
  - `action.json`: PPO → 环境（动作）
  - `result.json`: 环境 → PPO（结果）
  - `status.json`: 环境状态

### 3. `ppo_trainer.py` (Python 3)
**训练主程序**

- `EnvBridge`: 进程间通信桥接
  - 启动 Python 2 环境进程
  - 文件系统通信
  - 超时处理
  
- `train_ppo()`: 训练循环
  - 回合采样
  - 策略更新
  - 模型保存
  - 日志记录

## 使用方法

### 安装依赖

```bash
# Python 3 依赖
pip3 install torch numpy pandas loguru

# Python 2 依赖 (已有)
# 无需额外安装
```

### 训练命令

```bash
cd /home/ycy/ours/Deceiving-DNN-based-Binary-Matching

# 基础训练
python3 rl_framework/ppo_trainer.py \
    --binary workdir_1/pwd \
    --function usage \
    --save-path function_container_usage_pwd \
    --episodes 100 \
    --max-steps 50

# 使用 GPU 加速
python3 rl_framework/ppo_trainer.py \
    --binary workdir_1/pwd \
    --function usage \
    --save-path function_container_usage_pwd \
    --episodes 200 \
    --use-gpu

# 恢复训练
python3 rl_framework/ppo_trainer.py \
    --binary workdir_1/pwd \
    --function usage \
    --save-path function_container_usage_pwd \
    --resume rl_models/ppo_model_best.pt \
    --episodes 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--binary` | 原始二进制文件路径 | 必填 |
| `--function` | 目标函数名 | 必填 |
| `--save-path` | 变异结果保存路径 | 必填 |
| `--state-dim` | 状态特征维度 | 128 |
| `--lr` | 学习率 | 3e-4 |
| `--gamma` | 折扣因子 | 0.99 |
| `--epsilon` | PPO 裁剪参数 | 0.2 |
| `--episodes` | 训练回合数 | 100 |
| `--max-steps` | 每回合最大步数 | 50 |
| `--save-interval` | 保存间隔 | 10 |
| `--use-gpu` | 使用 GPU | False |

---

## 使用训练好的模型

### 推理命令

训练完成后，使用最佳模型对新的二进制文件进行变异：

```bash
# 方式1: 使用快速推理脚本（推荐）
cd /home/ycy/ours/Deceiving-DNN-based-Binary-Matching

./rl_framework/quick_inference.sh workdir_1/pwd usage

# 方式2: 直接调用推理脚本
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/pwd \
    --function usage \
    --save-path inference_pwd_usage \
    --max-steps 30 \
    --target-score 0.40

# 方式3: 批量推理多个二进制文件
# 首先创建批量配置文件 batch.txt:
# workdir_1/ls,usage,output_ls
# workdir_1/pwd,usage,output_pwd

python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --batch \
    --batch-file batch.txt
```

### 推理输出

推理完成后会生成：
- `inference_log.txt` - 详细的推理日志（每步的动作、分数、奖励等）
- `<hash>_container/` - 变异后的二进制文件目录

**详细使用说明请参考：[推理使用指南](INFERENCE_GUIDE.md)**

## 输出文件

```
rl_models/
├── ppo_model_ep10.pt      # 第10回合模型
├── ppo_model_ep20.pt      # 第20回合模型
├── ppo_model_best.pt      # 最佳模型
├── ppo_model_final.pt     # 最终模型
└── training_log.txt       # 训练日志

function_container_usage_pwd/
├── <hash1>_container/     # 变异样本1
├── <hash2>_container/     # 变异样本2
└── success.log            # 成功样本记录
```

## 训练监控

训练日志格式：
```
episode,steps,total_reward,avg_reward,loss
0,15,12.5432,0.8362,0.0234
1,22,15.2341,0.6925,0.0198
...
```

## 性能优化建议

### 1. 特征提取优化
当前 `extract_features()` 使用随机特征，需要实现：
- 使用预训练模型的嵌入层
- 提取指令序列、CFG 特征
- 降维到固定维度

### 2. 并行采样
```python
# 同时运行多个环境实例
envs = [EnvBridge(...) for _ in range(4)]
# 并行采样加速训练
```

### 3. 经验缓存
```python
# 缓存已评估样本，避免重复计算
cache = {}  # (binary_hash, action) -> (score, grad)
```

### 4. 课程学习
```python
# 逐步降低目标阈值
target_scores = [0.8, 0.6, 0.5, 0.4]
```

## 故障排查

### 问题1: 环境启动超时
- 检查 Python 2 路径
- 确认 uroboros 可执行
- 查看 stderr 输出

### 问题2: 变异失败
- 检查二进制文件权限
- 确认 save-path 可写
- 查看 uroboros 日志

### 问题3: 评估返回 None
- 检查模型文件路径
- 确认 checkdict 正确
- 查看 run_one 日志

## 与遗传算法对比

| 特性 | 遗传算法 | PPO |
|------|---------|-----|
| 样本效率 | 低 | 高 |
| 探索能力 | 中等 | 强 |
| 可解释性 | 高 | 中等 |
| 实现复杂度 | 低 | 中等 |
| 收敛速度 | 慢 | 快 |
| 局部最优 | 易陷入 | 不易 |

## 下一步计划

- [ ] 实现真实的特征提取
- [ ] 添加并行采样支持
- [ ] 实现经验缓存机制
- [ ] 集成 TensorBoard 监控
- [ ] 实现层次化 RL (HRL)
- [ ] 添加元学习支持

