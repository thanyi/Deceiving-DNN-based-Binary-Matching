# 强化学习网络改进说明

## 问题诊断

### 训练日志分析
从 `training_log.txt` 发现的严重问题：

| 问题 | 症状 | 严重性 |
|------|------|--------|
| **Loss 爆炸** | 从 260 → 2478 | ⚠️ 危急 |
| **奖励崩溃** | 从 +157 → -120 | ⚠️ 危急 |
| **策略退化** | 频繁负奖励 (-50, -70, -90) | ⚠️ 高 |
| **训练不稳定** | Loss 持续 > 800 | ⚠️ 高 |

### 根本原因

#### 1. 奖励尺度过大
```python
# 旧版本
reward = (1.0 - score) * 10.0       # [0, 10]
reward += 50.0                      # 成功奖励
reward += improvement * 100.0       # 进步奖励 [0, 60] ← 过大!
```
**问题**: 奖励范围 [-10, 100]，波动剧烈，导致价值函数估计不稳定

#### 2. 网络结构简单
```python
# 旧版本: 只有 2 层
nn.Linear(state_dim, 256)
nn.ReLU()
nn.Linear(256, 256)
nn.ReLU()
nn.Linear(256, action_dim)
```
**问题**: 容量不足，无归一化层，容易过拟合

#### 3. Actor-Critic 耦合
```python
# 旧版本: 共享优化器
self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
```
**问题**: Actor 和 Critic 相互干扰，学习率过高

#### 4. 学习率偏高
- 3e-4 在奖励不稳定时过高
- 导致策略震荡和崩溃

---

## 改进方案

### 1. 网络结构升级

#### 改进前
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
```

#### 改进后
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),        # ✓ 归一化
            nn.ReLU(),
            nn.Dropout(0.1),                  # ✓ 防过拟合
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # ✓ 额外层
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
```

**改进点**:
- ✅ 从 2 层增加到 4 层（提高容量）
- ✅ 添加 LayerNorm（稳定训练）
- ✅ 添加 Dropout（防止过拟合）
- ✅ Critic 独立架构（解耦）

---

### 2. 分离 Actor-Critic 优化器

#### 改进前
```python
self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

# 更新
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

#### 改进后
```python
# 分离优化器
self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=1e-4)
self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=2e-4)

# 分别更新
actor_loss = -torch.min(surr1, surr2).mean() - 0.02 * entropy
self.actor_optimizer.zero_grad()
actor_loss.backward(retain_graph=True)
self.actor_optimizer.step()

critic_loss = nn.functional.smooth_l1_loss(state_values.squeeze(), returns)
self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
```

**改进点**:
- ✅ 学习率: 3e-4 → 1e-4 (Actor)
- ✅ Critic 学习率稍高 (2e-4)
- ✅ 独立更新，互不干扰
- ✅ 使用 Huber Loss (smooth_l1_loss) 更鲁棒

---

### 3. 奖励塑形优化

#### 改进前
```python
reward = (1.0 - score) * 10.0           # [0, 10]
reward += 50.0                          # 成功: +50
reward += improvement * 100.0           # 进步: +[0, 60]
reward -= step_count * 0.05             # 步数: -[0, 1]
reward = max(-10.0, min(reward, 100.0)) # 裁剪: [-10, 100]
```
**范围**: [-10, 100]，波动 110

#### 改进后
```python
reward = -np.log(score + 0.01) * 2.0    # 对数缩放 [0, 6]
reward += 10.0 if done else 0           # 成功: +10
reward += improvement * 20.0            # 进步: +[0, 12]
reward -= 0.5 if no improvement         # 未进步: -0.5
reward -= step_count * 0.02             # 步数: -[0, 0.4]
reward = np.clip(reward, -5.0, 10.0)    # 裁剪: [-5, 10]
```
**范围**: [-5, 10]，波动 15

**改进点**:
- ✅ 使用对数缩放（更平滑）
- ✅ 成功奖励: 50 → 10
- ✅ 进步奖励系数: 100 → 20
- ✅ 奖励范围: [-10, 100] → [-5, 10]
- ✅ 波动幅度降低 86%

---

### 4. 训练稳定性增强

| 改进项 | 旧版本 | 新版本 | 效果 |
|--------|--------|--------|------|
| **梯度裁剪** | 0.5 | 0.5 | 保持 |
| **熵正则化** | 0.01 | 0.02 | 增强探索 |
| **Critic Loss** | MSE | Huber Loss | 更鲁棒 |
| **优势函数裁剪** | [-10, 10] | [-10, 10] | 保持 |
| **Log Ratio 裁剪** | [-20, 20] | [-20, 20] | 保持 |

---

## 预期效果

### 训练稳定性
| 指标 | 改进前 | 改进后（预期） |
|------|--------|----------------|
| **Loss 范围** | 260 ~ 2478 | 100 ~ 500 |
| **奖励波动** | -120 ~ +157 | -5 ~ +10 |
| **负奖励频率** | 60% | < 20% |
| **训练收敛** | 不收敛 | 稳定收敛 |

### Value 预期趋势
```
改进前:
Episode:  0  5  10  15  20
Value:   8.0 → 5.0 → 2.0 → -1.0 → -3.0  ❌ 持续下降

改进后:
Episode:  0  5  10  15  20  25  30
Value:   3.0 → 4.0 → 5.0 → 5.5 → 6.0 → 6.2 → 6.5  ✅ 稳定上升
```

---

## 使用方法

### 1. 备份旧模型
```bash
mv rl_models/ppo_model_*.pt rl_models/backup/
```

### 2. 重新训练
```bash
cd rl_framework
python3 ppo_trainer.py \
    --binary ../bin_bk/pwd \
    --function usage \
    --save-path ../function_container_usage_pwd \
    --episodes 50 \
    --max-steps 20 \
    --lr 1e-4
```

### 3. 实时监控
```bash
# 终端 1: 训练
python3 ppo_trainer.py ...

# 终端 2: TensorBoard
tensorboard --logdir=../rl_models/tensorboard
```

### 4. 关键指标观察
- `Episode/Policy_Loss`: 应该 < 500 且逐渐下降
- `Episode/Total_Reward`: 应该逐渐上升
- `Step/Value`: 应该稳定在正值
- `Episode/Final_Score`: 应该逐渐降低（目标 < 0.40）

---

## 超参数调优建议

如果训练仍不稳定，可以尝试：

### 进一步降低学习率
```bash
python3 ppo_trainer.py --lr 5e-5 ...
```

### 减少 PPO 更新轮数
```python
# ppo_agent.py
def __init__(self, ..., epochs=5):  # 从 10 改为 5
```

### 增加熵系数（加强探索）
```python
# ppo_agent.py, update() 方法
actor_loss = actor_loss - 0.05 * entropy  # 从 0.02 改为 0.05
```

### 调整 epsilon
```bash
python3 ppo_trainer.py --epsilon 0.1 ...  # 从 0.2 改为 0.1（更保守）
```

---

## 故障排除

### Loss 仍然很高 (> 800)
- 降低学习率到 5e-5
- 检查奖励是否过大（打印 shaped_reward）
- 增加梯度裁剪阈值到 0.3

### Value 仍然下降
- 检查相似度分数（Final_Score）是否在改善
- 如果 Final_Score 也在变差，说明策略确实退化
- 尝试从头开始训练（不加载旧模型）

### 训练过慢
- 检查二进制变异是否成功（是否崩溃）
- 减少 max_steps（如从 20 改为 15）
- 使用 GPU 加速（--use-gpu）

---

## 总结

### 核心改进
1. ✅ **网络结构**: 2 层 → 4 层 + LayerNorm + Dropout
2. ✅ **优化器**: 共享 → 分离 Actor-Critic
3. ✅ **学习率**: 3e-4 → 1e-4
4. ✅ **奖励尺度**: [-10, 100] → [-5, 10]
5. ✅ **Critic Loss**: MSE → Huber Loss

### 预期结果
- Loss 稳定在 100-500
- 奖励逐渐上升
- Value 保持正值并缓慢上升
- 相似度分数（Final_Score）逐渐降低

### 训练建议
- 使用 TensorBoard 实时监控
- 关注前 10 个回合，判断趋势
- 如果仍不稳定，进一步降低学习率
- 耐心等待，至少训练 30 个回合

