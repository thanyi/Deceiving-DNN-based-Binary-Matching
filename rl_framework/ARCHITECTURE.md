# PPO 框架架构详解

## 一、整体架构

### 1.1 核心思想
将 **PPO 训练循环 (Python 3)** 与 **变异执行环境 (Python 2)** 分离，通过文件系统进行进程间通信。

### 1.2 设计原则
- **职责分离**: PPO 负责决策，环境负责执行
- **语言隔离**: Python 3 和 Python 2 各自独立
- **松耦合**: 通过 JSON 文件通信，易于调试和扩展
- **容错性**: 超时机制、错误处理、状态检查

---

## 二、模块详解

### 2.1 PPO Agent (`ppo_agent.py`)

#### 核心组件

**PolicyNetwork** - Actor-Critic 网络
```python
输入: 状态向量 (state_dim维)
输出: 
  - Actor: 动作概率分布 (8维)
  - Critic: 状态价值 (1维)

网络结构:
  Actor:  state → FC(256) → ReLU → FC(256) → ReLU → FC(8) → Softmax
  Critic: state → FC(256) → ReLU → FC(256) → ReLU → FC(1)
```

**PPOAgent** - 智能体主体
```python
关键方法:
1. select_action(state, explore=True)
   - 输入状态，输出动作、对数概率、状态价值
   - explore=True: 采样模式 (训练)
   - explore=False: 贪婪模式 (测试)

2. store_transition(state, action, reward, log_prob, value)
   - 存储单步经验到缓冲区

3. compute_returns(next_value=0)
   - 使用 GAE 计算优势函数
   - Lambda = 0.95, Gamma = 0.99

4. update()
   - PPO 裁剪目标优化
   - 多轮更新 (epochs=10)
   - 梯度裁剪防止爆炸
```

**RewardShaper** - 奖励塑形
```python
奖励组成:
  reward = -score                    # 基础奖励 (越低越好)
         + 10.0 * (success)          # 成功奖励 (score < 0.40)
         + 5.0 * (improvement)       # 进步奖励 (相比历史最优)
         + 0.1 * |grad|              # 梯度引导
         - 0.01 * step_count         # 步数惩罚
```

---

### 2.2 环境包装器 (`env_wrapper.py`)

#### 核心组件

**BinaryPerturbationEnv** - 变异环境
```python
状态表示:
  state = extract_features(binary_file)
  维度: 128 (可配置)
  内容: 二进制文件的特征向量

动作空间:
  actions = [1, 2, 3, 5, 7, 8, 9, 11]
  对应变异策略:
    1: bb_reorder       - 基本块重排序
    2: bb_split         - 基本块分裂
    3: instr_replace    - 指令替换
    5: instr_garbage    - 垃圾指令插入
    7: bb_opaque        - 不透明谓词
    8: bb_flatten       - 控制流平坦化
    9: func_reorder     - 函数重排序
    11: bb_branchfunc   - 分支函数化

奖励信号:
  reward = -score (相似度分数的负值)
  目标: 最小化 score，使其 < 0.40
```

#### 关键方法

**apply_mutation** - 执行变异
```python
流程:
1. 调用 uroboros_automate-func-name.py
2. 传入参数: seed_binary, action, iteration
3. 生成变异二进制文件
4. 计算 MD5 hash
5. 移动到 container 目录
6. 返回变异文件路径
```

**evaluate** - 评估相似度
```python
流程:
1. 调用 run_one(original, mutated, model, checkdict, function)
2. 模型计算特征相似度
3. 返回 (score, grad)
   - score: 相似度分数 [0, 1]
   - grad: 梯度值，指导优化方向
```

**run_loop** - 监听循环
```python
伪代码:
while True:
    if action_file exists:
        command = read_json(action_file)
        
        if command == 'reset':
            state = reset_env()
        elif command == 'step':
            state, reward, done, info = step(action)
        elif command == 'exit':
            break
        
        write_json(result_file, result)
        remove(action_file)
    
    sleep(0.1)
```

---

### 2.3 训练器 (`ppo_trainer.py`)

#### 核心组件

**EnvBridge** - 进程间通信桥
```python
职责:
1. 启动 Python 2 环境进程
2. 管理通信文件
3. 发送命令、等待结果
4. 超时处理和错误恢复

通信协议:
  action.json: {"command": "step", "action": 5}
  result.json: {"state": [...], "reward": -0.5, "done": false, "info": {...}}
  status.json: {"status": "ready", "timestamp": 1234567890}
```

**train_ppo** - 训练主循环
```python
伪代码:
for episode in range(max_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # 1. 选择动作
        action, log_prob, value = agent.select_action(state)
        
        # 2. 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 3. 奖励塑形
        shaped_reward = reward_shaper.compute_reward(...)
        
        # 4. 存储经验
        agent.store_transition(state, action, shaped_reward, log_prob, value)
        
        # 5. 更新状态
        state = next_state
        
        if done:
            break
    
    # 6. PPO 更新
    loss = agent.update()
    
    # 7. 保存模型
    if episode % save_interval == 0:
        agent.save(model_path)
```

---

## 三、通信机制详解

### 3.1 文件系统通信

**为什么选择文件系统？**
- ✅ 跨语言兼容 (Python 2/3)
- ✅ 易于调试 (可直接查看 JSON)
- ✅ 容错性好 (进程崩溃不影响数据)
- ✅ 实现简单 (无需额外依赖)

**通信流程**
```
PPO (Python 3)                   Environment (Python 2)
     |                                   |
     | 1. write action.json              |
     |---------------------------------->|
     |                                   | 2. read action.json
     |                                   | 3. apply mutation
     |                                   | 4. evaluate
     |                                   | 5. write result.json
     |<----------------------------------|
     | 6. read result.json               |
     | 7. delete files                   |
```

### 3.2 超时与错误处理

**超时机制**
```python
# 环境启动超时: 30秒
# 单步执行超时: 300秒 (5分钟)
# 原因: 变异+编译+评估可能较慢

def _wait_for_result(timeout=300):
    waited = 0
    while waited < timeout:
        if result_file exists:
            return read_json(result_file)
        sleep(0.1)
        waited += 0.1
    raise TimeoutError
```

**错误恢复**
```python
try:
    state, reward, done, info = env.step(action)
except Exception as e:
    # 返回默认惩罚值
    state = default_state
    reward = -10.0
    done = False
    info = {'error': str(e)}
```

---

## 四、关键算法

### 4.1 PPO 裁剪目标

```python
# 重要性采样比率
ratio = exp(new_log_prob - old_log_prob)

# 裁剪目标 (防止更新过大)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
actor_loss = -min(surr1, surr2)

# Critic 损失
critic_loss = (return - value)^2

# 总损失 (加入熵正则化鼓励探索)
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

**为什么使用裁剪？**
- 防止策略更新过大导致性能崩溃
- 保证训练稳定性
- ε=0.2 是经验值 (来自 OpenAI 论文)

### 4.2 GAE (广义优势估计)

```python
# 时序差分误差
delta_t = r_t + γ*V(s_{t+1}) - V(s_t)

# GAE 递推
A_t = delta_t + (γ*λ) * A_{t+1}

# 回报
return_t = A_t + V(s_t)
```

**参数选择**
- γ = 0.99 (折扣因子，重视长期奖励)
- λ = 0.95 (GAE 参数，平衡偏差与方差)

---

## 五、训练流程图

```
开始训练
    │
    ├─> 初始化
    │   ├─ 加载模型 (可选)
    │   ├─ 启动 Python 2 环境
    │   └─ 创建保存目录
    │
    ├─> For each episode
    │   │
    │   ├─> 重置环境
    │   │   └─ state = env.reset()
    │   │
    │   ├─> For each step
    │   │   ├─ 选择动作: action = agent.select_action(state)
    │   │   ├─ 执行动作: next_state, reward, done = env.step(action)
    │   │   ├─ 奖励塑形: shaped_reward = shaper.compute(...)
    │   │   ├─ 存储经验: agent.store(state, action, reward)
    │   │   └─ 检查终止: if done or step >= max_steps: break
    │   │
    │   ├─> PPO 更新
    │   │   ├─ 计算回报: returns, advantages = compute_gae()
    │   │   ├─ 多轮更新: for epoch in epochs
    │   │   └─ 清空缓冲: clear_memory()
    │   │
    │   └─> 保存检查点
    │       ├─ 定期保存: if episode % interval == 0
    │       └─ 最佳保存: if score < best_score
    │
    └─> 训练结束
        ├─ 保存最终模型
        ├─ 关闭环境
        └─ 输出统计信息
```

---

## 六、扩展建议

### 6.1 特征提取改进

**当前**: 随机特征 (占位)
**改进方向**:
```python
def extract_features(binary_path):
    # 1. 反汇编
    asm = run_objdump(binary_path)
    
    # 2. 提取特征
    features = {
        'instruction_histogram': count_instructions(asm),
        'cfg_features': extract_cfg(asm),
        'register_usage': analyze_registers(asm),
        'constant_features': extract_constants(asm)
    }
    
    # 3. 使用预训练模型编码
    embedding = pretrained_model.encode(features)
    
    return embedding  # shape: (state_dim,)
```

### 6.2 并行采样

```python
# 创建多个环境实例
num_workers = 4
envs = [EnvBridge(...) for _ in range(num_workers)]

# 并行采样
states = [env.reset() for env in envs]
for step in range(max_steps):
    actions = [agent.select_action(s) for s in states]
    
    # 并行执行 (使用 multiprocessing.Pool)
    results = parallel_map(lambda e, a: e.step(a), zip(envs, actions))
    
    # 收集经验
    for state, action, result in zip(states, actions, results):
        agent.store_transition(state, action, result.reward, ...)
```

### 6.3 经验回放优化

```python
class ExperienceCache:
    """缓存已评估的样本"""
    
    def __init__(self):
        self.cache = {}  # (binary_hash, action) -> (score, grad)
    
    def get(self, binary, action):
        key = (hash(binary), action)
        return self.cache.get(key)
    
    def set(self, binary, action, score, grad):
        key = (hash(binary), action)
        self.cache[key] = (score, grad)
```

### 6.4 课程学习

```python
# 逐步降低目标难度
curriculum = [
    {'target': 0.8, 'episodes': 20},
    {'target': 0.6, 'episodes': 30},
    {'target': 0.5, 'episodes': 30},
    {'target': 0.4, 'episodes': 20},
]

for stage in curriculum:
    env.target_score = stage['target']
    train_episodes(stage['episodes'])
```

---

## 七、性能优化清单

- [ ] **特征提取**: 实现真实的二进制特征编码
- [ ] **并行采样**: 同时运行多个环境加速训练
- [ ] **经验缓存**: 避免重复评估相同样本
- [ ] **GPU 加速**: 使用 CUDA 加速网络训练
- [ ] **分布式训练**: 多机多卡并行训练
- [ ] **自适应学习率**: 根据训练进度调整 lr
- [ ] **优先经验回放**: 优先采样重要样本
- [ ] **模型蒸馏**: 压缩模型用于快速推理

---

## 八、常见问题 FAQ

**Q1: 为什么不直接用 Stable Baselines3？**
A: SB3 很强大，但需要标准 Gym 接口。我们的环境涉及外部进程调用和文件操作，自定义实现更灵活。当然，您也可以包装成 Gym 接口后使用 SB3。

**Q2: 训练需要多久？**
A: 取决于：
- 每次变异耗时 (~30-60秒)
- 回合数 (建议 50-100)
- 每回合步数 (建议 20-50)
估计: 50 回合 × 30 步 × 45 秒 ≈ 18 小时

**Q3: 如何加速训练？**
A:
1. 减少每次变异的迭代次数 (iter=1)
2. 使用并行采样 (4个环境)
3. 实现经验缓存避免重复评估
4. 使用 GPU 加速网络训练

**Q4: 如何调整超参数？**
A: 建议顺序:
1. 学习率 lr: [1e-4, 3e-4, 1e-3]
2. 折扣因子 gamma: [0.95, 0.99]
3. PPO epsilon: [0.1, 0.2, 0.3]
4. 网络层数/宽度

**Q5: 训练不收敛怎么办？**
A:
1. 检查奖励设计是否合理
2. 降低学习率
3. 增加训练回合数
4. 检查特征提取是否有效
5. 使用课程学习降低难度

---

## 九、参考资料

### 论文
1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
2. **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016
3. **TRPO**: Schulman et al., "Trust Region Policy Optimization", 2015

### 代码
1. OpenAI Spinning Up: https://spinningup.openai.com/
2. Stable Baselines3: https://stable-baselines3.readthedocs.io/
3. CleanRL: https://github.com/vwxyzjn/cleanrl

### 教程
1. Lilian Weng's Blog: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
2. Hugging Face RL Course: https://huggingface.co/learn/deep-rl-course/

