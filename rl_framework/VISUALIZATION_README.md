# 强化学习训练可视化指南

本项目提供了两种训练过程可视化方案：

## 方案 1: TensorBoard (推荐)

### 特点
- ✅ 实时查看训练进度
- ✅ 交互式图表，支持缩放、平滑
- ✅ 自动更新，无需重新生成
- ✅ 支持多次训练对比

### 使用方法

#### 1. 安装依赖
```bash
pip install tensorboard
```

#### 2. 启动训练
训练脚本会自动记录 TensorBoard 数据到 `./rl_models/tensorboard/`

```bash
cd rl_framework
python3 ppo_trainer.py \
    --binary ../bin_bk/pwd \
    --function usage \
    --save-path ./rl_output \
    --episodes 100
```

#### 3. 启动 TensorBoard
在另一个终端窗口运行：

```bash
tensorboard --logdir=./rl_models/tensorboard
```

#### 4. 浏览器访问
打开浏览器，访问：`http://localhost:6006`

### 可视化指标

#### 步级别指标（Step/）
- `Raw_Reward` - 原始奖励
- `Shaped_Reward` - 塑形后奖励
- `Value` - 价值函数估计
- `Action` - 执行的动作
- `Similarity_Score` - 相似度分数
- `Gradient` - 梯度值

#### 回合级别指标（Episode/）
- `Total_Reward` - 回合总奖励
- `Average_Reward` - 回合平均奖励
- `Policy_Loss` - 策略损失
- `Steps` - 回合步数
- `Success_Count` - 累计成功次数
- `Final_Score` - 回合最终分数
- `Best_Score` - 历史最佳分数

---

## 方案 2: Matplotlib 静态图表

### 特点
- ✅ 无需额外服务
- ✅ 生成高质量 PNG 图像
- ✅ 包含移动平均线
- ✅ 自动统计信息

### 使用方法

#### 1. 安装依赖
```bash
pip install matplotlib numpy
```

#### 2. 生成可视化图表
```bash
cd rl_framework
python3 plot_training.py \
    --log-file ../rl_models/training_log.txt \
    --output-dir ../rl_models/plots
```

#### 3. 查看生成的图表
图表将保存在 `./rl_models/plots/` 目录下：

- `total_reward.png` - 总奖励曲线
- `average_reward.png` - 平均奖励曲线
- `policy_loss.png` - 策略损失曲线
- `steps_per_episode.png` - 每回合步数
- `training_summary.png` - 综合面板（4 合 1）

---

## 对比

| 特性 | TensorBoard | Matplotlib |
|------|-------------|------------|
| 实时监控 | ✅ | ❌ |
| 交互式 | ✅ | ❌ |
| 多实验对比 | ✅ | ❌ |
| 离线查看 | ❌ | ✅ |
| 部署简单 | 需服务 | 直接图片 |
| 指标详细度 | 高 | 中 |

---

## 推荐工作流

### 训练期间
使用 **TensorBoard** 实时监控：
```bash
# 终端 1: 启动训练
python3 ppo_trainer.py ...

# 终端 2: 启动 TensorBoard
tensorboard --logdir=./rl_models/tensorboard
```

### 训练完成后
使用 **Matplotlib** 生成报告图：
```bash
python3 plot_training.py
```

---

## 故障排除

### TensorBoard 无法访问
```bash
# 检查端口占用
lsof -i :6006

# 指定其他端口
tensorboard --logdir=./rl_models/tensorboard --port=6007
```

### Matplotlib 中文显示问题
脚本已配置中文字体支持，如仍有问题：
```python
# 在 plot_training.py 中修改
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
```

### 图表不更新
TensorBoard 会自动刷新，Matplotlib 需要重新运行脚本：
```bash
python3 plot_training.py
```

---

## 扩展

### 添加自定义指标（TensorBoard）
在 `ppo_trainer.py` 中添加：
```python
writer.add_scalar('Custom/my_metric', value, step)
```

### 修改绘图样式（Matplotlib）
在 `plot_training.py` 中自定义：
```python
plt.style.use('seaborn-darkgrid')  # 更改主题
```

---

## 参考
- TensorBoard 文档: https://www.tensorflow.org/tensorboard
- Matplotlib 文档: https://matplotlib.org/

