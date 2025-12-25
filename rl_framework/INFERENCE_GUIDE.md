# PPO 模型推理使用指南

训练完成后如何使用训练好的模型进行二进制代码变异

---

## 📁 训练后的输出文件

训练完成后，会在模型目录（默认 `rl_models/`）生成以下文件：

```
rl_models/
├── ppo_model_best.pt          # ⭐ 最佳模型（推荐使用）
├── ppo_model_final.pt         # 最终模型（训练结束时的模型）
├── ppo_model_ep10.pt          # 第10回合的检查点
├── ppo_model_ep20.pt          # 第20回合的检查点
├── training_log.txt           # 训练日志
├── episode_binaries.txt       # 每个回合生成的二进制文件清单
└── tensorboard/               # TensorBoard 日志目录
```

**推荐使用 `ppo_model_best.pt`**，它是训练过程中相似度分数最低（性能最好）的模型。

---

## 🚀 使用方法

### 1. 单次推理（对单个二进制文件进行变异）

```bash
cd /home/ycy/ours/Deceiving-DNN-based-Binary-Matching

python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/ls \
    --function usage \
    --save-path inference_output_ls_usage \
    --max-steps 30 \
    --target-score 0.40
```

**参数说明：**
- `--model-path`: 训练好的模型路径（必填）
- `--binary`: 要变异的二进制文件路径（必填）
- `--function`: 目标函数名（必填）
- `--save-path`: 变异结果保存路径（必填）
- `--max-steps`: 最大变异步数（默认30）
- `--target-score`: 目标相似度分数（默认0.40，越低表示绕过检测越成功）
- `--state-dim`: 状态维度（必须与训练时一致，默认64）
- `--use-gpu`: 使用GPU加速（可选）

**输出示例：**
```
================================================================================
PPO 推理模式
================================================================================
模型路径: rl_models/ppo_model_best.pt
原始二进制: workdir_1/ls
目标函数: usage
保存路径: inference_output_ls_usage
最大步数: 30
目标分数: 0.40

环境初始化完成 ✓
模型加载完成 ✓ (设备: cpu)

步骤 1/30
------------------------------------------------------------
选择动作: 7 (索引: 2)
状态价值: 0.8521
相似度分数: 0.8934
✨ 发现更好的结果! 分数: 0.8934

步骤 2/30
------------------------------------------------------------
选择动作: 2 (索引: 1)
相似度分数: 0.7234
✨ 发现更好的结果! 分数: 0.7234

...

步骤 8/30
------------------------------------------------------------
选择动作: 11 (索引: 5)
相似度分数: 0.3821
✨ 发现更好的结果! 分数: 0.3821
🎉 成功达到目标! 分数: 0.3821 < 0.40

================================================================================
推理完成
================================================================================
执行步数: 8
最佳分数: 0.3821
✓ 成功达到目标 (分数 < 0.40)
最佳变异结果: inference_output_ls_usage/abc123_container/abc123
推理日志已保存: inference_output_ls_usage/inference_log.txt
```

### 2. 批量推理（对多个二进制文件进行变异）

首先创建批量配置文件 `batch_config.txt`：

```
# 格式：binary,function,save_path
workdir_1/ls,usage,inference_ls_usage
workdir_1/pwd,usage,inference_pwd_usage
workdir_1/cat,main,inference_cat_main
workdir_1/echo,usage,inference_echo_usage
```

然后执行批量推理：

```bash
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --batch \
    --batch-file batch_config.txt \
    --max-steps 30 \
    --target-score 0.40
```

**输出：**
- 每个任务会独立运行推理
- 最终生成 `batch_inference_results.txt` 汇总结果

### 3. 使用GPU加速推理

```bash
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/ls \
    --function usage \
    --save-path inference_output \
    --use-gpu
```

---

## 📊 推理结果分析

### 推理日志文件（`inference_log.txt`）

```
模型: rl_models/ppo_model_best.pt
二进制: workdir_1/ls
函数: usage
最佳分数: 0.3821
成功: True
最佳结果: inference_output/abc123_container/abc123

步骤详情:
step,action,score,grad,reward,value,binary
1,7,0.8934,0.1234,2.3456,0.8521,inference_output/xxx_container/xxx
2,2,0.7234,0.0987,3.4567,0.7892,inference_output/yyy_container/yyy
...
8,11,0.3821,0.0456,8.9012,0.9234,inference_output/abc123_container/abc123
```

### 变异后的二进制文件

推理完成后**自动清理中间文件**，只保留最佳结果：
```
inference_output/
├── abc123_container/       # 最佳变异结果（仅此一个目录）
│   ├── abc123              # 变异后的二进制文件
│   ├── abc123.s            # 对应的汇编代码
│   └── ...
└── inference_log.txt       # 推理日志
```

**清理策略：**
- ✅ 保留：最佳变异结果目录、推理日志
- ❌ 删除：所有中间生成的容器目录
- ❌ 删除：`rl_output/` 中的临时文件
- 💾 节省：显著减少磁盘占用

---

## 🔄 与训练流程的集成

### 完整工作流程

```bash
# 1. 训练模型
python3 rl_framework/ppo_trainer.py \
    --binary workdir_1/ls \
    --function usage \
    --save-path training_output \
    --episodes 50 \
    --max-steps 30 \
    --model-dir rl_models

# 2. 查看训练结果
cat rl_models/training_log.txt
cat rl_models/episode_binaries.txt

# 3. 使用TensorBoard可视化训练过程
tensorboard --logdir=rl_models/tensorboard

# 4. 使用最佳模型进行推理
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/pwd \
    --function usage \
    --save-path inference_pwd
```

### 继续训练（Fine-tuning）

如果训练结果不理想，可以继续训练：

```bash
python3 rl_framework/ppo_trainer.py \
    --binary workdir_1/ls \
    --function usage \
    --save-path training_output \
    --resume rl_models/ppo_model_best.pt \
    --episodes 50
```

---

## 🎯 使用技巧

### 1. 选择合适的模型

- **`ppo_model_best.pt`**: 推荐用于生产环境，性能最优
- **`ppo_model_final.pt`**: 训练结束时的模型，可能没有完全收敛
- **`ppo_model_ep{N}.pt`**: 特定回合的模型，用于调试或对比

### 2. 调整推理参数

```bash
# 快速模式（减少步数）
--max-steps 10 --target-score 0.50

# 精确模式（更多步数，更严格的目标）
--max-steps 50 --target-score 0.30

# 平衡模式（默认）
--max-steps 30 --target-score 0.40
```

### 3. 状态维度必须一致

⚠️ **重要**: 推理时的 `--state-dim` 必须与训练时一致！

```bash
# 训练时使用 state-dim=128
python3 rl_framework/ppo_trainer.py --state-dim 128 ...

# 推理时也要使用 state-dim=128
python3 rl_framework/ppo_inference.py --state-dim 128 ...
```

### 4. 批量推理优化

对于大量二进制文件，可以：
1. 分批处理（每批50-100个）
2. 使用多个GPU并行（修改脚本支持多进程）
3. 设置较小的 `max-steps` 加快速度

---

## 🐛 常见问题

### Q1: 推理时提示模型文件不存在

**解决方案：**
```bash
# 检查模型文件
ls -lh rl_models/

# 确认路径正确
--model-path rl_models/ppo_model_best.pt
```

### Q2: 推理结果不理想（分数较高）

**可能原因：**
1. 训练不充分 → 增加训练回合数
2. 目标函数特征不同 → 针对新函数继续训练
3. 模型选择不当 → 尝试其他检查点

**解决方案：**
```bash
# 使用新数据继续训练
python3 rl_framework/ppo_trainer.py \
    --resume rl_models/ppo_model_best.pt \
    --binary <新的二进制> \
    --function <新的函数> \
    --episodes 30
```

### Q3: 状态维度不匹配

**错误信息：**
```
RuntimeError: Error(s) in loading state_dict for PolicyNetwork:
    size mismatch for actor.0.weight: copying a param with shape torch.Size([256, 64]) 
    from checkpoint, the shape in current model is torch.Size([256, 128]).
```

**解决方案：**
检查训练时使用的 `state-dim` 并在推理时保持一致。

### Q4: GPU内存不足

**解决方案：**
```bash
# 使用CPU推理（去掉 --use-gpu）
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/ls \
    --function usage \
    --save-path inference_output
```

---

## 📈 性能评估

### 评估推理效果

1. **相似度分数**：越低越好，< 0.40 表示成功绕过检测
2. **步数**：越少越好，表示模型策略更高效
3. **成功率**：批量推理时的成功比例

### 对比不同模型

```bash
# 评估最佳模型
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/ls --function usage \
    --save-path eval_best

# 评估最终模型
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_final.pt \
    --binary workdir_1/ls --function usage \
    --save-path eval_final

# 比较结果
cat eval_best/inference_log.txt
cat eval_final/inference_log.txt
```

---

## 🔗 相关文档

- [训练指南](README.md)
- [架构说明](ARCHITECTURE.md)
- [可视化说明](VISUALIZATION_README.md)
- [改进日志](IMPROVEMENTS.md)

---

## 📝 示例脚本

创建一个快速推理脚本 `quick_inference.sh`：

```bash
#!/bin/bash
# Quick Inference Script

MODEL_PATH="rl_models/ppo_model_best.pt"
BINARY="${1:-workdir_1/ls}"
FUNCTION="${2:-usage}"
OUTPUT="inference_$(basename $BINARY)_$FUNCTION"

echo "使用模型: $MODEL_PATH"
echo "二进制文件: $BINARY"
echo "目标函数: $FUNCTION"
echo "输出目录: $OUTPUT"
echo ""

python3 rl_framework/ppo_inference.py \
    --model-path $MODEL_PATH \
    --binary $BINARY \
    --function $FUNCTION \
    --save-path $OUTPUT \
    --max-steps 30 \
    --target-score 0.40

echo ""
echo "推理完成！"
echo "查看结果: cat $OUTPUT/inference_log.txt"
```

使用方法：
```bash
chmod +x quick_inference.sh
./quick_inference.sh workdir_1/pwd usage
```

---

## ✅ 总结

训练后使用流程：

1. ✅ **训练模型** → 生成 `rl_models/ppo_model_best.pt`
2. ✅ **推理使用** → 用 `ppo_inference.py` 加载模型
3. ✅ **批量处理** → 使用批量模式处理多个目标
4. ✅ **结果分析** → 查看 `inference_log.txt`
5. ✅ **持续优化** → 基于结果继续训练或调整参数

Happy hacking! 🎉

---

## 📚 完整文档

**本文档已被合并到统一文档中，请查看：**

👉 **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** 👈

包含以下所有内容：
- ✅ 快速开始
- ✅ 架构设计  
- ✅ 核心模块详解
- ✅ **训练与推理指南（本文档内容）**
- ✅ 特征提取说明
- ✅ 训练可视化
- ✅ 改进日志
- ✅ 故障排除
- ✅ 开发参考

**一份文档，全部内容！**

