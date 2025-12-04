#!/bin/bash
# Quick Start Script for PPO Training

set -e

echo "=========================================="
echo "PPO-Based Binary Perturbation Framework"
echo "=========================================="

# 检查 Python 3
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python 3"
    exit 1
fi

# 检查 Python 2
if ! command -v python2 &> /dev/null; then
    echo "错误: 未找到 Python 2"
    exit 1
fi

# 安装 Python 3 依赖
echo ""
echo "安装 Python 3 依赖..."
pip3 install -r rl_framework/requirements.txt

# 设置默认参数
BINARY="${1:-workdir_1/ls}"
FUNCTION="${2:-usage}"
SAVE_PATH="${3:-function_container_usage_ls}"
EPISODES="${4:-50}"
MAX_STEPS="${5:-30}"

echo ""
echo "配置:"
echo "  二进制文件: $BINARY"
echo "  目标函数: $FUNCTION"
echo "  保存路径: $SAVE_PATH"
echo "  训练回合: $EPISODES"
echo "  最大步数: $MAX_STEPS"
echo ""

# 创建目录
mkdir -p rl_models
mkdir -p $SAVE_PATH

# 测试 PPO Agent
echo "测试 PPO Agent..."
python3 rl_framework/ppo_agent.py
if [ $? -ne 0 ]; then
    echo "错误: PPO Agent 测试失败"
    exit 1
fi

echo ""
echo "✓ 所有测试通过"
echo ""
echo "开始训练..."
echo ""
echo "python3 rl_framework/ppo_trainer.py \
--binary $BINARY \
--function $FUNCTION \
--save-path $SAVE_PATH \
--episodes $EPISODES \
--max-steps $MAX_STEPS \
--save-interval 10 \
--model-dir rl_models" 
echo ""

read -p "Press Enter to continue"

# 启动训练（简化版，无需进程通信）
python3 rl_framework/ppo_trainer.py \
    --binary $BINARY \
    --function $FUNCTION \
    --save-path $SAVE_PATH \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS \
    --save-interval 10 \
    --model-dir rl_models

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="
echo "模型保存在: rl_models/"
echo "训练日志: rl_models/training_log.txt"
echo "成功样本: $SAVE_PATH/success.log"

