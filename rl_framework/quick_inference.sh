#!/bin/bash
# Quick Inference Script - 快速推理脚本
# 使用训练好的模型对二进制文件进行变异

set -e

echo "=========================================="
echo "PPO 模型推理"
echo "=========================================="

# 默认参数
MODEL_PATH="${MODEL_PATH:-rl_models/ppo_model_best.pt}"
BINARY="${1}"
FUNCTION="${2}"
MAX_STEPS="${3:-30}"
TARGET_SCORE="${4:-0.40}"
STATE_DIM="${STATE_DIM:-64}"

# 检查必需参数
if [ -z "$BINARY" ] || [ -z "$FUNCTION" ]; then
    echo "用法: $0 <二进制文件> <函数名> [最大步数] [目标分数]"
    echo ""
    echo "示例:"
    echo "  $0 workdir_1/ls usage"
    echo "  $0 workdir_1/pwd usage 50 0.30"
    echo ""
    echo "环境变量:"
    echo "  MODEL_PATH  - 模型路径 (默认: rl_models/ppo_model_best.pt)"
    echo "  STATE_DIM   - 状态维度 (默认: 64, 必须与训练时一致)"
    echo ""
    echo "示例:"
    echo "  MODEL_PATH=rl_models/ppo_model_ep50.pt $0 workdir_1/ls usage"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo ""
    echo "可用的模型文件:"
    ls -lh rl_models/*.pt 2>/dev/null || echo "  (没有找到模型文件)"
    echo ""
    echo "请先训练模型或指定正确的模型路径:"
    echo "  export MODEL_PATH=rl_models/ppo_model_best.pt"
    exit 1
fi

# 检查二进制文件
if [ ! -f "$BINARY" ]; then
    echo "错误: 二进制文件不存在: $BINARY"
    exit 1
fi

# 生成输出目录名
BINARY_NAME=$(basename "$BINARY")
OUTPUT="inference_${BINARY_NAME}_${FUNCTION}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "配置:"
echo "  模型路径:   $MODEL_PATH"
echo "  二进制文件: $BINARY"
echo "  目标函数:   $FUNCTION"
echo "  输出目录:   $OUTPUT"
echo "  最大步数:   $MAX_STEPS"
echo "  目标分数:   $TARGET_SCORE"
echo "  状态维度:   $STATE_DIM"
echo ""

# 确认执行
read -p "按 Enter 键开始推理，或 Ctrl+C 取消..." 

# 执行推理
echo ""
echo "开始推理..."
echo ""

python3 rl_framework/ppo_inference.py \
    --model-path "$MODEL_PATH" \
    --binary "$BINARY" \
    --function "$FUNCTION" \
    --save-path "$OUTPUT" \
    --max-steps "$MAX_STEPS" \
    --target-score "$TARGET_SCORE" \
    --state-dim "$STATE_DIM"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "推理完成 ✓"
    echo "=========================================="
    echo ""
    echo "查看详细结果:"
    echo "  cat $OUTPUT/inference_log.txt"
    echo ""
    echo "变异后的文件:"
    echo "  ls -lh $OUTPUT/"
else
    echo "推理失败 ✗"
    echo "=========================================="
    echo ""
    echo "请检查日志输出排查问题"
fi

exit $EXIT_CODE

