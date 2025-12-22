# 自动清理功能说明

## 功能概述

推理完成后自动清理中间文件，仅保留必要结果。

## 清理策略

### ✅ 保留
- `inference_log.txt` - 推理日志
- 最佳变异结果的容器目录 - 唯一保留的 `*_container/`

### ❌ 删除
- 所有中间生成的容器目录
- `rl_output/` 中的 `mutant_*.bin*` 临时文件
- 其他临时文件

## 代码实现

```python
def cleanup_inference_files(save_path, keep_binary):
    """
    清理推理中间文件，仅保留最佳结果
    
    Args:
        save_path: 输出目录
        keep_binary: 要保留的二进制文件路径
    """
    if not os.path.exists(save_path) or not keep_binary:
        return
    
    # 确定要保留的容器
    keep_path = os.path.abspath(keep_binary)
    keep_container = None
    if '_container' in keep_path:
        parts = keep_path.split('_container')
        if parts:
            keep_container = parts[0] + '_container'
    
    # 删除其他容器
    for container in glob.glob(os.path.join(save_path, '*_container')):
        if keep_container and os.path.abspath(container) == keep_container:
            continue
        shutil.rmtree(container)
    
    # 删除临时文件（保留日志）
    for item in os.listdir(save_path):
        if item == 'inference_log.txt':
            continue
        path = os.path.join(save_path, item)
        if os.path.isfile(path):
            os.remove(path)
```

## 使用示例

### 单次推理
```bash
python3 rl_framework/ppo_inference.py \
    --model-path rl_models/ppo_model_best.pt \
    --binary workdir_1/ls \
    --function usage \
    --save-path inference_output
```

输出目录结构：
```
inference_output/
├── abc123_container/      # 仅最佳结果
│   └── abc123
└── inference_log.txt      # 推理日志
```

### 批量推理
每个任务独立清理，互不影响。

## 优势

- **节省空间**: 删除中间文件，只保留必要结果
- **简洁高效**: 自动化清理，无需手动操作  
- **保留关键**: 最佳结果和日志完整保留
- **安全可靠**: 异常处理，删除失败不影响主流程

## 设计原则

遵循 **"Do one thing and do it well"** 原则：
- 单一职责：只负责清理
- 错误容忍：删除失败不中断
- 清晰日志：报告清理结果
- 零配置：自动判断保留内容

---

*"Talk is cheap. Show me the code."* - Linus Torvalds

