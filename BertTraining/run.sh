#!/bin/bash
# BERT系列模型训练脚本示例
# 使用方法: bash run.sh [模式] [参数...]

# =============================================================================
# 配置参数 (可根据需要修改)
# =============================================================================

# 模型配置
MODEL_NAME_OR_PATH="bert-base-uncased"  # HuggingFace模型名称或本地路径
MODEL_TYPE="bert"                         # 模型类型: bert, roberta, deberta, custom
NUM_LABELS=2                             # 分类标签数量
MAX_SEQ_LENGTH=128                       # 最大序列长度

# 数据配置
TRAIN_DATA="data/train.jsonl"             # 训练数据路径
VALID_DATA="data/valid.jsonl"             # 验证数据路径
TEST_DATA="data/test.jsonl"               # 测试数据路径 (可选)
TEXT_FIELD="text"                          # JSONL中文本字段名
LABEL_FIELD="label"                       # JSONL中标签字段名

# 训练配置
OUTPUT_DIR="output/bert-base-experiment"  # 输出目录
NUM_TRAIN_EPOCHS=3                       # 训练轮数
PER_DEVICE_TRAIN_BATCH_SIZE=32            # 每GPU训练batch大小
PER_DEVICE_EVAL_BATCH_SIZE=64             # 每GPU评测batch大小
LEARNING_RATE=5e-5                        # 学习率
WEIGHT_DECAY=0.01                         # 权重衰减
WARMUP_RATIO=0.1                         # warmup比例
GRADIENT_ACCUMULATION_STEPS=1            # 梯度累积步数
MAX_GRAD_NORM=1.0                         # 梯度裁剪范数

# DeepSpeed配置
DEEPSPEED_CONFIG="configs/ds_config.json" # DeepSpeed配置文件
ZERO_STAGE=2                              # ZeRO优化阶段 (0, 1, 2, 3)
FP16=false                                 # 是否使用混合精度

# 其他配置
SEED=42                                   # 随机种子
LOGGING_STEPS=10                          # 日志打印步数
SAVE_STEPS=500                            # 保存checkpoint步数
SAVE_STRATEGY="epoch"                      # 保存策略: epoch, steps, no
SAVE_TOTAL_LIMIT=-1                        # 最多保存的checkpoint数量, -1表示不限制
EVAL_STEPS=500                            # 评测步数
NO_CUDA=false                             # 不使用CUDA

# =============================================================================
# 运行模式
# =============================================================================

MODE=${1:-"train"}  # 默认训练模式

if [ "$MODE" = "train" ]; then
    echo "=========================================="
    echo "训练模式"
    echo "=========================================="

# 构建参数
ARGS=(
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --model_type "$MODEL_TYPE"
    --num_labels $NUM_LABELS
    --max_seq_length $MAX_SEQ_LENGTH
    --train_data "$TRAIN_DATA"
    --valid_data "$VALID_DATA"
    --test_data "$TEST_DATA"
    --text_field "$TEXT_FIELD"
    --label_field "$LABEL_FIELD"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs $NUM_TRAIN_EPOCHS
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
    --max_grad_norm "$MAX_GRAD_NORM"
    --deepspeed_config "$DEEPSPEED_CONFIG"
    --zero_stage $ZERO_STAGE
    --seed $SEED
    --logging_steps $LOGGING_STEPS
    --save_steps $SAVE_STEPS
    --save_strategy "$SAVE_STRATEGY"
    --save_total_limit $SAVE_TOTAL_LIMIT
    --eval_steps $EVAL_STEPS
)

# 添加可选参数
if [ "$FP16" = "true" ]; then
    ARGS+=(--fp16)
fi

if [ "$NO_CUDA" = "true" ]; then
    ARGS+=(--no_cuda)
fi

python main.py "${ARGS[@]}"

elif [ "$MODE" = "eval" ]; then
    echo "=========================================="
    echo "评测模式"
    echo "=========================================="

    CHECKPOINT_PATH=${2:-"$OUTPUT_DIR/best_model"}

    ARGS=(
        --model_name_or_path "$MODEL_NAME_OR_PATH"
        --model_type "$MODEL_TYPE"
        --num_labels $NUM_LABELS
        --max_seq_length $MAX_SEQ_LENGTH
        --test_data "$TEST_DATA"
        --text_field "$TEXT_FIELD"
        --label_field "$LABEL_FIELD"
        --output_dir "$OUTPUT_DIR"
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE
        --num_train_epochs 0
        --seed $SEED
    )

    if [ "$NO_CUDA" = "true" ]; then
        ARGS+=(--no_cuda)
    fi

    python main.py "${ARGS[@]}"

elif [ "$MODE" = "predict" ]; then
    echo "=========================================="
    echo "预测模式"
    echo "=========================================="

    CHECKPOINT_PATH=${2:-"$OUTPUT_DIR/best_model"}

    ARGS=(
        --model_name_or_path "$MODEL_NAME_OR_PATH"
        --model_type "$MODEL_TYPE"
        --num_labels $NUM_LABELS
        --max_seq_length $MAX_SEQ_LENGTH
        --test_data "$TEST_DATA"
        --text_field "$TEXT_FIELD"
        --output_dir "$OUTPUT_DIR"
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE
        --num_train_epochs 0
        --seed $SEED
    )

    if [ "$NO_CUDA" = "true" ]; then
        ARGS+=(--no_cuda)
    fi

    python main.py "${ARGS[@]}"

else
    echo "未知模式: $MODE"
    echo "可用模式: train, eval, predict"
    echo "用法:"
    echo "  bash run.sh train           # 训练模型"
    echo "  bash run.sh eval [checkpoint] # 评测模型"
    echo "  bash run.sh predict [checkpoint] # 预测"
    exit 1
fi
