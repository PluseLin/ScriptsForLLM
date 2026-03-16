"""
预处理器：参数配置、数据加载、模型加载
"""
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader


class Preprocessor:
    """预处理器：负责参数解析、配置管理、数据加载、模型初始化"""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        self.args = args
        self._parse_args()

    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="BERT系列模型训练脚本",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # ===== 数据参数 =====
        parser.add_argument("--train_data", type=str, required=True,
                            help="训练数据路径 (JSONL格式)")
        parser.add_argument("--valid_data", type=str, default=None,
                            help="验证数据路径 (JSONL格式)")
        parser.add_argument("--test_data", type=str, default=None,
                            help="测试数据路径 (JSONL格式)")
        parser.add_argument("--text_field", type=str, default="text",
                            help="JSONL中文本字段名")
        parser.add_argument("--label_field", type=str, default="label",
                            help="JSONL中标签字段名")
        parser.add_argument("--max_seq_length", type=int, default=128,
                            help="最大序列长度")

        # ===== 模型参数 =====
        parser.add_argument("--model_name_or_path", type=str, required=True,
                            help="HuggingFace模型名称或本地路径")
        parser.add_argument("--model_type", type=str, default="bert",
                            choices=["bert", "roberta", "deberta", "custom"],
                            help="模型类型")
        parser.add_argument("--custom_model_class", type=str, default=None,
                            help="自定义模型类名 (需要自行实现)")
        parser.add_argument("--num_labels", type=int, default=2,
                            help="分类标签数量")
        parser.add_argument("--freeze_encoder", action="store_true",
                            help="是否冻结编码器")

        # ===== 训练参数 =====
        parser.add_argument("--output_dir", type=str, required=True,
                            help="输出目录")
        parser.add_argument("--num_train_epochs", type=int, default=3,
                            help="训练轮数")
        parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                            help="每设备训练batch大小")
        parser.add_argument("--per_device_eval_batch_size", type=int, default=64,
                            help="每设备评测batch大小")
        parser.add_argument("--learning_rate", type=float, default=5e-5,
                            help="学习率")
        parser.add_argument("--weight_decay", type=float, default=0.01,
                            help="权重衰减")
        parser.add_argument("--warmup_ratio", type=float, default=0.1,
                            help="warmup比例")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="梯度累积步数")
        parser.add_argument("--max_grad_norm", type=float, default=1.0,
                            help="最大梯度范数")

        # ===== DeepSpeed参数 =====
        parser.add_argument("--deepspeed_config", type=str, default=None,
                            help="DeepSpeed配置文件路径")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="分布式训练GPU编号")
        parser.add_argument("--zero_stage", type=int, default=2,
                            choices=[0, 1, 2, 3],
                            help="DeepSpeed ZeRO优化阶段")

        # ===== 其他参数 =====
        parser.add_argument("--seed", type=int, default=42,
                            help="随机种子")
        parser.add_argument("--logging_steps", type=int, default=10,
                            help="日志打印步数")
        parser.add_argument("--save_steps", type=int, default=500,
                            help="保存checkpoint步数 (当save_strategy为steps时)")
        parser.add_argument("--save_strategy", type=str, default="epoch",
                            choices=["epoch", "steps", "no"],
                            help="保存checkpoint的策略: epoch(每个epoch), steps(每N步), no(不保存)")
        parser.add_argument("--save_total_limit", type=int, default=-1,
                            help="最多保存的checkpoint数量, -1表示不限制")
        parser.add_argument("--eval_steps", type=int, default=500,
                            help="评测步数")
        parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="从checkpoint恢复训练")
        parser.add_argument("--no_cuda", action="store_true",
                            help="不使用CUDA")
        parser.add_argument("--fp16", action="store_true",
                            help="使用混合精度训练")

        if self.args is not None:
            # 如果传入了预定义的args，直接使用
            self.config = self.args
        else:
            self.config = parser.parse_args()

        # 设置环境变量
        if self.config.local_rank == -1:
            self.config.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_data(self) -> Dict[str, Dataset]:
        """加载训练、验证、测试数据，返回 (Dataset, raw_data) 元组字典"""
        from fileio import read_jsonl

        datasets = {}
        raw_data = {}

        # 加载训练数据
        if os.path.exists(self.config.train_data):
            train_data = read_jsonl(self.config.train_data)
            raw_data["train"] = train_data
            datasets["train"] = self._create_dataset(train_data)
            print(f"训练数据加载完成: {len(train_data)} 条")

        # 加载验证数据
        if self.config.valid_data and os.path.exists(self.config.valid_data):
            valid_data = read_jsonl(self.config.valid_data)
            raw_data["valid"] = valid_data
            datasets["valid"] = self._create_dataset(valid_data)
            print(f"验证数据加载完成: {len(valid_data)} 条")

        # 加载测试数据
        if self.config.test_data and os.path.exists(self.config.test_data):
            test_data = read_jsonl(self.config.test_data)
            raw_data["test"] = test_data
            datasets["test"] = self._create_dataset(test_data)
            print(f"测试数据加载完成: {len(test_data)} 条")

        # 返回包含数据集和原始数据的元组
        return datasets, raw_data

    def _create_dataset(self, data: List[Dict]) -> "BertDataset":
        """创建数据集对象"""
        return BertDataset(
            data=data,
            tokenizer=self.load_tokenizer(),
            max_length=self.config.max_seq_length,
            text_field=self.config.text_field,
            label_field=self.config.label_field,
        )

    def load_tokenizer(self):
        """加载分词器"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=True
        )
        return tokenizer

    def load_model(self, num_labels: Optional[int] = None):
        """加载模型"""
        num_labels = num_labels or self.config.num_labels

        # 检查是否为自定义模型
        if self.config.model_type == "custom" and self.config.custom_model_class:
            model = self._load_custom_model(num_labels)
        else:
            model = self._load_pretrained_model(num_labels)

        # 冻结编码器
        if self.config.freeze_encoder:
            for param in model.bert.parameters():
                param.requires_grad = False

        return model

    def _load_pretrained_model(self, num_labels: int):
        """加载预训练模型"""
        from transformers import AutoModelForSequenceClassification

        config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            num_labels=num_labels
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            config=config
        )

        return model

    def _load_custom_model(self, num_labels: int):
        """加载自定义模型"""
        from transformers import AutoModelForSequenceClassification

        # 这里可以添加自定义模型的加载逻辑
        # 用户可以在这里实现自己的模型架构
        config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            num_labels=num_labels
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            config=config
        )
        # 替换分类头为自定义结构
        hidden_size = config.hidden_size
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, num_labels)
        )
        return model

    def create_optimizer(self, model):
        """创建优化器"""
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )
        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """创建学习率调度器"""
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    def get_deepspeed_config(self):
        """获取DeepSpeed配置"""
        if self.config.deepspeed_config and os.path.exists(self.config.deepspeed_config):
            with open(self.config.deepspeed_config, 'r') as f:
                return json.load(f)

        # 默认DeepSpeed配置
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,
            "zero_optimization": {
                "stage": self.config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "fp16": {
                "enabled": self.config.fp16 and not self.config.no_cuda
            },
            "logging": {
                "interval": self.config.logging_steps
            }
        }

    def save_deepspeed_config(self, output_dir: str):
        """保存DeepSpeed配置"""
        config = self.get_deepspeed_config()
        config_path = os.path.join(output_dir, "ds_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config_path


class BertDataset(Dataset):
    """BERT数据集类"""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        text_field: str = "text",
        label_field: str = "label"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item.get(self.text_field, "")
        label = item.get(self.label_field, 0)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False,
                      num_workers: int = 4, pin_memory: bool = True):
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
