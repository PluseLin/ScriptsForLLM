"""
训练器：DeepSpeed训练循环
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedEngine = None


class Trainer:
    """DeepSpeed训练器"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        config,
        postprocessor,
        device: Optional[torch.device] = None,
        local_rank: int = -1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.postprocessor = postprocessor
        self.local_rank = local_rank

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        else:
            self.device = device

        # DeepSpeed
        self.deepspeed_engine = None
        self.use_deepspeed = DEEPSPEED_AVAILABLE and config.deepspeed_config is not None

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.total_train_batch_size = 0

    def prepare_deepspeed(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        """初始化DeepSpeed"""
        # 检查是否可以使用DeepSpeed
        if not self.use_deepspeed:
            print("DeepSpeed not enabled, using native PyTorch training")
            return

        # 检查是否满足分布式训练条件
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", None)

        # 如果是单机单卡模式，跳过DeepSpeed
        if world_size == 1 or local_rank == -1 or master_addr is None:
            print(f"Single GPU mode detected (world_size={world_size}, local_rank={local_rank}), using native PyTorch training")
            self.use_deepspeed = False
            return

        # 获取DeepSpeed配置
        ds_config = self._get_deepspeed_config()

        # 计算总batch size
        train_batch_size = self.config.per_device_train_batch_size
        if torch.cuda.is_available():
            train_batch_size *= torch.cuda.device_count()

        # 初始化分布式环境
        self.local_rank = local_rank

        if not torch.distributed.is_initialized():
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=self.local_rank
            )

        # 初始化DeepSpeed
        self.deepspeed_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=ds_config,
            dist_init_required=True
        )

        print(f"DeepSpeed initialized with config: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')} stage")

    def _get_deepspeed_config(self):
        """获取DeepSpeed配置"""
        # 尝试从配置文件读取
        if self.config.deepspeed_config and os.path.exists(self.config.deepspeed_config):
            import json
            with open(self.config.deepspeed_config, 'r') as f:
                return json.load(f)

        # 默认配置
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

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        """执行训练"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)

        # 保存配置
        self.postprocessor.save_config(self.config)

        # 准备DeepSpeed
        self.prepare_deepspeed(train_loader, eval_loader)

        # 计算总训练步数
        num_update_steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        num_train_epochs = self.config.num_train_epochs
        total_steps = num_update_steps_per_epoch * num_train_epochs

        print(f"总训练步数: {total_steps}")
        print(f"每epoch步数: {num_update_steps_per_epoch}")
        print(f"训练轮数: {num_train_epochs}")

        # 恢复checkpoint
        if self.config.resume_from_checkpoint:
            self._resume_from_checkpoint(self.config.resume_from_checkpoint)

        # 训练循环
        for epoch in range(num_train_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_train_epochs}")
            print(f"{'='*60}")

            epoch_start_time = time.time()

            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader)

            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1} 完成, 耗时: {epoch_time:.2f}秒")

            # 记录训练指标
            self.postprocessor.train_history["train_loss"].append(train_metrics.get("loss", 0))
            self.postprocessor.train_history["epoch_time"].append(epoch_time)

            # 保存checkpoint - 基于epoch
            save_strategy = getattr(self.config, 'save_strategy', 'epoch')
            if save_strategy == "epoch":
                self._save_checkpoint(eval_loader, epoch, num_update_steps_per_epoch, train_metrics)

            # 评测
            if eval_loader is not None and (epoch + 1) % max(1, self.config.eval_steps // num_update_steps_per_epoch) == 0:
                eval_metrics = self.evaluate(eval_loader)
                is_best = self.postprocessor.is_best_model(eval_metrics)
                # 每次评测后也可以保存checkpoint
                if save_strategy in ["steps", "epoch"]:
                    self._save_checkpoint(eval_loader, epoch, num_update_steps_per_epoch, eval_metrics, is_best=is_best)

        # 训练结束
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        self.postprocessor.finalize()

    def _save_checkpoint(self, eval_loader, epoch, steps_per_epoch, metrics, is_best=False):
        """保存检查点"""
        save_strategy = getattr(self.config, 'save_strategy', 'epoch')
        if save_strategy == "no":
            return

        if save_strategy == "epoch":
            # 每个epoch保存
            checkpoint_path = self.postprocessor.save_checkpoint(
                self.model,
                self.optimizer if not self.use_deepspeed else None,
                self.scheduler if not self.use_deepspeed else None,
                self.global_step,
                epoch + 1,
                metrics,
                is_best=is_best,
                save_optimizer=not self.use_deepspeed
            )
        elif save_strategy == "steps":
            # 每N步保存
            if self.global_step % self.config.save_steps == 0:
                checkpoint_path = self.postprocessor.save_checkpoint(
                    self.model,
                    self.optimizer if not self.use_deepspeed else None,
                    self.scheduler if not self.use_deepspeed else None,
                    self.global_step,
                    epoch + 1,
                    metrics,
                    is_best=is_best,
                    save_optimizer=not self.use_deepspeed
                )

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 进度条
        if self.local_rank <= 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = train_loader

        for step, batch in enumerate(pbar):
            batch_start_time = time.time()

            # 将数据移到设备
            if self.use_deepspeed:
                # DeepSpeed会自动处理数据移到设备
                pass
            else:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # 前向传播
            if self.use_deepspeed:
                loss = self.deepspeed_engine(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                # 获取logits (根据模型架构可能不同)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs.last_hidden_state[:, 0, :]  # [CLS] token

                # 计算损失 - 使用 reshape 避免 view 问题
                loss_fct = nn.CrossEntropyLoss()
                logits_flat = logits.view(-1, logits.size(-1))  # [batch_size, num_labels]
                labels_flat = batch["labels"].view(-1)
                loss = loss_fct(logits_flat, labels_flat)

            # 梯度累积
            if self.use_deepspeed:
                self.deepspeed_engine.backward(loss)
                self.deepspeed_engine.step()
            else:
                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1

            # 日志记录
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                step_time = time.time() - batch_start_time

                log_metrics = {
                    "loss": avg_loss,
                    "lr": lr,
                    "step_time": step_time
                }

                if self.local_rank <= 0:
                    self.postprocessor.log_metrics(log_metrics, self.global_step, "train")
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        print("\n" + "=" * 60)
        print("开始评估")
        print("=" * 60)

        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(eval_loader, desc="Evaluating") if self.local_rank <= 0 else eval_loader

            for batch in pbar:
                # 数据移到设备
                if not self.use_deepspeed:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                # 前向传播
                if self.use_deepspeed:
                    outputs = self.deepspeed_engine(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss

                    # 获取logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs.last_hidden_state[:, 0, :]
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs.last_hidden_state[:, 0, :]

                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), batch["labels"].view(-1))

                total_loss += loss.item()
                num_batches += 1

                # 获取预测
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # 计算指标
        import numpy as np
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        eval_metrics = self.postprocessor.compute_classification_metrics(
            all_predictions,
            all_labels,
            average="binary" if self.config.num_labels == 2 else "macro"
        )
        eval_metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        # 打印结果
        print("\n评估结果:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 保存评估指标
        self.postprocessor.save_metrics(eval_metrics, self.global_step, "eval")

        # 恢复训练模式
        self.model.train()

        return eval_metrics

    def predict(self, test_loader: DataLoader) -> Dict[str, Any]:
        """预测"""
        print("\n" + "=" * 60)
        print("开始预测")
        print("=" * 60)

        self.model.eval()
        all_predictions = []
        all_logits = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Predicting") if self.local_rank <= 0 else test_loader

            for batch in pbar:
                # 数据移到设备
                if not self.use_deepspeed:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                # 前向传播
                if self.use_deepspeed:
                    outputs = self.deepspeed_engine(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs.last_hidden_state[:, 0, :]
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs.last_hidden_state[:, 0, :]

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        return {
            "predictions": all_predictions,
            "logits": all_logits
        }

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """从checkpoint恢复训练"""
        print(f"从checkpoint恢复: {checkpoint_path}")
        # DeepSpeed的恢复需要在initialize时指定
        # 这里可以添加恢复逻辑
        pass
