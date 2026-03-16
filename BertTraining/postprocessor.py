"""
后处理器：结果保存、评测指标计算、checkpoint管理
"""
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from fileio import write_json, write_jsonl


class Postprocessor:
    """后处理器：负责结果保存、评测指标计算、checkpoint管理"""

    def __init__(self, output_dir: str, config: Any = None):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(output_dir, exist_ok=True)

        # 训练历史
        self.train_history = {
            "train_loss": [],
            "train_lr": [],
            "eval_metrics": [],
            "epoch_time": []
        }

        # 最佳模型跟踪
        self.best_metric = None
        self.best_metric_name = "eval_accuracy"

        # 检查点保存管理
        self.save_total_limit = getattr(config, 'save_total_limit', -1) if config else -1
        self.saved_checkpoints = []

    def save_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "eval"):
        """保存评估指标"""
        metrics_dict = {f"{prefix}_{k}": v for k, v in metrics.items()}
        metrics_dict["step"] = step

        # 保存到历史记录
        if prefix == "eval":
            self.train_history["eval_metrics"].append(metrics_dict)

        # 保存到文件
        output_file = os.path.join(self.output_dir, f"{prefix}_metrics.json")
        with open(output_file, 'a' if os.path.exists(output_file) else 'w') as f:
            if os.path.getsize(output_file) > 0:
                f.write("\n")
            f.write(json.dumps(metrics_dict))

        return metrics_dict

    def save_training_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.output_dir, "training_history.json")
        write_json(self.train_history, history_file, indent=2)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """打印训练指标"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        print(f"[{prefix.capitalize()} Step {step}] {metrics_str}")

    def compute_classification_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "binary"
    ) -> Dict[str, float]:
        """计算分类指标"""
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # 去除padding (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_labels: int
    ) -> Dict[str, Any]:
        """计算混淆矩阵"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # 去除padding
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        cm = confusion_matrix(labels, predictions, labels=list(range(num_labels)))

        return {
            "confusion_matrix": cm.tolist(),
            "num_labels": num_labels
        }

    def is_best_model(self, metrics: Dict[str, float]) -> bool:
        """判断是否为最佳模型"""
        metric_value = metrics.get(self.best_metric_name)
        if metric_value is None:
            return False

        if self.best_metric is None or metric_value > self.best_metric:
            self.best_metric = metric_value
            return True
        return False

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        save_optimizer: bool = True
    ):
        """保存模型checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch{epoch}-step{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        # DeepSpeed会自己保存模型，这里保存分词器配置等
        model_path = os.path.join(checkpoint_dir, "pytorch_model.pt")
        # 注意：DeepSpeed训练时模型参数被分布式管理，需要特殊处理
        # 这里只保存非DeepSpeed部分的checkpoint
        if hasattr(model, 'module'):
            # DataParallel/DistributedDataParallel
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state_dict": state_dict,
            "metrics": metrics or {}
        }, model_path)

        # 保存优化器和调度器
        if save_optimizer and optimizer is not None:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save({
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None
            }, optimizer_path)

        # 保存指标
        if metrics:
            metrics_path = os.path.join(checkpoint_dir, "metrics.json")
            write_json(metrics, metrics_path)

        print(f"Checkpoint saved to {checkpoint_dir}")

        # 记录保存的checkpoint
        self.saved_checkpoints.append(checkpoint_dir)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            best_model_path = os.path.join(best_dir, "pytorch_model.pt")
            torch.save({
                "step": step,
                "epoch": epoch,
                "model_state_dict": state_dict,
                "metrics": metrics or {}
            }, best_model_path)

            if metrics:
                best_metrics_path = os.path.join(best_dir, "metrics.json")
                write_json(metrics, best_metrics_path)

            print(f"Best model saved to {best_dir} (metric: {self.best_metric:.4f})")

        # 限制保存的checkpoint数量
        if self.save_total_limit > 0 and len(self.saved_checkpoints) > self.save_total_limit:
            # 删除旧的checkpoint
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                import shutil
                shutil.rmtree(old_checkpoint)
                print(f"Old checkpoint removed: {old_checkpoint}")

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """加载checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 加载模型
        model_path = os.path.join(checkpoint_path, "pytorch_model.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {model_path}")

        # 加载优化器和调度器
        if optimizer and scheduler:
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                opt_checkpoint = torch.load(optimizer_path, map_location="cpu")
                optimizer.load_state_dict(opt_checkpoint["optimizer_state_dict"])
                if scheduler and opt_checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(opt_checkpoint["scheduler_state_dict"])
                print(f"Optimizer and scheduler loaded from {optimizer_path}")

        # 加载指标
        metrics_path = os.path.join(checkpoint_path, "metrics.json")
        if os.path.exists(metrics_path):
            metrics = json.load(open(metrics_path))
            print(f"Metrics loaded: {metrics}")

        return checkpoint.get("step", 0), checkpoint.get("epoch", 0)

    def save_predictions(
        self,
        predictions: List[int],
        labels: Optional[List[int]],
        texts: List[str],
        output_file: str = "predictions.jsonl"
    ):
        """保存预测结果"""
        results = []
        for i, (pred, text) in enumerate(zip(predictions, texts)):
            result = {
                "index": i,
                "text": text[:200] if len(text) > 200 else text,  # 截断长文本
                "prediction": int(pred)
            }
            if labels is not None:
                result["label"] = int(labels[i])
                result["correct"] = int(pred == labels[i])
            results.append(result)

        output_path = os.path.join(self.output_dir, output_file)
        write_jsonl(results, output_path)
        print(f"Predictions saved to {output_path}")

    def save_config(self, config: Any):
        """保存训练配置"""
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        config_file = os.path.join(self.output_dir, "training_config.json")
        write_json(config_dict, config_file, indent=2)
        print(f"Config saved to {config_file}")

    def finalize(self):
        """训练结束后的收尾工作"""
        self.save_training_history()
        print(f"Training history saved to {os.path.join(self.output_dir, 'training_history.json')}")


class MetricsTracker:
    """指标追踪器 - 用于跟踪训练过程中的各项指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.loss = 0.0
        self.count = 0
        self.metrics = {}

    def update(self, loss: float, batch_size: int, **kwargs):
        """更新指标"""
        self.loss += loss * batch_size
        self.count += batch_size
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
            self.metrics[key] += value * batch_size

    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        avg_metrics = {"loss": self.loss / self.count if self.count > 0 else 0.0}
        for key, value in self.metrics.items():
            avg_metrics[key] = value / self.count if self.count > 0 else 0.0
        return avg_metrics


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, output_dir: str, log_file: str = "training.log"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, log_file)
        self.start_time = time.time()

    def log(self, message: str, print_console: bool = True):
        """记录日志"""
        elapsed = time.time() - self.start_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}][{elapsed:.1f}s] {message}"

        if print_console:
            print(log_line)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

    def log_hyperparameters(self, config: Any):
        """记录超参数"""
        self.log("=" * 60)
        self.log("Hyperparameters:")
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        for key, value in config_dict.items():
            if not key.startswith('_') and value is not None:
                self.log(f"  {key}: {value}")
        self.log("=" * 60)
