"""
主入口：命令行接口
"""
import os
import sys
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessor import Preprocessor, create_dataloader
from postprocessor import Postprocessor, TrainingLogger
from trainer import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    # 1. 初始化预处理器（解析参数）
    print("正在初始化...")
    preprocessor = Preprocessor()
    config = preprocessor.config

    # 设置随机种子
    set_seed(config.seed)

    # 2. 初始化后处理器
    postprocessor = Postprocessor(
        output_dir=config.output_dir,
        config=config
    )

    # 3. 初始化日志记录器
    logger = TrainingLogger(config.output_dir)
    logger.log_hyperparameters(config)

    # 4. 加载数据
    logger.log("加载数据...")
    datasets, raw_data = preprocessor.load_data()

    # 5. 创建数据加载器
    train_loader = None
    valid_loader = None
    test_loader = None

    if "train" in datasets:
        train_loader = create_dataloader(
            datasets["train"],
            batch_size=config.per_device_train_batch_size,
            shuffle=True
        )
        logger.log(f"训练集大小: {len(datasets['train'])}, batch数: {len(train_loader)}")

    if "valid" in datasets:
        valid_loader = create_dataloader(
            datasets["valid"],
            batch_size=config.per_device_eval_batch_size,
            shuffle=False
        )
        logger.log(f"验证集大小: {len(datasets['valid'])}, batch数: {len(valid_loader)}")

    if "test" in datasets:
        test_loader = create_dataloader(
            datasets["test"],
            batch_size=config.per_device_eval_batch_size,
            shuffle=False
        )
        logger.log(f"测试集大小: {len(datasets['test'])}, batch数: {len(test_loader)}")

    # 6. 加载模型
    logger.log("加载模型...")
    model = preprocessor.load_model()
    logger.log(f"模型类型: {config.model_type}, 标签数: {config.num_labels}")

    # 7. 创建优化器和调度器
    logger.log("创建优化器和调度器...")
    optimizer = preprocessor.create_optimizer(model)

    num_training_steps = len(train_loader) * config.num_train_epochs // config.gradient_accumulation_steps if train_loader else 0
    scheduler = preprocessor.create_scheduler(optimizer, num_training_steps)

    # 8. 初始化训练器
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    logger.log(f"使用设备: {device}")

    # 将模型移到设备上
    model = model.to(device)
    logger.log(f"模型已移至: {device}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        postprocessor=postprocessor,
        device=device,
        local_rank=config.local_rank
    )

    # 9. 保存DeepSpeed配置
    if config.deepspeed_config is None:
        ds_config_path = postprocessor.save_deepspeed_config(config.output_dir)
        logger.log(f"DeepSpeed配置已保存到: {ds_config_path}")

    # 10. 开始训练或评测
    if config.resume_from_checkpoint:
        # 从checkpoint恢复
        logger.log(f"从checkpoint恢复: {config.resume_from_checkpoint}")
        # TODO: 实现checkpoint恢复

    # 执行训练
    if train_loader:
        trainer.train(train_loader, valid_loader)

    # 执行测试（如果有测试集或训练后需要评测）
    if test_loader and config.num_train_epochs == 0:
        # 仅评测模式
        logger.log("仅执行评测...")
        eval_metrics = trainer.evaluate(test_loader)
        logger.log(f"测试集评测结果: {eval_metrics}")
    elif test_loader and train_loader:
        # 训练后评测
        logger.log("训练完成，在测试集上进行评测...")
        test_metrics = trainer.evaluate(test_loader)
        logger.log(f"测试集评测结果: {test_metrics}")

        # 保存测试结果
        predictions = trainer.predict(test_loader)
        postprocessor.save_predictions(
            predictions["predictions"],
            None,
            [item["text"] for item in datasets["test"]],
            output_file="test_predictions.jsonl"
        )

    logger.log("全部任务完成!")


if __name__ == "__main__":
    main()
