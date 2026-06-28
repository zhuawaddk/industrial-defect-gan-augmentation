#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块
包含训练循环、损失函数和Optuna集成
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import lpips
from tqdm import tqdm

from model.models.focus_stylegan import FocusStyleGAN
from model.data.loader import MVTecADDataset, create_dataloader
from model.utils.config import Config


class PerceptualLoss(nn.Module):
    """感知损失"""

    def __init__(self):
        super().__init__()
        # 使用预训练的VGG19，优先从本地加载
        from model.utils.weights import find_weight, load_state_dict_from_local
        try:
            state_dict = load_state_dict_from_local("vgg19", map_location="cpu")
            vgg = models.vgg19(pretrained=False)
            vgg.load_state_dict(state_dict)
            vgg = vgg.features
        except (FileNotFoundError, RuntimeError):
            vgg = models.vgg19(pretrained=True).features

        # 提取特定层的特征
        self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.layers = nn.ModuleList()

        layer_idx = 0
        target_layers_collected = set()

        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                layer_idx += 1
                name = f'conv{layer_idx}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{layer_idx}'
                layer = nn.ReLU(inplace=False)  # 需要inplace=False以获取梯度
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{layer_idx}'
            else:
                continue

            self.layers.append(layer)
            if name in self.layer_names:
                target_layers_collected.add(name)
                # 如果已经收集了所有目标层，可以提前停止
                # 但实际上需要收集到最后一个目标层（relu5_1）的所有前置层
                # 所以继续直到 layer_idx >= 5 且 name == 'relu5_1'
            if name == 'relu5_1':
                break

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

        # 归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失

        Args:
            x: 输入图像
            y: 目标图像

        Returns:
            感知损失值
        """
        # 归一化
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            if isinstance(layer, nn.ReLU):
                loss += F.mse_loss(x, y)

        return loss


class FocusStyleGANTrainer:
    """Focus-StyleGAN训练器"""

    def __init__(
        self,
        model: FocusStyleGAN,
        dataset: MVTecADDataset,
        config: Config,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        初始化训练器

        Args:
            model: 模型
            dataset: 数据集
            config: 配置
            device: 设备
            logger: 日志记录器
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger

        # 划分数据集
        self.train_dataset, self.val_dataset = dataset.split_dataset(
            train_ratio=config.data.train_split
        )

        # 创建数据加载器
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )

        # 初始化优化器
        self.g_optimizer = optim.Adam(
            list(model.defect_branch.parameters()) +
            list(model.background_branch.parameters()) +
            list(model.fusion_module.parameters()),
            lr=config.model.generator.lr,
            betas=(config.model.generator.beta1, config.model.generator.beta2)
        )
        self.d_optimizer = optim.Adam(
            model.discriminator.parameters(),
            lr=config.model.discriminator.lr,
            betas=(config.model.discriminator.beta1, config.model.discriminator.beta2)
        )

        # 损失函数
        self.perceptual_loss = PerceptualLoss().to(device)
        # LPIPS损失：尝试从本地加载权重
        from model.utils.weights import find_weight
        vgg16_path = find_weight("vgg16")
        if vgg16_path is not None:
            try:
                self.lpips_loss = lpips.LPIPS(net='vgg', model_path=vgg16_path).to(device)
            except TypeError:
                self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        else:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.reconstruction_loss = nn.L1Loss()

        # 训练状态
        self.current_epoch = 0
        self.best_fid = float('inf')
        self.train_history = {
            'g_loss': [],
            'd_loss': [],
            'perceptual_loss': [],
            'reconstruction_loss': [],
            'fid': [],
            'is_score': [],
            'lpips': []
        }

        # 检查点目录
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def compute_gradient_penalty(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """
        计算梯度惩罚（WGAN-GP）

        Args:
            real_images: 真实图像
            fake_images: 生成图像

        Returns:
            梯度惩罚值
        """
        batch_size = real_images.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(self.device)

        # 插值
        interpolates = epsilon * real_images + (1 - epsilon) * fake_images
        interpolates.requires_grad_(True)

        # 调试信息
        self.logger.debug(f"compute_gradient_penalty: interpolates.requires_grad = {interpolates.requires_grad}")
        self.logger.debug(f"compute_gradient_penalty: real_images.requires_grad = {real_images.requires_grad}")
        self.logger.debug(f"compute_gradient_penalty: fake_images.requires_grad = {fake_images.requires_grad}")

        # 判别器输出
        d_interpolates, _ = self.model.discriminator(interpolates)
        self.logger.debug(f"compute_gradient_penalty: d_interpolates.requires_grad = {d_interpolates.requires_grad}")

        # 计算梯度
        create_graph = self.model.training  # 仅在训练时创建计算图
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            训练统计信息
        """
        self.model.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_reconstruction_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            real_images = batch['image'].to(self.device)
            batch_size = real_images.size(0)

            # 生成随机潜在向量
            z_defect = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)
            z_background = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)

            # ====================
            # 训练判别器
            # ====================
            self.d_optimizer.zero_grad()

            # 前向传播
            outputs = self.model(real_images, z_defect, z_background)
            fake_images = outputs['fused_images']

            # WGAN损失
            d_real_loss = -outputs['d_real'].mean()
            d_fake_loss = outputs['d_fake'].mean()
            wgan_loss = d_real_loss + d_fake_loss

            # 梯度惩罚
            gradient_penalty = self.compute_gradient_penalty(real_images, fake_images)
            gradient_penalty = self.config.training.lambda_gp * gradient_penalty

            # 总判别器损失
            d_loss = wgan_loss + gradient_penalty

            # 反向传播
            d_loss.backward()
            self.d_optimizer.step()

            # ====================
            # 训练生成器（每n_critic次迭代）
            # ====================
            if batch_idx % self.config.training.n_critic == 0:
                self.g_optimizer.zero_grad()

                # 重新前向传播（需要新的计算图）
                outputs = self.model(real_images, z_defect, z_background)
                fake_images = outputs['fused_images']

                # WGAN生成器损失
                d_fake_for_g, _ = self.model.discriminator(fake_images)
                g_wgan_loss = -d_fake_for_g.mean()

                # 感知损失
                perceptual = self.perceptual_loss(fake_images, real_images)
                perceptual_loss = self.config.training.lambda_perceptual * perceptual

                # 重构损失（缺陷图像与背景图像的差异）
                defect_images = outputs['defect_images']
                background_images = outputs['background_images']
                # 上采样缺陷图像以匹配背景图像尺寸
                if defect_images.shape[2:] != background_images.shape[2:]:
                    defect_images = F.interpolate(defect_images, size=background_images.shape[2:], mode='bilinear', align_corners=True)
                reconstruction = self.reconstruction_loss(defect_images, background_images)
                reconstruction_loss = self.config.training.lambda_reconstruction * reconstruction

                # LPIPS损失
                lpips_loss_value = self.lpips_loss(fake_images, real_images).mean()
                lpips_weighted = self.config.training.lambda_lpips * lpips_loss_value

                # 总生成器损失
                g_loss = g_wgan_loss + perceptual_loss + reconstruction_loss + lpips_weighted

                # 反向传播
                g_loss.backward()
                self.g_optimizer.step()

                # 记录损失
                epoch_g_loss += g_loss.item()
                epoch_perceptual_loss += perceptual_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            # 记录判别器损失
            epoch_d_loss += d_loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'g_loss': g_loss.item() if batch_idx % self.config.training.n_critic == 0 else 0,
                'd_loss': d_loss.item(),
                'gp': gradient_penalty.item()
            })

        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_g_loss = epoch_g_loss / max(num_batches // self.config.training.n_critic, 1)
        avg_d_loss = epoch_d_loss / num_batches
        avg_perceptual_loss = epoch_perceptual_loss / max(num_batches // self.config.training.n_critic, 1)
        avg_reconstruction_loss = epoch_reconstruction_loss / max(num_batches // self.config.training.n_critic, 1)

        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'perceptual_loss': avg_perceptual_loss,
            'reconstruction_loss': avg_reconstruction_loss
        }

    def validate(self) -> Dict[str, float]:
        """
        验证

        Returns:
            验证指标
        """
        self.model.eval()
        val_losses = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'perceptual_loss': 0.0,
            'reconstruction_loss': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                real_images = batch['image'].to(self.device)
                batch_size = real_images.size(0)

                # 生成随机潜在向量
                z_defect = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)
                z_background = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)

                # 前向传播
                outputs = self.model(real_images, z_defect, z_background)

                # 计算损失
                d_real_loss = -outputs['d_real'].mean()
                d_fake_loss = outputs['d_fake'].mean()
                wgan_loss = d_real_loss + d_fake_loss

                fake_images = outputs['fused_images']
                with torch.enable_grad():
                    gradient_penalty = self.compute_gradient_penalty(real_images, fake_images)
                gradient_penalty = gradient_penalty.detach()
                d_loss = wgan_loss + self.config.training.lambda_gp * gradient_penalty

                d_fake_for_g, _ = self.model.discriminator(fake_images)
                g_wgan_loss = -d_fake_for_g.mean()

                perceptual = self.perceptual_loss(fake_images, real_images)
                perceptual_loss = self.config.training.lambda_perceptual * perceptual

                # 上采样缺陷图像以匹配背景图像尺寸
                defect_images = outputs['defect_images']
                background_images = outputs['background_images']
                if defect_images.shape[2:] != background_images.shape[2:]:
                    defect_images = F.interpolate(defect_images, size=background_images.shape[2:], mode='bilinear', align_corners=True)
                reconstruction = self.reconstruction_loss(defect_images, background_images)
                reconstruction_loss = self.config.training.lambda_reconstruction * reconstruction

                lpips_loss_value = self.lpips_loss(fake_images, real_images).mean()

                g_loss = g_wgan_loss + perceptual_loss + reconstruction_loss + lpips_loss_value

                # 累计损失
                val_losses['g_loss'] += g_loss.item()
                val_losses['d_loss'] += d_loss.item()
                val_losses['perceptual_loss'] += perceptual_loss.item()
                val_losses['reconstruction_loss'] += reconstruction_loss.item()

        # 计算平均损失
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def train(self, n_epochs: int = None, resume_from: str = None):
        """
        训练模型

        Args:
            n_epochs: 训练轮数，如果为None则使用配置中的值
            resume_from: 恢复训练的检查点路径
        """
        if n_epochs is None:
            n_epochs = self.config.training.epochs

        # 恢复训练
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        self.logger.info(f"开始训练，共{n_epochs}个epoch")
        self.logger.info(f"训练集大小: {len(self.train_dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_dataset)}")

        start_time = time.time()

        for epoch in range(self.current_epoch, n_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_stats = self.train_epoch()

            # 验证
            val_stats = self.validate()

            # 记录历史
            for key in train_stats:
                if key in self.train_history:
                    self.train_history[key].append(train_stats[key])

            # 打印统计信息
            self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            self.logger.info(f"  训练 - G损失: {train_stats['g_loss']:.4f}, D损失: {train_stats['d_loss']:.4f}")
            self.logger.info(f"  验证 - G损失: {val_stats['g_loss']:.4f}, D损失: {val_stats['d_loss']:.4f}")

            # 定期保存检查点
            if (epoch + 1) % self.config.training.save_interval == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"保存检查点到: {checkpoint_path}")

            # 保存最佳模型
            if val_stats['g_loss'] < self.best_fid:
                self.best_fid = val_stats['g_loss']
                best_model_path = self.checkpoint_dir / "best_model.pth"
                self.save_checkpoint(best_model_path, is_best=True)
                self.logger.info(f"新的最佳模型，保存到: {best_model_path}")

        # 训练完成
        training_time = time.time() - start_time
        self.logger.info(f"训练完成! 总时间: {training_time:.2f}秒")

        # 保存最终模型
        final_model_path = self.checkpoint_dir / "final_model.pth"
        self.save_checkpoint(final_model_path)
        self.logger.info(f"保存最终模型到: {final_model_path}")

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        保存检查点

        Args:
            path: 保存路径
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_fid': self.best_fid,
            'train_history': self.train_history,
            'config': self.config._config,
            'is_best': is_best
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"检查点不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        # 加载训练状态
        self.current_epoch = checkpoint['epoch']
        self.best_fid = checkpoint['best_fid']
        self.train_history = checkpoint['train_history']

        self.logger.info(f"从检查点恢复: {path}")
        self.logger.info(f"恢复epoch: {self.current_epoch}, 最佳FID: {self.best_fid:.4f}")


class OptunaOptimizer:
    """Optuna超参数优化器"""

    def __init__(
        self,
        config: Config,
        dataset: MVTecADDataset,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        初始化Optuna优化器

        Args:
            config: 配置
            dataset: 数据集
            device: 设备
            logger: 日志记录器
        """
        self.config = config
        self.dataset = dataset
        self.device = device
        self.logger = logger

        # 创建Optuna study
        self.study = optuna.create_study(
            study_name=config.optuna.study_name,
            storage=config.optuna.storage,
            direction=config.optuna.direction,
            load_if_exists=True
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数

        Args:
            trial: Optuna试验

        Returns:
            目标值（FID）
        """
        # 定义超参数空间
        g_lr = trial.suggest_float('g_lr', 1e-5, 1e-3, log=True)
        d_lr = trial.suggest_float('d_lr', 1e-5, 1e-3, log=True)
        lambda_gp = trial.suggest_float('lambda_gp', 1.0, 20.0)
        lambda_perceptual = trial.suggest_float('lambda_perceptual', 0.01, 1.0, log=True)
        lambda_reconstruction = trial.suggest_float('lambda_reconstruction', 1.0, 20.0)

        # 更新配置
        self.config.update('model.generator.lr', g_lr)
        self.config.update('model.discriminator.lr', d_lr)
        self.config.update('training.lambda_gp', lambda_gp)
        self.config.update('training.lambda_perceptual', lambda_perceptual)
        self.config.update('training.lambda_reconstruction', lambda_reconstruction)

        # 创建模型
        model = FocusStyleGAN(config=self.config.model).to(self.device)

        # 创建训练器
        trainer = FocusStyleGANTrainer(
            model=model,
            dataset=self.dataset,
            config=self.config,
            device=self.device,
            logger=self.logger
        )

        # 训练少量epoch进行评估
        n_epochs = 10  # 快速评估
        best_fid = float('inf')

        for epoch in range(n_epochs):
            # 训练一个epoch
            trainer.train_epoch()

            # 验证
            val_stats = trainer.validate()
            fid = val_stats['g_loss']  # 使用生成器损失作为代理指标

            # 报告中间值
            trial.report(fid, epoch)

            # 提前停止
            if trial.should_prune():
                raise optuna.TrialPruned()

            # 更新最佳FID
            if fid < best_fid:
                best_fid = fid

        return best_fid

    def optimize(self):
        """执行超参数优化"""
        self.logger.info("开始超参数优化...")
        self.logger.info(f"试验次数: {self.config.optuna.n_trials}")

        try:
            self.study.optimize(
                self.objective,
                n_trials=self.config.optuna.n_trials,
                timeout=self.config.optuna.timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            self.logger.info("优化被中断")

        # 输出最佳超参数
        self.logger.info("优化完成!")
        self.logger.info(f"最佳试验: {self.study.best_trial.number}")
        self.logger.info(f"最佳值 (FID): {self.study.best_value:.4f}")
        self.logger.info("最佳超参数:")
        for key, value in self.study.best_params.items():
            self.logger.info(f"  {key}: {value}")

        # 保存最佳配置
        best_config_path = self.config.config_path.parent / "best_config.yaml"
        self.config.save(best_config_path)
        self.logger.info(f"保存最佳配置到: {best_config_path}")


def create_trainer_from_config(config_path: str, device: torch.device, logger: logging.Logger):
    """
    从配置创建训练器

    Args:
        config_path: 配置文件路径
        device: 设备
        logger: 日志记录器

    Returns:
        训练器实例
    """
    # 加载配置
    config = Config(config_path)

    # 加载数据集
    dataset = MVTecADDataset(
        root_path=config.data.dataset_path,
        image_size=config.data.image_size,
        augment=config.data.augmentations.use
    )

    # 创建模型
    model = FocusStyleGAN(config=config.model).to(device)

    # 创建训练器
    trainer = FocusStyleGANTrainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        logger=logger
    )

    return trainer