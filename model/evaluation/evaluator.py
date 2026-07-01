#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器模块
计算FID、IS、LPIPS等指标
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from scipy import linalg
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
# 可选导入
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None
# 可选导入
try:
    from pytorch_fid import fid_score
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    fid_score = None

from model.models.focus_stylegan import FocusStyleGAN
from model.data.loader import MVTecADDataset
from model.utils.config import Config


class InceptionScore:
    """Inception Score计算器"""

    def __init__(self, device: torch.device):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        # 优先从本地加载InceptionV3权重
        from model.utils.weights import find_weight, load_state_dict_from_local
        try:
            inception = models.inception_v3(pretrained=False, transform_input=False)
            state_dict = load_state_dict_from_local("inception_v3", map_location="cpu")
            inception.load_state_dict(state_dict)
            self.inception = inception.to(device)
        except (FileNotFoundError, RuntimeError):
            self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        self.inception.fc = nn.Identity()

        for param in self.inception.parameters():
            param.requires_grad = False

    def compute_score(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """
        计算Inception Score

        Args:
            images: 图像张量 [N, 3, H, W]，范围[-1, 1]
            splits: 分割数

        Returns:
            (IS均值, IS标准差)
        """
        n_images = images.shape[0]

        # 预处理图像到[0, 1]
        images = (images + 1) / 2  # 从[-1, 1]到[0, 1]

        # 调整到Inception输入尺寸
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # 获取预测概率
        preds = []
        with torch.no_grad():
            for i in range(0, n_images, 32):
                batch = images[i:i + 32].to(self.device)
                pred = self.inception(batch)
                preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0)
        preds = F.softmax(preds, dim=1)

        # 计算IS
        scores = []
        for k in range(splits):
            part = preds[k * (n_images // splits): (k + 1) * (n_images // splits), :]
            py = part.mean(0)
            scores.append((part * (part.log() - py.log())).sum(1).mean().exp())

        scores = torch.stack(scores)
        return scores.mean().item(), scores.std().item()


class FIDScore:
    """FID计算器"""

    def __init__(self, device: torch.device):
        """
        Args:
            device: 计算设备
        """
        self.device = device
        # 优先从本地加载InceptionV3权重
        from model.utils.weights import find_weight, load_state_dict_from_local
        try:
            inception = models.inception_v3(pretrained=False, transform_input=False)
            state_dict = load_state_dict_from_local("inception_v3", map_location="cpu")
            inception.load_state_dict(state_dict)
            self.inception = inception.to(device)
        except (FileNotFoundError, RuntimeError):
            self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        self.inception.fc = nn.Identity()

        for param in self.inception.parameters():
            param.requires_grad = False

        # 获取中间层特征
        # 尝试注册钩子到合适的特征提取层
        hook_registered = False

        # 尝试常见的InceptionV3层名
        layer_candidates = ['Mixed_5c', 'Mixed_7c', 'avgpool']

        for layer_name in layer_candidates:
            try:
                layer = getattr(self.inception, layer_name)
                layer.register_forward_hook(self._hook_fn)
                hook_registered = True
                print(f"成功注册钩子到层: {layer_name}")
                break
            except (AttributeError, KeyError):
                continue

        # 如果常见层名失败，尝试数字索引
        if not hook_registered:
            for idx in range(10):  # 尝试索引0-9
                try:
                    self.inception._modules[str(idx)].register_forward_hook(self._hook_fn)
                    hook_registered = True
                    print(f"成功注册钩子到数字索引: {idx}")
                    break
                except KeyError:
                    continue

        # 如果所有尝试都失败，打印错误信息
        if not hook_registered:
            print(f"错误: 无法注册InceptionV3前向钩子")
            print(f"可用模块键: {list(self.inception._modules.keys())}")
            raise RuntimeError("无法注册InceptionV3特征提取钩子")

        # 统计量累加变量
        self.feature_sum = None
        self.feature_sum_sq = None
        self.feature_count = 0
        self.current_features = []  # 当前批次特征

    def _hook_fn(self, module, input, output):
        """钩子函数获取特征"""
        # 展平特征
        flattened = output.view(output.shape[0], -1).cpu()
        self.current_features.append(flattened)

    def _reset_batch_features(self):
        """重置当前批次特征"""
        self.current_features = []

    def _get_batch_features(self):
        """获取当前批次特征并重置"""
        if not self.current_features:
            return None
        features = torch.cat(self.current_features, dim=0)
        self._reset_batch_features()
        return features.numpy()

    def _update_statistics(self, batch_features: np.ndarray):
        """使用当前批次特征更新统计量"""
        if batch_features is None or len(batch_features) == 0:
            return

        k = len(batch_features)  # 当前批次样本数

        if self.feature_sum is None:
            # 初始化统计量
            self.feature_sum = np.sum(batch_features, axis=0)
            self.feature_sum_sq = np.sum(batch_features**2, axis=0)
            self.feature_count = k
        else:
            # 更新统计量
            self.feature_sum += np.sum(batch_features, axis=0)
            self.feature_sum_sq += np.sum(batch_features**2, axis=0)
            self.feature_count += k

    def _compute_statistics_from_accumulators(self):
        """从累加器计算均值和方差（返回方差向量而非完整协方差矩阵）"""
        if self.feature_count == 0:
            return None, None

        # 计算均值
        mu = self.feature_sum / self.feature_count

        # 计算方差（对角协方差）
        expectation_x_sq = self.feature_sum_sq / self.feature_count
        variance = expectation_x_sq - mu**2

        # 确保方差为正
        variance = np.maximum(variance, 1e-12)

        return mu, variance

    def compute_statistics(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像特征的统计量（均值和协方差）
        使用流式处理避免内存爆炸

        Args:
            images: 图像张量 [N, 3, H, W]

        Returns:
            (均值, 协方差矩阵)
        """
        # 重置统计量
        self.feature_sum = None
        self.feature_sum_sq = None
        self.feature_count = 0

        # 预处理
        images = (images + 1) / 2  # 从[-1, 1]到[0, 1]
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # 小批次处理
        batch_size = 8  # 进一步减小批次大小
        n_images = images.shape[0]

        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                end_idx = min(i + batch_size, n_images)
                batch = images[i:end_idx].to(self.device)

                # 重置当前批次特征
                self._reset_batch_features()

                # 前向传播（会触发钩子）
                _ = self.inception(batch)

                # 获取当前批次特征
                batch_features = self._get_batch_features()

                # 更新统计量
                self._update_statistics(batch_features)

                # 清理内存
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 打印进度
                if (i // batch_size) % 10 == 0:
                    print(f"  处理进度: {end_idx}/{n_images}")

        # 计算最终统计量
        mu, sigma = self._compute_statistics_from_accumulators()

        # 重置统计量
        self.feature_sum = None
        self.feature_sum_sq = None
        self.feature_count = 0

        return mu, sigma

    def compute_score(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        计算FID（使用对角协方差近似）

        Args:
            real_images: 真实图像
            fake_images: 生成图像

        Returns:
            FID值
        """
        print("计算真实图像统计量...")
        mu_real, var_real = self.compute_statistics(real_images)

        print("计算生成图像统计量...")
        mu_fake, var_fake = self.compute_statistics(fake_images)

        # 计算对角FID
        diff = mu_real - mu_fake
        covmean = np.sqrt(var_real * var_fake)  # 对角协方差矩阵的平方根

        fid = diff.dot(diff) + np.sum(var_real + var_fake - 2 * covmean)

        print(f"FID计算完成: 均值差范数={diff.dot(diff):.4f}, 协方差项={np.sum(var_real + var_fake - 2 * covmean):.4f}")

        return fid


class Evaluator:
    """模型评估器"""

    def __init__(
        self,
        model: FocusStyleGAN,
        dataset: MVTecADDataset,
        config: Config,
        device: torch.device,
        logger: logging.Logger
    ):
        """
        初始化评估器

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

        # 创建数据加载器
        self.loader = DataLoader(
            dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            num_workers=4
        )

        # 评估指标
        self.metrics = {}

        # FID评分
        if PYTORCH_FID_AVAILABLE:
            self.metrics['fid'] = FIDScore(device)
        else:
            self.logger.warning("pytorch_fid未安装，FID指标不可用")

        # Inception Score
        self.metrics['is'] = InceptionScore(device)

        # LPIPS
        if LPIPS_AVAILABLE and lpips is not None:
            from model.utils.weights import find_weight
            vgg16_path = find_weight("vgg16")
            if vgg16_path is not None:
                try:
                    self.metrics['lpips'] = lpips.LPIPS(net='vgg', model_path=vgg16_path).to(device)
                except TypeError:
                    self.metrics['lpips'] = lpips.LPIPS(net='vgg').to(device)
            else:
                self.metrics['lpips'] = lpips.LPIPS(net='vgg').to(device)
            # 设置为评估模式，不计算梯度
            self.metrics['lpips'].eval()
            for param in self.metrics['lpips'].parameters():
                param.requires_grad = False
        else:
            self.logger.warning("lpips未安装，LPIPS指标不可用")

    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """
        生成样本

        Args:
            n_samples: 样本数量

        Returns:
            生成的图像张量
        """
        self.model.eval()
        generated_images = []

        with torch.no_grad():
            # 分批生成
            for _ in range(0, n_samples, self.config.evaluation.batch_size):
                batch_size = min(self.config.evaluation.batch_size, n_samples - len(generated_images))

                # 生成随机潜在向量
                z_defect = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)
                z_background = torch.randn(batch_size, self.config.model.generator.latent_dim).to(self.device)

                # 生成图像
                images = self.model.generate(z_defect, z_background)
                generated_images.append(images.cpu())

        return torch.cat(generated_images, dim=0)

    def get_real_samples(self, n_samples: int) -> torch.Tensor:
        """
        获取真实样本

        Args:
            n_samples: 样本数量

        Returns:
            真实图像张量
        """
        real_images = []
        count = 0

        for batch in self.loader:
            images = batch['image']
            real_images.append(images)

            count += images.shape[0]
            if count >= n_samples:
                break

        real_images = torch.cat(real_images, dim=0)[:n_samples]
        return real_images

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估

        Returns:
            评估指标字典
        """
        self.logger.info("开始模型评估...")

        n_samples = min(self.config.evaluation.n_samples, len(self.dataset))
        self.logger.info(f"评估样本数: {n_samples}")

        # 生成样本
        self.logger.info("生成样本...")
        generated_images = self.generate_samples(n_samples)

        # 获取真实样本
        self.logger.info("加载真实样本...")
        real_images = self.get_real_samples(n_samples)

        # 确保样本数量一致
        min_samples = min(generated_images.shape[0], real_images.shape[0])
        generated_images = generated_images[:min_samples]
        real_images = real_images[:min_samples]

        self.logger.info(f"实际评估样本数: {min_samples}")

        results = {}

        # 计算FID
        if 'fid' in self.config.evaluation.metrics and 'fid' in self.metrics:
            self.logger.info("计算FID...")
            fid_value = self.metrics['fid'].compute_score(real_images, generated_images)
            results['fid'] = fid_value
            self.logger.info(f"FID: {fid_value:.4f}")
        elif 'fid' in self.config.evaluation.metrics:
            self.logger.warning("FID指标配置但不可用（需要安装pytorch_fid）")

        # 计算Inception Score
        if 'is' in self.config.evaluation.metrics:
            self.logger.info("计算Inception Score...")
            is_mean, is_std = self.metrics['is'].compute_score(generated_images)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            self.logger.info(f"IS: {is_mean:.4f} ± {is_std:.4f}")

        # 计算LPIPS
        if 'lpips' in self.config.evaluation.metrics and 'lpips' in self.metrics:
            self.logger.info("计算LPIPS...")
            lpips_values = []

            # 分批计算
            batch_size = self.config.evaluation.batch_size
            with torch.no_grad():
                for i in range(0, min_samples, batch_size):
                    real_batch = real_images[i:i + batch_size].to(self.device)
                    fake_batch = generated_images[i:i + batch_size].to(self.device)

                    lpips_batch = self.metrics['lpips'](real_batch, fake_batch)
                    lpips_values.extend(lpips_batch.cpu().detach().numpy())

            lpips_mean = np.mean(lpips_values)
            lpips_std = np.std(lpips_values)
            results['lpips_mean'] = lpips_mean
            results['lpips_std'] = lpips_std
            self.logger.info(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")
        elif 'lpips' in self.config.evaluation.metrics:
            self.logger.warning("LPIPS指标配置但不可用（需要安装lpips）")

        # 计算PSNR和SSIM
        if 'psnr' in self.config.evaluation.metrics or 'ssim' in self.config.evaluation.metrics:
            self.logger.info("计算PSNR和SSIM...")
            psnr_values = []
            ssim_values = []

            for i in range(min_samples):
                real_img = real_images[i].unsqueeze(0).to(self.device)
                fake_img = generated_images[i].unsqueeze(0).to(self.device)

                # PSNR
                mse = F.mse_loss(real_img, fake_img)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                psnr_values.append(psnr.item())

                # SSIM
                ssim_val = self._compute_ssim(real_img, fake_img)
                ssim_values.append(ssim_val)

            if 'psnr' in self.config.evaluation.metrics:
                results['psnr_mean'] = np.mean(psnr_values)
                results['psnr_std'] = np.std(psnr_values)
                self.logger.info(f"PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}")

            if 'ssim' in self.config.evaluation.metrics:
                results['ssim_mean'] = np.mean(ssim_values)
                results['ssim_std'] = np.std(ssim_values)
                self.logger.info(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

        # 计算PPS (Physical Plausibility Score)
        if 'pps' in self.config.evaluation.metrics:
            self.logger.info("计算PPS...")
            try:
                from model.evaluation.pps import PhysicalPlausibilityScore
                pps_calculator = PhysicalPlausibilityScore()
                pps_results = pps_calculator.compute(generated_images, real_images)
                results['pps'] = pps_results['pps']
                results['pps_s_geo'] = pps_results['s_geo']
                results['pps_s_illum'] = pps_results['s_illum']
                self.logger.info(f"PPS: {results['pps']:.4f} (S_geo={results['pps_s_geo']:.4f}, S_illum={results['pps_s_illum']:.4f})")
            except Exception as e:
                self.logger.warning(f"PPS计算失败: {e}")
                results['pps'] = float('nan')

        self.logger.info("评估完成!")
        return results

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """
        计算SSIM

        Args:
            img1: 图像1
            img2: 图像2
            window_size: 窗口大小

        Returns:
            SSIM值
        """
        # 从[-1, 1]转换到[0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # 创建高斯窗口
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                  for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        channel = img1.shape[1]
        window = create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def ablation_study(self) -> Dict[str, Dict[str, float]]:
        """
        消融实验

        Returns:
            消融实验结果
        """
        self.logger.info("开始消融实验...")

        ablation_results = {}

        # 基准模型
        self.logger.info("基准模型...")
        baseline_results = self.evaluate()
        ablation_results['baseline'] = baseline_results

        # 无AdaIN
        self.logger.info("无AdaIN...")
        original_adain = self.model.defect_branch.blocks[0].use_adain
        for block in self.model.defect_branch.blocks:
            block.use_adain = False
        for block in self.model.background_branch.decoder_blocks:
            block.use_adain = False

        no_adain_results = self.evaluate()
        ablation_results['no_adain'] = no_adain_results

        # 恢复AdaIN
        for block in self.model.defect_branch.blocks:
            block.use_adain = original_adain
        for block in self.model.background_branch.decoder_blocks:
            block.use_adain = original_adain

        # 无CBAM
        self.logger.info("无CBAM...")
        original_cbam = self.model.defect_branch.blocks[0].use_cbam
        for block in self.model.defect_branch.blocks:
            block.use_cbam = False
        for block in self.model.background_branch.decoder_blocks:
            block.use_cbam = False

        no_cbam_results = self.evaluate()
        ablation_results['no_cbam'] = no_cbam_results

        # 恢复CBAM
        for block in self.model.defect_branch.blocks:
            block.use_cbam = original_cbam
        for block in self.model.background_branch.decoder_blocks:
            block.use_cbam = original_cbam

        # 无感知损失
        self.logger.info("训练无感知损失的模型...")
        # 注意：这需要重新训练模型，这里简化处理
        ablation_results['no_perceptual'] = {
            'note': '需要重新训练模型'
        }

        self.logger.info("消融实验完成!")
        return ablation_results