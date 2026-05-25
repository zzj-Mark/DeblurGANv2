# 基于DeblurGAN-v2的图像去模糊实验报告

## 一、实验目的

图像模糊是计算机视觉领域常见的退化问题，广泛存在于日常摄影、安防监控、自动驾驶等场景中。运动模糊（Motion Blur）和相机抖动（Camera Shake）是造成图像退化的两大主要原因。传统的图像去模糊方法通常基于能量函数优化或统计先验建模，计算量大且效果有限。近年来，基于深度学习的方法在图像去模糊任务中取得了显著突破。

本实验旨在：

1. 理解并实践基于生成对抗网络（GAN）的图像去模糊方法；
2. 掌握DeblurGAN-v2模型的核心架构设计，包括特征金字塔网络（FPN）与Inception-ResNet-v2骨干网络的结合；
3. 对GOPRO数据集中的模糊图像进行去模糊处理，并对结果进行后处理增强；
4. 通过实验结果分析，评估深度学习方法在图像去模糊任务中的有效性和局限性。

## 二、实验原理

### 2.1 生成对抗网络（GAN）基础

生成对抗网络由Goodfellow等人于2014年提出，包含两个相互博弈的神经网络：生成器（Generator, G）和判别器（Discriminator, D）。生成器的目标是生成足以欺骗判别器的逼真图像，判别器的目标则是区分真实图像和生成图像。两者的训练过程可表示为以下极小极大博弈：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

其中 $x$ 为真实数据样本，$z$ 为随机噪声向量。当训练达到纳什均衡时，生成器能够产生与真实数据分布一致的样本。

### 2.2 图像到图像翻译与条件GAN

图像去模糊本质上是一个图像到图像翻译（Image-to-Image Translation）问题。条件GAN（cGAN）通过引入条件信息 $y$（在去模糊任务中即模糊图像），将随机噪声输入替换为条件输入：

$$G: I_{blur} \rightarrow I_{sharp}$$

判别器的任务变为区分"真实清晰图像"和"由模糊图像生成的去模糊图像"。

### 2.3 DeblurGAN-v2 模型架构

DeblurGAN-v2（Kupyn等, ICCV 2019）是对第一代DeblurGAN的重要改进，核心创新在于将特征金字塔网络（Feature Pyramid Network, FPN）引入去模糊任务，并提供了多种骨干网络选择。

#### 2.3.1 生成器架构：FPN + Inception-ResNet-v2

本实验采用的生成器为FPNInception，其结构包含以下关键组件：

**（1）骨干网络（Backbone）**

采用Inception-ResNet-v2作为特征提取骨干网络。该网络通过Inception模块的并行多尺度卷积与残差连接的结合，能够提取丰富的多尺度特征表示。骨干网络将输入图像逐层编码为五个不同分辨率的特征图：

| 编码层 | 输出分辨率 | 通道数 |
|--------|-----------|--------|
| enc0   | 1/2       | 32     |
| enc1   | 1/4       | 64     |
| enc2   | 1/8       | 192    |
| enc3   | 1/16      | 1088   |
| enc4   | 1/32      | 2080   |

**（2）特征金字塔网络（FPN）**

FPN通过自顶向下的路径和横向连接（Lateral Connection）融合多尺度特征：

- **自底向上路径**：通过骨干网络逐层提取特征，分辨率递减，语义信息递增；
- **自顶向下路径**：将高层语义特征逐层上采样，与底层细节特征融合；
- **横向连接**：使用 $1 \times 1$ 卷积将不同通道数的特征映射到统一维度（256通道），再进行逐元素相加。

FPN的数学表达为：

$$P_i = \text{Conv}_{td}(L_i + \text{Upsample}(P_{i+1}))$$

其中 $L_i$ 为第 $i$ 层的横向连接输出，$P_i$ 为融合后的特征金字塔层级特征。

**（3）分割头与特征融合**

每个金字塔层级配备独立的分割头（FPNHead），由两个 $3 \times 3$ 卷积层和ReLU激活函数组成。四层特征图分别上采样到统一分辨率后在通道维度拼接：

$$F_{fused} = \text{Concat}(\text{Upsample}(H_4(P_4), \times 8), \text{Upsample}(H_3(P_3), \times 4), \text{Upsample}(H_2(P_2), \times 2), H_1(P_1))$$

融合后的特征经过平滑卷积层和最终卷积层，输出3通道的残差图像。

**（4）残差学习策略**

模型采用残差学习（Residual Learning）策略，最终输出为：

$$I_{deblur} = \tanh(\text{Conv}_{final}(F_{smooth})) + I_{blur}$$

即网络学习模糊图像到清晰图像的残差（差异），而非直接生成完整图像。这种策略大幅降低了学习难度，加速了训练收敛。

#### 2.3.2 判别器架构：Double-Scale GAN

DeblurGAN-v2采用双尺度判别器（Double-Scale Discriminator）策略：

- **PatchGAN判别器**：在图像局部区域（patch）级别判断真实性，关注局部纹理细节的恢复质量；
- **FullGAN判别器**：在全图级别判断真实性，关注全局结构和语义一致性。

两个判别器协同工作，使生成器同时关注局部细节和全局结构的恢复。

#### 2.3.3 损失函数设计

模型采用复合损失函数：

**（1）对抗损失**：使用WGAN-GP（Wasserstein GAN with Gradient Penalty），通过Wasserstein距离度量分布差异，并引入梯度惩罚提升训练稳定性：

$$\mathcal{L}_{adv} = \mathbb{E}[D(G(I_{blur}))] - \mathbb{E}[D(I_{sharp})] + \lambda_{gp} \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

**（2）内容损失**：采用感知损失（Perceptual Loss），在预训练VGG网络的高层特征空间中计算差异，使去模糊结果在感知质量上更接近真实图像：

$$\mathcal{L}_{content} = \sum_l \frac{1}{C_l H_l W_l} \|\phi_l(I_{deblur}) - \phi_l(I_{sharp})\|_2^2$$

其中 $\phi_l$ 表示VGG网络第 $l$ 层的特征提取。

**（3）总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{content} + \lambda_{adv} \cdot \mathcal{L}_{adv}$$

其中 $\lambda_{adv} = 0.001$，内容损失占据主导地位。

### 2.4 后处理增强流水线

去模糊后的图像可能存在颜色偏移、对比度不足、细节模糊等问题。本实验设计了五步后处理增强流水线进行改善：

**（1）灰度世界偏色矫正**：基于灰度世界假设，认为图像各通道均值应趋于相等，通过缩放各通道校正白平衡。

**（2）Lab空间CLAHE增强**：将图像转换到Lab色彩空间，仅对L（亮度）通道进行限制对比度自适应直方图均衡化（CLAHE），增强局部对比度同时保留全局色调。

**（3）Unsharp Masking锐化**：通过高斯模糊产生低频图像，与原图加权差分增强边缘细节：

$$I_{sharp} = \alpha \cdot I - (\alpha - 1) \cdot G_\sigma(I)$$

**（4）HSV饱和度微调**：在HSV色彩空间中对S（饱和度）通道进行5%的增强，使色彩更鲜明。

**（5）双边滤波去噪**：在保持边缘的同时平滑噪声，消除可能由去模糊过程引入的振铃效应。

## 三、实验环境

### 3.1 硬件环境

| 项目     | 配置                          |
|----------|-------------------------------|
| 操作系统 | Windows                       |
| GPU      | NVIDIA（支持CUDA加速）         |
| 深度学习框架 | PyTorch（含CUDA支持）       |

### 3.2 软件环境

| 项目         | 版本/说明                              |
|--------------|----------------------------------------|
| Python       | 虚拟环境（.venv）                      |
| PyTorch      | 深度学习框架                            |
| OpenCV       | 图像读写与处理                          |
| NumPy        | 数值计算                                |
| pretrainedmodels | Inception-ResNet-v2预训练权重加载  |
| albumentations | 图像增强库                            |
| fire         | 命令行接口                              |

### 3.3 模型与数据

| 项目           | 说明                                                    |
|----------------|---------------------------------------------------------|
| 预训练权重     | fpn_inception.h5（FPN + Inception-ResNet-v2骨干网络）   |
| 配置文件       | config/config.yaml                                       |
| 测试图像       | GOPRO数据集模糊图像 + Kohler模糊图像（共17张）           |
| 图像分辨率     | 720 × 1280 × 3                                          |

## 四、实验过程

### 4.1 项目结构与配置

实验基于DeblurGAN-v2开源项目，模型配置如下（config/config.yaml关键参数）：

```yaml
model:
  g_name: fpn_inception          # 生成器：FPN + Inception-ResNet-v2
  d_name: double_gan             # 判别器：双尺度GAN
  content_loss: perceptual       # 内容损失：感知损失
  disc_loss: wgan-gp             # 对抗损失：WGAN-GP
  learn_residual: True           # 残差学习
  norm_layer: instance           # 归一化层：Instance Normalization
  adv_lambda: 0.001              # 对抗损失权重
```

### 4.2 去模糊推理

使用预训练权重文件 `fpn_inception.h5` 对测试图像进行推理。推理流程如下：

1. **图像预处理**：将输入图像归一化到 [-1, 1] 范围，并对尺寸进行32的倍数padding；
2. **模型推理**：将预处理后的张量输入生成器网络，在前向传播中依次经过FPN特征提取、多尺度特征融合、残差输出；
3. **后处理**：将输出张量从 [-1, 1] 反归一化到 [0, 255]，裁剪到原始图像尺寸。

运行命令：

```bash
python predict.py --img_pattern "test_img/*.png" --weights_path fpn_inception.h5 --out_dir submit_deblur/
```

处理了以下测试图像：

**GOPRO数据集图像（9张）**：
- 000001.png, 000019.png, 000027.png（原始测试集）
- 000144_19.png, 000174_5.png（从GOPRO压缩包提取）
- 000201.png, 011658_19.png, 016356_21.png, 023166_7.png

**Kohler模糊图像（4张）**：
- kohler_blur_1.png, kohler_blur_2.png, kohler_blur_3.png, kohler_blur_4.png

所有去模糊结果输出到 `submit_deblur/` 目录。

### 4.3 后处理增强

对去模糊结果运行增强流水线：

```bash
python enhance_submit.py submit_deblur/ submit_enhanced/
```

每张图像依次执行5步增强处理：
1. 灰度世界偏色矫正 — 自动白平衡
2. CLAHE对比度增强（clipLimit=0.8, tileGridSize=(8,8)）
3. Unsharp Masking锐化（amount=1.1, sigma=1.5）
4. 饱和度微调（factor=1.05，即+5%）
5. 双边滤波去噪（d=5, sigmaColor=15, sigmaSpace=15）

所有增强结果输出到 `submit_enhanced/` 目录。

## 五、实验结果与分析

### 5.1 定性分析

通过对比原始模糊图像、去模糊输出和增强后输出，可以观察到以下结果：

**（1）运动模糊消除效果**

对于GOPRO数据集中的运动模糊图像，DeblurGAN-v2能够有效消除由相机快速运动或物体运动产生的模糊条纹。去模糊后的图像中，建筑物轮廓、文字边缘和人物细节得到了明显恢复。

**（2）纹理恢复**

FPN的多尺度特征融合机制使模型能够在不同尺度上恢复图像细节。大尺度的全局结构由高层语义特征恢复，小尺度的局部纹理由底层细节特征补充。残差学习策略确保了输出图像与输入图像在颜色和亮度上的一致性。

**（3）后处理增强效果**

五步增强流水线对去模糊结果有进一步改善：
- 偏色矫正修正了部分图像中存在的轻微色偏；
- CLAHE增强了图像的局部对比度，使暗部细节更加清晰；
- Unsharp Masking使边缘更加锐利；
- 饱和度微调使色彩更接近自然；
- 双边滤波在保留边缘的同时减少了振铃效应。

### 5.2 模型特性分析

**（1）推理效率**

DeblurGAN-v2相比第一代DeblurGAN，在推理速度上有数量级的提升。这主要归功于：
- FPN架构替代了原来的ResNet生成器，计算路径更高效；
- 残差学习减少了网络需要学习的信息量；
- 推理时使用train模式以获取归一化层的真实统计量（而非批统计量）。

**（2）Inception-ResNet-v2骨干网络的作用**

Inception-ResNet-v2通过Inception模块的并行多尺度卷积（1×1、3×3、5×5等）提取不同感受野的特征，残差连接保证了梯度流动的畅通。该骨干网络在ImageNet上的预训练知识为去模糊任务提供了良好的特征初始化。

**（3）双尺度判别器的优势**

PatchGAN关注局部纹理真实性，适合评估去模糊后的细节恢复质量；FullGAN关注全局结构一致性，避免产生全局失真。两者的协同约束使生成器在局部细节和全局结构上同时达到较好的效果。

### 5.3 局限性观察

1. 对于极端严重的模糊（如长曝光造成的严重拖影），去模糊效果有限，可能产生伪影；
2. 去模糊结果在部分区域可能出现过度平滑，丢失部分高频纹理细节；
3. 后处理增强的参数为通用设置，对特定图像可能不是最优，需要根据具体场景调整。

## 六、实验总结

本实验基于DeblurGAN-v2实现了图像去模糊的完整流程，包括模型推理和后处理增强两个阶段。主要成果和收获如下：

1. **深入理解了GAN在图像恢复任务中的应用**：通过分析DeblurGAN-v2的生成器-判别器架构、损失函数设计和训练策略，掌握了条件GAN在图像到图像翻译任务中的应用方法。

2. **掌握了FPN在图像恢复中的作用机制**：FPN通过自顶向下的特征融合路径，有效结合了高层语义信息和底层细节信息，为图像恢复提供了多尺度的特征表示。这一机制不仅适用于目标检测，同样适用于图像恢复任务。

3. **理解了残差学习在图像恢复中的重要性**：残差学习策略使网络只需学习模糊图像与清晰图像之间的差异，大幅降低了学习难度，提升了训练效率和恢复质量。

4. **实践了图像后处理的系统工程方法**：通过设计五步增强流水线，将颜色校正、对比度增强、锐化和去噪等传统图像处理技术有机结合，对深度学习模型的输出进行了进一步改善。

5. **认识到深度学习与传统方法结合的价值**：DeblurGAN-v2的去模糊结果已经较好，但仍可通过传统图像处理技术进一步改善。这种深度学习与传统方法的结合是实际工程中的有效策略。

本实验加深了对深度学习图像恢复方法的理解，为后续研究和应用奠定了基础。

## 参考文献

[1] Kupyn O, Budzan V, Mykhailych M, et al. DeblurGAN: Blind motion deblurring using conditional adversarial networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 8183-8192.

[2] Kupyn O, Martyniuk T, Wu J, et al. DeblurGAN-v2: Deblurring (orders-of-magnitude) faster and better[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2019: 8878-8887.

[3] Lin T Y, Dollar P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 2117-2125.

[4] Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, Inception-ResNet and the impact of residual connections on learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2017, 31(1).

[5] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]//Advances in Neural Information Processing Systems (NeurIPS). 2014: 2672-2680.

[6] Nah S, Kim T H, Lee K M. Deep multi-scale convolutional neural network for dynamic scene deblurring[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 3883-3891.

[7] Johnson J, Alahi A, Fei-Fei L. Perceptual losses for real-time style transfer and super-resolution[C]//European Conference on Computer Vision (ECCV). 2016: 694-711.

[8] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks[C]//International Conference on Machine Learning (ICML). 2017: 214-223.
