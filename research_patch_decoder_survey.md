# Patch Decoder / Unpatchify Strategy Survey

**主题**: 主流 DiT / ViT 模型如何处理 patch boundary discontinuity / grid artifact 问题

---

## 核心发现总结

| Model | Decoder Strategy | Anti-Artifact Mechanism | Patch Boundary Handling |
|-------|-----------------|------------------------|------------------------|
| **DiT** | Linear → reshape (unpatchify) | 无 | 无，依赖 VAE decoder |
| **U-ViT** | Linear → reshape → **Conv2d(3×3)** | **3×3 卷积后处理** | Conv 提供跨 patch 平滑 |
| **SiT** | 同 DiT (Linear → reshape) | 无 | 无，依赖 VAE decoder |
| **PixArt-α** | 同 DiT (Linear → reshape) | 无 | 无，依赖 VAE decoder |
| **SD3 / MMDiT** | Linear → reshape (unpatchify) | 无 | 无，依赖 VAE decoder |
| **Flux** | LastLayer → rearrange (unpack) | 无 | 无，依赖 VAE decoder |
| **FourCastNet** | Linear head → rearrange | 无 | 无显式处理 |
| **Pangu-Weather** | **ConvTranspose3d/2d** | 学习的转置卷积 | 转置卷积提供隐式平滑 |
| **GenCast** | Graph mesh → grid (非 patch-based) | N/A | 无 patch 问题 |
| **DPOT** | **ConvTranspose2d → Conv2d × 2** | 多层卷积细化 | 转置卷积 + 卷积后处理 |
| **Poseidon (scOT)** | **ConvTranspose2d → Conv2d(5×5)** | U-Net skip + ConvNeXt + Conv 后处理 | 多层级卷积提供跨 patch 平滑 |

---

## 1. DiT (Peebles & Xie, ICCV 2023)

**Paper**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
**Code**: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)

### Decoder Architecture

DiT 使用最简单的 unpatchify 策略: **纯 linear projection + reshape**。

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

def unpatchify(self, x):
    # x: (N, T, patch_size**2 * C)
    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs
```

### Forward 末尾
```python
x = self.final_layer(x, c)       # (N, T, patch_size**2 * out_channels)
x = self.unpatchify(x)            # (N, out_channels, H, W)
```

### Patch Boundary 分析

- **完全没有跨 patch 机制**: 每个 patch token 独立映射到 `p×p` 像素块
- **FinalLayer 的 linear 权重初始化为 0**: `nn.init.constant_(self.final_layer.linear.weight, 0)`，这意味着初始阶段输出为零（类似 residual learning 的 zero init trick），但不解决 boundary 问题
- **为什么没问题**: DiT 工作在 VAE latent space（8× downsampled），不是 pixel space。latent space 的 patch boundary artifact 被 VAE decoder（卷积网络）自然平滑掉了

### 关键洞察
> DiT 系列模型之所以不需要处理 patch boundary artifact，是因为它们操作在 **latent space**，由 VAE decoder 负责最终的 pixel-level 重建。VAE decoder 本身是卷积网络，天然具有跨 patch 的感受野。

---

## 2. U-ViT (Bao et al., CVPR 2023)

**Paper**: [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)
**Code**: [baofff/U-ViT](https://github.com/baofff/U-ViT)

### Decoder Architecture

U-ViT 是所有 DiT-like 模型中 **唯一显式处理 patch boundary artifact** 的:

```python
# Step 1: Linear projection
self.decoder_pred = nn.Linear(embed_dim, patch_dim, bias=True)  # patch_dim = patch_size² × in_chans

# Step 2: Unpatchify (same reshape as DiT)
def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)',
                         h=h, p1=patch_size, p2=patch_size)
    return x

# Step 3: 3×3 Conv for artifact reduction  ← 关键！
self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()
```

### Patch Boundary 分析

- **3×3 Conv with padding=1**: 这是一个非常简单但有效的设计。在 unpatchify 之后，相邻 patch 之间的像素没有任何交互，3×3 卷积让每个像素能看到 ±1 范围的邻居，从而平滑 patch 边界
- **论文实验证实**: 加了 3×3 conv 比不加有明显的性能提升，特别是减少了可视化时的 grid artifact
- **为什么用 3×3 而不是更大**: 更大的 kernel 引入更多参数但收益递减；3×3 已经足够覆盖相邻 patch 的边界像素

### 关键洞察
> U-ViT 的 3×3 Conv 后处理是最 lightweight 的跨 patch 平滑方案。它说明 **即使在 latent space，一个小卷积就能显著减少 boundary artifact**。

---

## 3. SiT (Ma et al., ECCV 2024)

**Paper**: [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)
**Code**: [willisma/SiT](https://github.com/willisma/SiT)

### Decoder Architecture

SiT 直接复用 DiT 的 unpatchify:

```python
def unpatchify(self, x):
    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs
```

- **与 DiT 完全相同**，纯 linear + reshape
- SiT 的创新在 interpolant framework（连接 noise 和 data 的方式），而非 architecture
- 同样依赖 VAE decoder 平滑 patch boundary

---

## 4. PixArt-α (Chen et al., 2023)

**Paper**: [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)
**Code**: [PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)

### Decoder Architecture

PixArt-α 基于 DiT-XL 架构，使用相同的 unpatchify:

- 28 个 Transformer blocks
- Patch size = 2（在 latent space 上）
- Final layer → linear projection → unpatchify (reshape)
- 修改了 text conditioning（cross attention with T5）和 timestep embedding

### Patch Boundary 分析

- 与 DiT 一致，**无显式 patch boundary 处理**
- 在 VAE latent space 操作，由 VAE decoder 处理 pixel-level 重建

---

## 5. SD3 / Flux / MMDiT

### SD3 (Esser et al., 2024)

**Paper**: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)

```python
# Unpatchify (from diffusers/transformer_sd3.py)
self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

# In forward:
hidden_states = hidden_states.reshape(
    shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
)
hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
output = hidden_states.reshape(
    shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
)
```

- **与 DiT 完全一致的 unpatchify 模式**: Linear → reshape → einsum → reshape
- SD3 使用 16-channel VAE（比 SD1/2 的 4-channel 更多），latent space patch size = 2
- MMDiT 的创新在于 dual-stream attention（text + image 分开的权重），不在 decoder

### Flux (Black Forest Labs, 2024)

**Code**: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)

```python
# LastLayer（与 DiT 的 FinalLayer 几乎相同）
class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

# Unpack (in sampling.py)
def unpack(x, height, width):
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2, pw=2,
    )
```

- **同样是纯 reshape 操作**，无跨 patch 机制
- Flux 用 patch_size=1 但在更高压缩的 VAE 上操作（实际效果等价于 patch_size=2 + 8× VAE）

### 关键洞察
> **所有 latent diffusion 系列的 DiT 模型（DiT, SiT, PixArt, SD3, Flux）都使用相同的 unpatchify 策略: Linear projection → reshape。它们不需要处理 patch boundary artifact，因为 VAE decoder 负责了 pixel-level 的平滑重建。**

---

## 6. FourCastNet / Pangu-Weather / GenCast

### FourCastNet (Pathak et al., 2022)

**Paper**: [FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)
**Code**: [NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet)

```python
# Linear head projection
self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1])

# Unpatchify via rearrange
x = self.head(x)
x = rearrange(x, "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
              p1=self.patch_size[0], p2=self.patch_size[1])
```

- **与 DiT 几乎完全相同**: Linear → rearrange
- **但关键区别**: FourCastNet 直接在 **pixel space** 操作（不像 DiT 有 VAE decoder 兜底）
- AFNO token mixing 在 Fourier domain 做，有一定的全局感受野，但 **最后的 linear head 仍然是 per-patch 独立的**
- 论文报告了训练时会出现 edge artifact

### Pangu-Weather (Bi et al., Nature 2023)

**Paper**: [Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast](https://arxiv.org/abs/2211.02556)
**Code**: [198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather)

```python
class PatchRecovery:
    def __init__(self):
        # 3D 上层大气: patch_size = (2, 4, 4)
        self.conv = ConvTranspose3d(input_dims=dim, output_dims=5,
                                     kernel_size=patch_size, stride=patch_size)
        # 2D 地表变量: patch_size = (4, 4)
        self.conv_surface = ConvTranspose2d(input_dims=dim, output_dims=4,
                                             kernel_size=patch_size[1:], stride=patch_size[1:])
```

- **使用 ConvTranspose（转置卷积）而非 linear + reshape**
- 转置卷积的 kernel_size = stride = patch_size，数学上等价于 linear unpatchify，但权重是 **学习的**
- 配合 Swin Transformer encoder-decoder 的 U-Net 结构，有 skip connection
- **转置卷积提供了 patch 间隐式的权重共享**（当 kernel 与 stride 相等时，本质与 linear 相同，但当训练时有正则化效果）

### GenCast (Price et al., Nature 2024)

**Paper**: [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)

- **完全不同的范式**: GenCast 使用 graph neural network (GNN) on icosahedral mesh
- 不使用 patch-based tokenization
- Encoder: lat-lon grid → icosahedral mesh（41,162 nodes）
- Processor: graph transformer on mesh
- Decoder: mesh → lat-lon grid (message passing)
- **天然没有 patch boundary 问题**，因为 mesh nodes 之间通过 edge 连接，信息流是连续的

---

## 7. DPOT (Hao et al., ICML 2024)

**Paper**: [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training](https://arxiv.org/abs/2403.03542)
**Code**: [thu-ml/DPOT](https://github.com/thu-ml/DPOT)

### Decoder Architecture

DPOT 使用 **多层卷积 decoder**，是所有 PDE foundation model 中最精心设计的 output layer:

```python
self.out_layer = nn.Sequential(
    # Step 1: Transposed conv to upsample from patch space
    nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim,
                       kernel_size=patch_size, stride=patch_size),
    self.act,
    # Step 2: 1×1 conv for feature refinement
    nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim,
              kernel_size=1, stride=1),
    self.act,
    # Step 3: 1×1 conv to project to output channels
    nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,
              kernel_size=1, stride=1)
)
```

### Patch Boundary 分析

- **ConvTranspose2d 做 upsampling**: kernel_size = stride = patch_size，类似 Pangu-Weather
- **后接两层 1×1 Conv**: 提供了非线性变换能力，但 **1×1 Conv 无法提供跨 patch 的感受野**
- 虽然比 DiT 多了非线性变换，但严格来说 **仍然是 per-pixel 独立的**（因为 1×1 conv 不提供空间感受野）
- 如果 ConvTranspose2d 的 kernel 大于 stride，才会有 overlap（但这里 kernel = stride）

### 关键洞察
> DPOT 的 decoder 比 DiT 复杂一些（多了非线性变换），但其 1×1 Conv 不提供空间上的跨 patch 交互。实际的跨 patch 信息流主要依赖 Fourier attention 在 encoder 阶段的全局感受野。

---

## 8. Poseidon / scOT (Herde et al., NeurIPS 2024)

**Paper**: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101)
**Code**: [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon)

### Decoder Architecture

Poseidon 基于 scOT（scalable Operator Transformer），使用 **完整的 U-Net 式 encoder-decoder**:

```python
# Patch Recovery (final output layer)
class ScOTPatchRecovery:
    self.projection = nn.ConvTranspose2d(
        in_channels=hidden_size,
        out_channels=num_out_channels,
        kernel_size=patch_size,
        stride=patch_size,
    )
    self.mixup = nn.Conv2d(..., kernel_size=5, ...)  # 5×5 conv for smoothing!
```

### 完整 Decoder Pipeline

1. **多层级 Swin Transformer decoder**: 使用 SwinV2 blocks 逐层上采样
2. **Patch Unmerging**: `nn.Linear(dim, 2*dim)` 增加空间分辨率
3. **Skip Connections**: 每个 scale 有 ConvNeXt 卷积层连接 encoder 和 decoder
4. **Patch Recovery**: ConvTranspose2d 恢复到原始分辨率
5. **Mixup Layer**: **5×5 Conv** 做最终平滑

### Patch Boundary 分析

- **最完善的方案**: 多层级 ConvNeXt skip connection + 5×5 Conv 后处理
- ConvNeXt skip connection 在每个 scale 提供了 encoder 的高分辨率特征
- 5×5 Conv mixup layer 提供了比 U-ViT 的 3×3 更大的感受野
- 支持 residual learning (`learn_residual` 选项)

### 关键洞察
> Poseidon/scOT 是所有模型中 patch boundary 处理最完善的。它的 **U-Net 结构 + ConvNeXt skip + 5×5 Conv 后处理** 提供了多层级的跨 patch 信息流。这与它直接在 function space 操作（而非 latent space）有关——没有 VAE decoder 兜底，必须自己解决 boundary 问题。

---

## 整体分析与启示

### 为什么图像生成模型不 care patch boundary？

1. **VAE Latent Space 操作**: DiT/SiT/PixArt/SD3/Flux 都在 8× 或 16× downsampled latent space 操作
2. **VAE Decoder 是卷积网络**: 自然具有跨 spatial 的感受野，会平滑 patch boundary
3. **Diffusion 过程的平滑性**: 多步去噪过程本身有平滑效果

### 为什么 PDE/Weather 模型需要 care？

1. **直接在 pixel/function space 操作**: FourCastNet、DPOT 没有 VAE decoder 兜底
2. **物理量需要连续性**: PDE 解应该是（至少 C⁰）连续的，patch boundary 的不连续直接违反物理
3. **误差会累积**: autoregressive rollout 时，patch boundary artifact 会被放大

### 各策略的 trade-off

| 策略 | 跨 patch 能力 | 参数开销 | 代表模型 |
|------|-------------|---------|---------|
| Linear + reshape | 无 | 最少 | DiT, SiT, SD3, FourCastNet |
| + 3×3 Conv 后处理 | ±1 pixel | 极少 | U-ViT |
| + 5×5 Conv 后处理 | ±2 pixel | 少 | Poseidon |
| ConvTranspose2d | 无（当 kernel=stride 时） | 类似 linear | Pangu-Weather, DPOT |
| ConvTranspose2d + Conv 后处理 | 有（取决于 Conv kernel） | 中等 | Poseidon |
| U-Net + skip + Conv | 多层级 | 较多 | Poseidon |
| Overlap patch + blending | 有（overlap 区域） | 推理时间增加 | Patched Diffusion (非本文模型) |
| Graph/Mesh 方法 | 天然连续 | 完全不同的范式 | GenCast |

### 对我们项目的启示

如果我们的 PDE foundation model 直接在 function space 操作（无 VAE），最推荐的方案:

1. **最简单**: 在 unpatchify 后加一个 **3×3 或 5×5 Conv**（U-ViT / Poseidon 方案）
2. **中等复杂**: 使用 **ConvTranspose2d** 替代 linear + reshape，再加 Conv 后处理（DPOT 方案的改进版）
3. **最完善**: 采用 **U-Net style decoder with skip connections**（Poseidon 方案），但增加参数和计算量

---

## References

- [DiT - Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [U-ViT - All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)
- [SiT - Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)
- [PixArt-α](https://arxiv.org/abs/2310.00426)
- [SD3 - Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206)
- [Flux - Black Forest Labs](https://github.com/black-forest-labs/flux)
- [FourCastNet](https://arxiv.org/abs/2202.11214)
- [Pangu-Weather](https://arxiv.org/abs/2211.02556)
- [GenCast](https://arxiv.org/abs/2312.15796)
- [DPOT](https://arxiv.org/abs/2403.03542)
- [Poseidon](https://arxiv.org/abs/2405.19101)
