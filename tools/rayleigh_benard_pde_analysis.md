# Rayleigh-Bénard PDE Loss 分析报告

## 数据集概览

| 属性 | 值 |
|------|-----|
| 网格 | 512×128 (x×y), uniform |
| 域 | x∈[0,4] periodic, y∈[0,1] Dirichlet |
| dx, dy | 4/512≈0.0078, 1/127≈0.0079 |
| 时间步 | dt=0.25, T=200 |
| Pr=1 | κ=ν=1e-5 |
| Pr=10 | κ=3.16e-6, ν=3.16e-5 |

## PDE 方程组

```
∂b/∂t + u·∇b = κΔb           (浮力输运)
∂u/∂t + u·∇u = -∇p + νΔu + b·ê_y  (动量)
div(u) = 0                    (不可压缩)
```

BC: b(y=0)≈1, b(y=1)≈0, u(y=0)=u(y=1)=0

## 残差验证结果 (Pr=1, Ra=1e10, fp64)

### 方法对比 (skip 15 BL rows)

| 方程 | FD4_x + central_y | FFT_x + central_y | n-PINN |
|------|-------------------|-------------------|--------|
| Continuity | **2.27e-3** | 6.36e-2 | ~3.77e-2 |
| Buoyancy | **1.51e-2** | 1.51e-2 | ~similar |
| Momentum-x | **7.97e-4** | 7.97e-4 | ~similar |
| Momentum-y | **2.98e-3** | 2.98e-3 | ~similar |
| Vorticity | - | - | **25.5** (unusable) |

### 关键发现

1. **误差集中在底部边界层**: y≈0.02 处 MSE 高达 22，顶部仅 0.002
2. **FD4 优于 FFT**: FD 有隐式低通滤波效果，对湍流更友好
3. **n-PINN 和 FD 精度相当**: 无明显优势，FD 更简单
4. **涡度方程不可用**: 高阶导数放大截断误差，MSE~89
5. **Pr=10 比 Pr=1 好 5-10×**: 粘度更高→速度场更光滑

### Skip BL rows 效果 (FD4, Pr=1)

| skip | Continuity | Buoyancy | Mom-x | Mom-y |
|------|-----------|----------|-------|-------|
| 0 | 5.14e-1 | 1.22e-1 | 3.53e-3 | 2.35e-2 |
| 5 | 2.65e-1 | 6.41e-2 | 2.20e-3 | 8.93e-3 |
| 10 | 1.01e-1 | 2.72e-2 | 1.35e-3 | 4.80e-3 |
| **15** | **3.77e-2** | **1.51e-2** | **7.97e-4** | **2.98e-3** |
| 20 | 1.44e-2 | 7.43e-3 | 4.45e-4 | 2.18e-3 |

## 推荐方案

### 可用方程 (不需要压力)
1. **Continuity**: div(u) = 0 — 只需速度
2. **Buoyancy**: ∂b/∂t + u·∇b = κΔb — 需速度+浮力

### 数值方法
- **x 方向**: FD4 (4阶中心差分, periodic roll)
- **y 方向**: 2阶中心差分, interior only (skip boundary rows)
- **时间**: 2阶中心差分
- **BL mask**: skip 15 rows (top & bottom)

### 注意事项
- Ground truth 残差 ~1e-2，模型预测残差会远大于此
- PDE loss 作为 soft regularizer，权重不宜太大 (λ ~ 0.01-0.1)
- 动量方程需要压力，除非模型也预测 pressure
