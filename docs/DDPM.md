# Denoising Diffusion Probabilistic Models (DDPM)

DDPM（去噪扩散概率模型）是一类生成模型，通过逐步向数据中添加噪声并学习逆过程来实现高质量的数据生成。其核心思想包括：
- **前向加噪过程（Forward Process）**：逐步将数据加噪，最终变为高斯噪声。
- **反向去噪过程（Reverse Process）**：训练一个神经网络逐步去噪，恢复原始数据。

---

## 1. 前向加噪过程（Forward Process）

前向过程是一个马尔可夫链（Markov Chain）过程。假设原始数据为 $x_0$，每一步加噪如下：

$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t
$$


其中各符号的意义如下：

- $x_t$：第 $ t $ 步加噪后的图像或数据。
- $x_{t-1}$：第 $ t-1 $ 步的图像或数据。
- $\beta_t$：预定义的噪声调度参数（variance schedule），表示第 $ t $ 步添加噪声的比例 [[2]]。
- $\epsilon_t$：从标准正态分布中采样的噪声项，即 $ \epsilon_t \sim \mathcal{N}(0, I) $ [[6]]。



其中 $\epsilon_t \sim \mathcal{N}(0, I)$，$\beta_t$ 是预设的噪声调度表。经过 $T$ 步后，$x_T$ 近似为标准高斯分布。




实际上，因为前后两项之间的依赖关系，可以得到$xt$和$x_0$之间的关系，在实际的代码中，也是如此操作的。

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t
$$

其中：
- $ x_t $：第 $ t $ 步加噪后的图像或数据 [[4]]。
- $ x_0 $：初始的干净图像或数据 [[6]]。
- $ \bar{\alpha}_t $：前向扩散过程中从初始到第 $ t $ 步的累积噪声系数，定义为 $ \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s) $，其中 $ \beta_s $ 是每一步的噪声调度参数 [[9]]。
- $ \epsilon_t $：从标准正态分布中采样的噪声项，即 $ \epsilon_t \sim \mathcal{N}(0, I) $ [[6]]。


$ x_t $与$ x_0 $和$x_{t-1}$分布之间的关系，符合下面两个公式，下面这两个公式可以通过重采样得到上面的两个加噪公式。

在 DDPM 中，$ x_t $ 与 $ x_0 $ 和 $ x_{t-1} $ 的分布关系可以通过以下两个公式表示：

1. **从 $ x_0 $ 得到 $ x_t $ 的分布**：
   $$
   q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
   $$
   其中：
   - $ \bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s) $ 是从初始到第 $ t $ 步的累积噪声系数 [[9]]。
   - $ \mathcal{N}(\cdot; \mu, \Sigma) $ 表示均值为 $ \mu $、协方差为 $ \Sigma $ 的高斯分布。

2. **从 $ x_{t-1} $ 得到 $ x_t $ 的分布**：
   $$
   q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
   $$
   其中：
   - $ \beta_t $ 是第 $ t $ 步的噪声调度参数 [[2]]。

这两个分布描述了前向扩散过程中，如何通过逐步添加高斯噪声将 $ x_0 $ 或 $ x_{t-1} $ 转换为 $ x_t $。并且可以通过重参数化技巧采样得到 $ x_t $ [[1]]。

## 2. 反向去噪过程（Reverse Process）

反向过程同样是马尔可夫链，但参数未知，需要用神经网络 $\epsilon_\theta$ 进行建模：

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

训练目标是最小化预测噪声与真实噪声的均方误差（MSE）：

$$
L_{simple} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

---

## 3. DDPM 采样流程

1. 从高斯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始
2. 依次用神经网络预测噪声，逐步去噪 $x_{T-1}, x_{T-2}, ..., x_0$
3. $x_0$ 即为生成样本

---

## 4. 代码简例（PyTorch）

```python
# 前向加噪
x_t = sqrt_alpha_cumprod[t] * x_0 + sqrt_one_minus_alpha_cumprod[t] * torch.randn_like(x_0)

# 反向去噪（伪代码）
for t in reversed(range(T)):
    pred_noise = model(x_t, t)
    x_t = ... # 按论文公式更新
```

---

