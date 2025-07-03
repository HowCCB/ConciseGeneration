# Denoising Diffusion Probabilistic Models (DDPM)

DDPM（去噪扩散概率模型）是一类生成模型，通过逐步向数据中添加噪声并学习逆过程来实现高质量的数据生成。其核心思想包括：

- **前向加噪过程（Forward Process）**：逐步将数据加噪，最终变为高斯噪声。
- **反向去噪过程（Reverse Process）**：训练一个神经网络逐步去噪，恢复原始数据。

---

## 1. 前向加噪过程（Forward Process）



下面这个公式就是代码中给 $x_0$ 加噪声时用到的公式：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t
$$

后面内容为详细些的解释，可以忽略掉不看。

---


前向过程是一个马尔可夫链（Markov Chain）过程。假设原始数据为 $x_0$，每一步加噪如下：

$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \, \epsilon_t
$$


其中各符号的意义如下：

- $x_t$：第 $t$ 步加噪后的图像或数据。
- $x_{t-1}$：第 $t-1$ 步的图像或数据。
- $\beta_t$：预定义的噪声调度参数（variance schedule），表示第 $t$ 步添加噪声的比例。
- $\epsilon_t$：从标准正态分布中采样的噪声项，即 $\epsilon_t \sim \mathcal{N}(0, I)$。

经过 $T$ 步后，$x_T$ 近似为标准高斯分布。




实际上，因为前后两项之间的依赖关系，可以得到 $x_t$ 和 $x_0$ 之间的关系，在实际的代码中，也是如此操作的：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t
$$

其中：

- $x_0$：初始的干净图像或数据。
- $\bar{\alpha}_t$：前向扩散过程中从初始到第 $t$ 步的累积噪声系数，定义为 $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$。
- $\epsilon_t$：从标准正态分布中采样的噪声项。


$x_t$ 与 $x_0$ 和 $x_{t-1}$ 的分布关系如下：

1. 从 $x_0$ 得到 $x_t$ 的分布：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

2. 从 $x_{t-1}$ 得到 $x_t$ 的分布：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$


上述两个分布通过重参数化，就可以得到


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t
$$


$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \, \epsilon_t
$$

这两个公式了。



---

## 2. 反向去噪过程（Reverse Process）

在 DDPM 的反向去噪过程中，核心目标是逐步从噪声数据中恢复出原始图像。该过程是一个马尔可夫链，每一步都依赖于当前时间步的预测值。反向过程中的分布公式如下：

1. **反向过程的每一步分布**：
   $$
   p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
   $$
   其中：

   - $\mu_\theta(x_t, t)$ 是模型预测的均值（就是那个Unet）。
   - $\Sigma_\theta(x_t, t)$ 是模型预测的方差，实验中，可以设置为固定值或者学习值（如果是学习值，则也是那个Unet）。

2. **均值和方差的具体形式**：
   根据前向扩散过程的推导，反向过程的均值和方差可以表示为：
   
   $$
   \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
   $$

   $$
   \Sigma_\theta(x_t, t) = \beta_t I
   $$
   其中：

   - $\alpha_t = 1 - \beta_t$ 是前向扩散过程中的系数。
   - $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ 是累积噪声系数。
   - $\epsilon_\theta(x_t, t)$ 是模型预测的噪声项 [[6]]。

3. **初始分布**：
   反向过程的起始点是从标准正态分布采样的噪声：
   $$
   x_T \sim \mathcal{N}(0, I)
   $$
   其中 $x_T$ 是纯噪声输入 [[8]]。

通过这些公式，反向过程能够从纯噪声 $ x_T $ 逐步去噪生成 $ x_{T-1}, x_{T-2}, \cdots, x_0 $，最终得到生成的样本 $ x_0 $ [[10]]。