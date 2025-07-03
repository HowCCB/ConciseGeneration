# Denoising Diffusion Probabilistic Models (DDPM)

DDPM（去噪扩散概率模型）是一类生成模型，通过逐步向数据中添加噪声并学习逆过程来实现高质量的数据生成。其核心思想包括：
- **前向加噪过程（Forward Process）**：逐步将数据加噪，最终变为高斯噪声。
- **反向去噪过程（Reverse Process）**：训练一个神经网络逐步去噪，恢复原始数据。

---

## 1. 前向加噪过程（Forward Process）

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

---

