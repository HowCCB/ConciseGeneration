# 变分自编码器（Variational Autoencoders, VAE）

变分自编码器（VAE）是在[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)这篇论文中提出的，是一种基于概率图模型和变分推断的生成模型。

## 基本原理

VAE 假设观测数据 $x$ 由潜变量 $z$ 生成，目标是最大化观测数据的边缘似然 $p(x)$。由于直接计算 $p(x)$ 很困难，VAE 采用变分推断，引入近似后验 $q_\phi(z|x)$，并最大化证据下界（ELBO）：

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

其中：

- $p_\theta(x|z)$：解码器（生成器），给定潜变量 $z$ 生成数据 $x$
- $q_\phi(z|x)$：编码器（推断器），给定数据 $x$ 推断潜变量 $z$
- $p(z)$：先验分布，通常为标准正态分布 $\mathcal{N}(0, I)$
- $D_{KL}$：KL 散度，用于衡量两个分布的差异

## 目标函数

VAE 的损失函数即为负的 ELBO：

$$
\mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) \| p(z))
$$

- 第一项是重构误差，衡量生成样本与原始样本的相似度
- 第二项是正则项，使编码分布接近先验分布

## 代码实现

（此处留空，后续可补充 PyTorch/TensorFlow 等实现代码）

---

> 如果公式仍无法正常显示，请确保 mkdocs serve 后强制刷新浏览器缓存（Ctrl+F5），并确认 mathjax CDN 能正常访问。