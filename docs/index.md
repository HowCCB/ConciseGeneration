从我身边观察到的，越来越多的同学选择AI作为自己读取更高学历的方向。然而事实是，即使AI工具非常好用的今天，仍旧有许多同学难以使用它们cover自己的科研工作。

更加值得提到的是，许多同学或许并没有深入科研的想法，选择AI是希望得到一个Master's degree，而有一些内容在理论上的理解有一定的门坎，这些门坎或许让大家对自己正在做的事情没有充足的自信。这让我想要做这件事情，于是有了这本Notebook。我先来介绍一下这本Notebook的特性。

+ 涵盖Generative Model的几乎所有你可能用到的（AE/VAE/VQ-VAE/DDPM/DDIM/flow_matching/Stable Diffusion）（classifier diffusion/classifier free diffusion/diffusion posterior sampling）（ContrlNet/condition generation）（latent interpolation/ddim inversion）
+ Keep it stupid，仅仅展示和代码相关的部分（但是会保证阅读的连贯性），忽略掉论文中大量在实践中无用的证明，从代码的角度，展示出对你理解有作用的部分。
+ 从最stupid的代码开始丰富，逐步加入Openai/Meta等开源的repo中使用到的tech code。（工程代码在原理上也是如此！只不过...是为了...）

如果你认真阅读这本Notebook，可以对Generative Model有一个总体的了解，同时对实现细节会更加自信（**我希望在读完以后，你会说：我确定在代码中，大家都是这样做的，或许和paper中一样，或许和paper中有一些区别，因此我更加有信心了**）。或许当你看完整个Notebook，会更加有自信开展自己正在进行的工作。

同时，必须要声明的一点是，这本Notebook中，不包含任何学术上的建议。为了理解方便，我会忽略许多细节，但是这对你理解整个流程或者代码或许是有帮助的。

在char1中，我们介绍了需要准备的数据，我们希望这个数据是实用的（不要离自己的科研工作太远，比如选用MNIST手写数字集我认为就实在偏离太远，对于你的理解并不一定会有太多的帮助），同时也是尽可能简化的（不选LAION-5B作为学习数据集，因为这个它实在太大）。

所有的一切，都是一个目的，希望能用最短的时间，了解学术和工程上的实现，让你对正在做的事情有充足的信息，尽快完成您的毕业工作。

时间匆忙，难免疏漏，感谢你通过这个邮箱与我讨论：liushr6688@gmail.com

打赏可以扫这里：
<div style="display: flex; gap: 20px; align-items: center;">
  <img src="506c214d3415c1f4c6f69ca69a8b7ed.jpg" alt="alt text" width="200"/>
  <img src="3371047bf6c7fab8317e819fb04db0f.jpg" alt="alt text" width="200"/>
    <img src="05e66ec06ac6fa4cc4436b0285e4dbe.jpg" alt="alt text" width="200"/>

</div>
