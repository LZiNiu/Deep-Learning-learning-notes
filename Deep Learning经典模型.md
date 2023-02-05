# 各类模型

### 1 经典CNN卷积神经网络

#### 1.1 互相关运算

严格来说，卷积层是个错误的叫法，因为它所表达的运算其实是**互相关运算**（cross-correlation），而不是卷积运算，但是为了方便起见，下面仍称之为卷积。

互相关运算如下：
$$
\begin{array}{}
\begin{array}{|c|c|c|}
	\hline 0&1&2\\
	\hline 3&4&5\\
	\hline 6&7&8\\
	\hline
\end{array} \quad * \quad 
\begin{array}{|c|c|}
	\hline 0&1\\
	\hline 2&3\\
	\hline
\end{array} \quad=\quad
\begin{array}{|c|c|}
	\hline 19&25\\
	\hline 37&43\\
	\hline
\end{array}
\tag{2.1}
\\
\\ 0*0+1*1+3*2+4*3=19
\end{array}
$$
假设二维被卷积的对象的尺寸为$m*n$，卷积核的尺寸为$h*w$，（行，列亦或为高，宽）我们可以得到互相关运算的结果尺寸如下：
$$
(m-h+1\quad,\quad n-w+1)
$$
当涉及`padding(填充)`和`stride(步长)`时可以得到更加普适的公式如下：

首先定义`padding`尺寸$(p_h\;,\; p_w)$，`stride`尺寸$(s_h\;,\;s_w)$，$s_h$表示行之间移动步长，$s_w$表示列之间移动的步长，可以得到卷积互相关运算的结果的尺寸公式如下（其中$[·]$表示向下取整）：
$$
([\frac{m+2*p_h-h}{s_h}]+1\quad ,\quad [\frac{n+2*p_w-w}{s_w}]+1)
$$
> **Calculation Example**

```python
X = torch.rand(size=(8, 8))
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape

# 得到结果如下 ((8-3+0)/3+1,(8-5+2*1)/4+1) 相除向下取整
torch.Size([2, 2])
```

> **Code Example**

```python
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

#### 1.2 多通道情况下的卷积（互相关运算）

##### (1) 多通道单输出

运算示意图：

![../_images/conv-multi-in.svg](https://zh.d2l.ai/_images/conv-multi-in.svg)

上述运算总结为将各通道的卷积值相加得到最终结果。

> **Code Example**

```python
def corr2d_multi_in(X, K):
    # X:(chanels,height,width)
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(corr2d(x, k) for x, k in zip(X, K))
```

##### (2) 多通道多输出

定义$c_i$，$c_o$为输入通道数和输出通道数，单个卷积核`kernel`的尺寸$(k_h\;,\;k_w)$，由多通道单输出的分析我们知道，由 $c_i \times k_h\times k_w$ 可以得到单输出的一个结果，如果想要获得 $c_o$ 个通道数的结果，那么我们便需要 $c_o$ 个 由Tensor序列 $c_i \times k_h\times k_w$ 构成的Tensor序列，即构造大小为 $c_o\times c_i \times k_h\times k_w$ 的`kernel`Tensor序列。

示意图：（1×1尺寸的kernel，输入通道为3，输出通道为2）

![../_images/conv-1x1.svg](https://d2l.ai/_images/conv-1x1.svg)

#### 1.3 池化（Pooling）

池化层常见的有最大池化和均值池化，池化可以看做为一种特殊的卷积，它也有`padding`和`stride`等参数，且依然遵守上面提到的卷积结果尺寸公式。

（碎碎念）有关池化的作用：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

#### 1.4 几个著名的CNN网络

##### (1) LeNet

* `LeNet`主要由两部分构成：卷积编码层和全连接层

数据流为手写数字数据集，在控制全连接层的模型复杂度时采用的策略是==weight decay==

![../_images/lenet.svg](https://zh.d2l.ai/_images/lenet.svg)

```python
LeNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

##### (2) AlexNet

* `AlexNet`在控制全连接层的模型复杂度时采用的策略是==Drop out==

> 左边是LeNet，右边是AlexNet（略有改动）

![../_images/alexnet.svg](https://d2l.ai/_images/alexnet.svg)

```python
AlexNet = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
```

### 2 现代CNN

#### 2.1 使用块（Blocks）的网络VGG

我认为这种设计理念主要方便了现在训练好的网络模型与开发人员个人设计的网络的衔接，使得客制化神经网络更加容易，比如一个`hugging face`里的某些transformer模型权重已经训练好，个人有需求做一个NLP之类的任务，将训练好的模型直接拿过来作为一个Blocks，我们可以在其后面添加全连接层等之类作为另一个Blocks做一些简单的处理与训练即可满足需求。

此外，通过使用循环和子程序，可以很容易地在任何现代深度学习框架的代码中实现一些重复的架构。颇有封装的意味。

> 从AlexNet到VGG，主要区别为VGG由包含layers的块组成，AlexNet全部需要自主设计

![../_images/vgg.svg](https://d2l.ai/_images/vgg.svg)

> **Code Example**

```python
class VGG():
    def __init__(self, arch, lr=0.1, num_classes=10):
        self.lr = lr
        self.num_classes = num_classes
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(self.vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        # 权重初始化
        self.net.apply(init_cnn)
        
    def vgg_block(self, num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)
```

#### 2.2 Network in Network（NiN）

AlexNet、VGG的工作主要都是为了继续拓展LeNet，继续拓宽加深网络结构，然而这造成了计算参数的大量增加

`《Dive into Deep Learning》`书中还提到了其不能提早添加全连接线性层来提高非线性度。

NiN主要改变在以下两点：

* 使用 $1\times1$ 的卷积核利用其激活函数来为模型增加非线性度（==nonlinearities==）
* 在最后的表征层添加全局的平均池化层来进行整合，需要注意的是如果没有添加非线性那么这层将不起作用

> 改进对象都是AlexNet，VGG和NiN的比较

![../_images/nin.svg](https://d2l.ai/_images/nin.svg)

> **Code Example**

```python
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        self.lr = lr
        self.num_classes = num_classes
        self.net = nn.Sequential(
            self.nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self.nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        self.net.apply(init_cnn)

    def nin_block(self, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
            # 后接两层1*1型卷积核层
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

可以看到网络最后没有用到`fully connected layer`。

##### *  有关上面提到的网络的非线性度

从数学的角度来看，纯线性的组合
$$
\begin{aligned}
z^1 &= W^{1}x+b^1 \\
z^2 &= W^{2}z^1+b^2 \\ 
 &= W^2(W^1x+b^1)+b^2 \\
 &=W^2W^1x+(W^2b^1+b^2)\\

 \end{aligned}
$$

$$
\begin{flalign}
 令w = W^2W^1，b = W^2b^1+b^2，
 那么原方程z^2=wx+b显然仍是线性的
 即多层线性的组合等价于单线性层
 \end{flalign}
$$

非线性（Activation Function）可以增加模型的复杂度，使之更精确的逼近目标

那激活函数还有些什么用或者说类似于什么呢？

* 考虑神经网络的灵感，经过激活函数后的输出应该象征了神经元的活性，接近0说明不活跃，即激活函数在模拟人类神经的传递规则。
* 将输出限定在我们需要的范围内（==softmax==和==logistic==使之与概率联系起来）



