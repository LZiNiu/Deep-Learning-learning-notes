# Neural Network 中的一些操作

## 1、Softmax

### 1.1 典型的softmax代码

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition
```

其虽然直观便于公式理解，但是对于数值不稳定没有足够的健壮性（robust），不建议在正式项目中这么写。

###  1.2 SoftMax的数值计算问题

我们计算模型的输出并且使用了交叉熵损失。由于指数运算造成的数值向下溢出和向上溢出，这在数学上时完全合理的。通过softmax的函数表达形式$ \hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} $，其中$k$表示有类别数，$j$则表示第$j$类，$o_j$表示模型在第$j$类上的输出，如果$o_j$很大且是正数，那么$exp(o_j)$将会是一个十分大的数造成数值的`overflow`，反之$o_j$很小（负数）那么将会造成数值的`underflow`，例如单精度浮点数大概范围为$10^{-38}$到$10^{38}$，因此，如果$o$的最大项位于区间$[-90,90]$外，那么计算结果将变得数值不稳定，一个解决办法如下：

定义：$\bar{o} = \stackrel{k}\max o_k$，上述softmax公式变换如下：
$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}
$$
显然，对于一个$q$分类问题，上式分母的数值必然被限定在区间$[1,q)$，而分子必定小于等于$1$，这解决了数值的overflow问题，但是，underflow问题依然存在，其出现时 $exp(o_j-\bar o)$ 与 $0$ 无异，在我们计算交叉熵 $log\hat y_j$ 时将会导致计算 $log0$ ，在反向传播时，我们将可能遇见许多NaN（Not a Number)的结果。

万幸的是，通过交叉熵和Softmax二者结合能发现我们可以避开数值稳定性问题：

计算交叉熵损失时我们将会对Softmax的结果进行取 $log$ 的操作：
$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o})
$$

* 一点疑问：$log\;exp(o_j-\bar o)==o_j-\bar o???$



## 2、神经网络Dropout(暂退法)

克里斯托弗·毕晓普证明了具有输入噪声的训练等价于Tikhonov正则化。这项工作用数学证实了“要求函数光滑”和“要求函数对输入的随机噪声具有适应性”之间的联系。

* 经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标。参数的范数代表了一种有用的简单性度量。
* 另一个角度是平滑性，平滑性，即函数不应该对其输入的微小变化敏感。

在训练过程中，在计算后续层之前向网络的每一层注入噪声。 因为当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增强平滑性。

### 2.1 理论操作

`Dropout`在前向传播过程中，计算每一内部层的同时注入噪声。从表面上看是在训练过程中丢弃（drop out）一些神经元。 在整个训练过程的每一次迭代中，标准`Dropout`包括在计算下一层之前将当前层中的一些节点置零。

在标准`Dropout`正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。 换言之，每个中间活性值 $h$ 以暂退概率 $p$ 由随机变量 $h'$ 替换，如下所示：
$$
\begin{split}\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 其他情况}
\end{cases}
\end{aligned}\end{split}
$$
由以上公式易得 $E(h')=h$ ，即期望仍保持不变。

### 2.2 简单的单层Dropout代码实现

```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
------------------------------------------------------
# 结果如下
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
        
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
        
tensor([[ 0.,  2.,  0.,  6.,  8., 10.,  0.,  0.],
        [16.,  0.,  0., 22.,  0., 26.,  0.,  0.]])
        
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

### 2.3 调用Pytorch框架

对于深度学习框架的高级API，我们只需在每个全连接层之后添加一个`Dropout`层， 将暂退概率作为唯一的参数传递给它的构造函数。 在训练时，`Dropout`层将根据指定的暂退概率随机丢弃上一层的输出（相当于下一层的输入）。 在测试时，`Dropout`层仅传递数据。

> **Code Example**

```python
# dropout1，dropout2为概率
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    # 权重初始化
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

