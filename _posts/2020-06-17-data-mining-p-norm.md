---
title: data-mining-p-norm
date: 2020-06-09 09:45:00 +0800
categories: [data-mining]
tags: [p-norm]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAyMC0wNy0yNg==
---

# Foreword

这里介绍数据挖掘中的正则化技术。

# 定义

Given a vector space V over a field F of the real numbers $\mathbb {R}$  or complex numbers $\mathbb {C}$ , a norm on V is a nonnegative-valued function p: V → $\mathbb {R}$  with the following properties:[2]

For all a ∈ F and all u, v ∈ V,

​	p(u + v) ≤ p(u) + p(v) (being subadditive or satisfying the triangle inequality).
​	p(av) = |a| p(v) (being absolutely homogeneous or absolutely scalable).
​	If p(v) = 0 then v = 0 is the zero vector (being positive definite or being point-separating).

A seminorm on V is a function p : V → $\mathbb {R}$  with the properties 1 and 2 above.[3] An ultraseminorm or a non-Archimedean seminorm is a seminorm p with the additional property that p(x+y) ≤ max { p(x), p(y) } for all x, y ∈ V.[4]

Every vector space V with seminorm p induces a normed space V/W, called the quotient space, where W is the subspace of V consisting of all vectors v in V with p(v) = 0. The induced norm on V/W is defined by:

​	p(W + v) = p(v).

Two norms (or seminorms) p and q on a vector space V are equivalent if there exist two real constants c and C, with c > 0, such that

​	for every vector v in V, one has that: c q(v) ≤ p(v) ≤ C q(v).



# 范数

在线性代数等数学分支中，范数（Norm）是一个函数，其给予某向量空间（或矩阵）中的每个向量以长度或称之为大小。对于零向量，其长度为零。直观的说，向量或矩阵的范数越大，则我们可以说这个向量或矩阵也就越大。有时范数有很多更为常见的叫法，如绝对值其实便是一维向量空间中实数或复数的范数，范数的一般化定义：设𝑝≥1，p-norm用以下来表示

$$
\large {\Vert x \Vert}{p} = \lgroup {\sum{i}{\vert x_i \vert}^p }\rgroup ^{\frac{1}{p}}
$$

## L0 范数

当p=0时，严格的说此时p已不算是范数了，L0范数是指向量中非0的元素的个数，但很多人仍然称之为L0范数（Zero norm零范数）。

$$
\large \Vert x \Vert = \sqrt[0]{\sum_i x_i^0} (i|xi \neq0)
$$

## L1 范数

当p=1时，我们称之曼哈顿范数(Manhattan Norm)。其来源是曼哈顿的出租车司机在四四方方的曼哈顿街道中从一点到另一点所需要走过的距离。也即我们所要讨论的L1范数。其表示某个向量中所有元素绝对值的和。

$$
\large {\Vert x \Vert}{1} = \lgroup {\sum_{i}{\vert xi \vert} }\rgroup
$$

## L2 范数

当p=2时，则是我们最为常见的Euclidean norm。也称为Euclidean distance，中文叫欧几里得范数，也即我们要讨论的L2范数，他也经常被用来衡量向量的大小。 

$$
\large {\Vert x \Vert}{2} = \lgroup {\sum_{i}{\vert x_i \vert}^2 }\rgroup ^{\frac{1}{2}}
$$

## L∞ 范数

当 𝑝−>∞ 时，我们称之为 𝐿∞范数，也被称为“maximum norm（max范数）”。也称为切比雪夫距离Chebyshev distance。这个范数表示向量中具有最大幅度的元素的绝对值：

$$
\large {\Vert x \Vert}^{\infty} =  \max_{i}{\vert x_i \vert}
$$

## 计算向量的范数

```python
import numpy as np


def p_norm(x, p, infty=False):
    """
    参数
        x: numpy/list数组，计算向量
        p: int型整数或者None，范数的阶
        infty: bool型变量，是否计算L∞范数，True的时候表示计算L∞范数，False的时候计算Lp范数

    返回
        float类型数值，向量的范数
    """
    if not infty:
        # 计算p范数
        if p == 0:
            return np.nonzero(x)[0].size  # 非零元素个数
        else:
            # 使用 np.abs()将x中的元素取其绝对值，内置函数power，用于单个元素的次幂运算
            return np.power(np.power(np.abs(x), p).sum(), 1 / p)

    else:
        # L∞范数
        return np.max(x)
```

## 计算矩阵的范数

矩阵大小的衡量在很多优化问题中是非常重要的。而在深度学习中，最常见的做法是使用Frobenius 范数(Frobenius norm)，也称作矩阵的F范数，其定义如下：

$$
\large {\Vert A \Vert}{F} = \sqrt {\sum{i,j}{\vert A_{i,j} \vert}^2 }
$$

```python
def f_norm(matrix):
    """
    参数
        matrix: list/numpy数组，给定的任意二维矩阵

    返回
        float类型数值，矩阵的Frobenius范数
    """
    return np.power(np.power(np.abs(matrix), 2).sum(), 1 / 2)
```

## 计算矩阵的条件数

矩阵的条件数(condition number)是矩阵（或者它所描述的线性系统）的稳定性或者敏感度的度量，我们这里为了简化条件，这里只考虑矩阵是奇异矩阵的时候，如何计算以及理解条件数(condition number):

当矩阵A为奇异矩阵的时候，condition number为无限大；当矩阵A非奇异的时候，我们定义condition number如下：

$$
\large \kappa{(A)} =  {\Vert A \Vert}_F {\Vert A^{-1} \Vert}_F
$$

```python
def f_norm_condition(matrix):
    """
    参数
        matrix: list/numpy数组，给定的任意二维矩阵

    返回
        float类型数值，矩阵的condition number
    """
    if np.linalg.det(matrix) != 0:
        # np.linalg.inv()：矩阵求逆
        # np.linalg.det()：矩阵求行列式（标量）
        return f_norm(matrix) * f_norm(np.linalg.inv(matrix))
    else:
        return float('inf')
```

## numpy实现p-norm计算

```python
"""
x: 表示矩阵（也可以是一维）

ord：范数类型

　　向量的范数：

　　矩阵的范数：
　　　　ord=1：列和的最大值
　　　　ord=2：|λE-ATA|=0，求特征值，然后求最大特征值得算术平方根
　　　　ord=∞：行和的最大值

axis：处理类型

　　　　axis=1表示按行向量处理，求多个行向量的范数
　　　　axis=0表示按列向量处理，求多个列向量的范数
　　　　axis=None表示矩阵范数。

keepding：是否保持矩阵的二维特性

　　　　True表示保持矩阵的二维特性，False相反
"""
x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False)
```

# 正则化

正则化损失函数的惩罚项。所谓惩罚是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用 L1 正则化的模型建叫做 Lasso 回归，使用 L2 正则化的模型叫做 Ridge 回归（岭回归）。

## L1正则化

L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。

稀疏矩阵指的是很多元素为 0，只有少数元素是非零值的矩阵，即得到的线性回归模型的大部分系数都是 0。 通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个稀疏模型，表示只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，或者贡献微小（因为它们前面的系数是 0 或者是很小的值，即使去掉对模型也没有什么影响），此时我们就可以只关注系数是非零值的特征。这就是稀疏模型与特征选择的关系。

## L2正则化

L2 正则化可以防止模型过拟合（overfitting）。一定程度上，L1 也可以防止过拟合。

拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响。但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响 – 泛化能力强。

# unit ball

在p范数下定义的单位球（unit ball）都是凸集（convex set，简单地说，若集合A中任意两点的连线段上的点也在集合A中，则A是凸集），但是当0<p<1时，在该定义下的unit ball并不是凸集（注意：我们没说在该范数定义下，因为如前所述，0<p<1时，并不是范数）。

	当0<p<1时，上面类似p范数的定义不能对任意两点满足三角不等式，也就是说，存在两点，它们不满足三角不等式。这个论断证明起来很简单，只要找出两个这样的点就行了。
	
	在一维空间中，按照p范数的定义，三角不等式总是成立。于是我们可以考虑在二维空间选点（因为二维空间比较简单），考虑特殊一点的，比如，取x=(0,1), y=(1,0)
	
	||x|| = 1， ||y|| = 1，||x+y|| = 2^(1/p) > 2 == ||x|| + ||y||，这就是一个违反三角不等式的例子，证毕。

对于更高维空间都可以取类似的例子，比如三维就取(0,0,1), (0, 1, 0), (1,0,0)

下面的python代码可以用来画p取不同值时的unit ball：

```python
import pylab
from numpy import array, linalg, random


def plot_p_norm_unit_ball(p):
    """
    plot some 2D vectors with p-norm
    """
    for i in range(5000):
        x = array([random.rand() * 2 - 1, random.rand() * 2 - 1])
        if linalg.norm(x, p) < 1:
            pylab.plot(x[0], x[1], 'bo')

    pylab.axis([-1.5, 1.5, -1.5, 1.5])
    pylab.show()


plot_p_norm_unit_ball(1)
plot_p_norm_unit_ball(2)
```

结果长这样：

![P-norm](https://upload.wikimedia.org/wikipedia/commons/e/e6/Minkowski3.png "P-norm")


更多细节，可参考：

https://en.wikipedia.org/wiki/Norm_(mathematics)

http://www.digtime.cn/articles/101/python-xian-xing-dai-shu-ji-qi-xue-xi-bei-hou-de-you-hua-yuan-li-wu-shi-wu

https://izhangzhihao.github.io/2017/11/18/%E8%8C%83%E6%95%B0%E5%92%8C%E8%B7%9D%E7%A6%BB/