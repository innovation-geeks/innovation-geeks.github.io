---
title: data-mining-preprocessing
date: 2020-06-08 09:45:00 +0800
categories: [data-mining]
tags: [preprocessing]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAyMC0wNi0xOA==
---

# Foreword

数据挖掘中数据预处理方法介绍。数据预处理主要使用以下几种手段：

	归一化（normalization）：归一化是将样本的特征值转换到同一量纲下，把数据映射到[0,1]或者[-1, 1]区间内。
	
	标准化（standardization）：标准化是将样本的特征值转换为标准值（z值），每个样本点都对标准化产生影响。

以下以MinMaxScaler为例，介绍sklearn中数据预处理模块使用方法：

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

# 可根据数据实际情况，选择不同的标准化方法
minmax_scalar = MinMaxScaler()
minmax_scalar = minmax_scalar.fit(X_train)

# transform train data
X_train_scalar = minmax_scalar.transform(X_train)

# transform test data
X_test_scalar = minmax_scalar.transform(X_test)

print(X.shape, X_train.shape, X_train_scalar.shape, X_test.shape, X_test_scalar.shape)
```

# 标准化

## 缩放特征到指定范围

数据集的标准化（Standardization）对scikit-learn中实现的大多数机器学习算法来说是常见的要求。如果个别特征或多或少看起来不是很像标准正态分布（不具有零均值和单位方差），那么这些机器学习算法的表现可能会比较差。

	例如，在机器学习算法的目标函数（例如SVM的RBF内核或线性模型的L1和L2正则化）中有很多地方都假定了所有特征都是以0为中心而且它们的方差也具有相同的阶数。如果某个特征的方差比其他特征大几个数量级，那么它就会在学习算法的目标函数中占据主导位置，导致学习器并不能像我们所期望的那样，从数据中很好地学习特征信息。

### 标准化

一般的统计分析方法都在假设数据服从正态分布，所有有的模型要求输入数据需为正态分布，遇到这类的模型时需要应用正态分布标准化变换。正态分布标准化使被缩放的数据具有零均值和单位方差。

$$
X_{scalar}=\dfrac {X-mean\left( X\right) }{std\left( X\right) }
$$

sklearn使用：

```python
from sklearn.preprocessing import StandardScaler
```

### 最大最小标准化

最大最小标准化方法主要用于将数据缩放到[0, 1]范围内，避免数据的分布太过广泛，但是这种方法有一个致命的缺点，就是其容易受到异常值的影响，一个异常值可能会将变换后的数据变为偏左或者是偏右的分布，因此在做最大最小标准化之前一定要去除相应的异常值才行。

$$
X_{scalar}=\dfrac {X-\min \left( X\right) }{\max \left( X\right) -\min \left( X\right) }
$$

sklearn使用：

```python
from sklearn.preprocessing import MinMaxScaler
```

### 最大绝对值标准化

通过除以每个特征的最大值将训练数据缩放至[-1, 1]区间内，使用这种缩放的动机包括实现对特征的极小标准差的鲁棒性以及在稀疏矩阵中保留零元素。适用于稀疏数据。

$$
X_{scalar}=\dfrac {X}{\max\left( \|X\|\right) }
$$

sklearn使用：

```python
from sklearn.preprocessing import MaxAbsScaler
```

## 缩放稀疏数据

中心化稀疏(矩阵)数据会破坏数据的稀疏结构，因此很少有一个比较合理的实现方式。但是缩放稀疏输入是有意义的，尤其是当几个特征在不同的量级范围时。

	注意：缩放器（scaler）可以接受压缩过的稀疏行也可以接受压缩过的稀疏列（参见scipy.sparse.csr_matrix以及scipy.sparse.csc_matrix ）。任何其他稀疏输入将会转化为压缩稀疏行表示。为了避免不必要的内存复制，建议在早期选择CSR或CSC表示。

## 缩放带有异常值的数据

如果你的数据包含许多异常值，使用均值和方差缩放可能并不是一个很好的选择。 介绍以下几种标准化方式。

### Robust标准化

计算过程，先减去中位数，再除以四分位间距（interquartile range），因为不涉及极值，因此在数据里有异常值的情况下表现比较稳健。适用于异常值处理。

	缩放与白化(Scaling vs Whitening)：
	有时候独立地中心化和缩放数据是不够的，因为下游的机器学习模型会进一步对特征之间的线性依赖做出一些假设。
	要解决这个问题，你可以使用 sklearn.decomposition.PCA 类 并指定参数 whiten=True 来进一步移除特征间的线性关联。

$$
X_{scalar}=\dfrac {X-median\left( X\right) }{IQR\left( X\right) }
$$

sklearn使用：

```python
from sklearn.preprocessing import RobustScaler
```

### 中心化核矩阵

如果你有一个核\(K\)的核矩阵（核 \(K\) 在由函数 \(phi\) 定义的特征空间上计算点积），类KernelCenterer可以变换核矩阵（kernel matrix）以使得它包含由函数 \(phi\) 定义的被去除了均值的特征空间上的内积。

sklearn使用：

```python
from sklearn.preprocessing import KernelCenterer
```

## 非线性变换

### 映射到均匀分布

类似于缩放器(scalers)，QuantileTransformer类将所有特征放到同样的已知范围或同样的已知分布 下。但是，通过执行一个排序变换(rank transformation)，它能够使异常的分布(unusual distributions) 被平滑化，并且能够做到比使用缩放器(scalers)方法更少地受到离群值的影响。然而，它的确使特征间及特征内的关联和距离被打乱了。
QuantileTransformer 提供了一个基于分位数函数的无参数变换器将数据映射到一个取值在0到1之间的均匀分布上。

sklearn使用：

```python
from sklearn.preprocessing import QuantileTransformer
```

### 映射到高斯分布

在许多建模场景中，需要数据集中的特征的正态化(normality)。幂变换(Power transforms)是一类参数化的单调变换(parametric, monotonic transformations)， 其目的是将数据从任何分布映射到尽可能接近高斯分布，以便稳定方差(stabilize variance)和最小化偏斜(minimize skewness)。

类 PowerTransformer 目前提供两个这样的幂变换：the Yeo-Johnson transform 和 the Box-Cox transform。

The Yeo-Johnson transform 如下定义:

$$
[\begin{split}x_i^{(\lambda)} = \begin{cases} [(x_i + 1)^\lambda - 1] / \lambda & \text{if } \lambda \neq 0, x_i \geq 0, \\[8pt] \ln{(x_i) + 1} & \text{if } \lambda = 0, x_i \geq 0 \\[8pt] -[(-x_i + 1)^{2 - \lambda} - 1] / (2 - \lambda) & \text{if } \lambda \neq 2, x_i < 0, \\[8pt] - \ln (- x_i + 1) & \text{if } \lambda = 2, x_i < 0 \end{cases}\end{split}]
$$

而 the Box-Cox transform 如下定义:

$$
[\begin{split}x_i^{(\lambda)} = \begin{cases} \dfrac{x_i^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, \\[8pt] \ln{(x_i)} & \text{if } \lambda = 0, \end{cases}\end{split}]
$$

Box-Cox 仅能被应用于严格正(positive)的数据上。 在这两种方法中，变换(transformation)通过 \(\lambda\) 被参数化： 参数 \(\lambda\) 的值可以通过极大似然估计法(maximum likelihood estimation)得到。这里有一个使用例子：使用 Box-Cox 把从对数正态分布(lognormal distribution)抽取出的样本映射成服从正太分布(normal distribution)的数据:

sklearn使用：

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='box-cox', standardize=False)
transformer.fit_transform(X_train)
```

# 归一化

归一化(Normalization) 是 缩放单个样本使其具有单位范数 的过程。 如果你计划使用二次形式(quadratic form 如点积或任何其他核函数)来量化任何样本间的相似度，则此过程将非常有用。

## 正则化

使用原因：
每个样本被单独缩放，使得其范数等于1。注意，是对每个样本，不再像之前的（默认对列进行规范化）规范化。
文本分类或聚类中常用。可用于稠密的numpy数组和scipy.sparse矩阵（如果你想避免复制/转换的负担，请使用CSR格式）
Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（L1-norm, L2-norm）等于1。

p-范数的计算公式详见下篇：[P-Norm](/posts/data-mining-p-norm/ "data-mining-p-norm")

```python
from sklearn.preprocessing import Normalizer
```

## 离散型特征编码

在机器学习中，特征经常不是连续的数值型的而是离散型(categorical)。举个例子，一个人的特征可能有：
	["male", "female"]，
	["from Europe", "from US", "from Asia"]，
	["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]
等等。 这些特征能够被有效地编码成整数，比如：
	["male", "from US", "uses Internet Explorer"] 可以被表示为 [0, 1, 3]，
	["female", "from Asia", "uses Chrome"] 表示为 [1, 2, 1] 。

### OrdinalEncoder

要把离散型特征(categorical features) 转换为这样的整数编码(integer codes), 我们可以使用 OrdinalEncoder。 这个 Encoder 把每一个 categorical feature 变换成一个新的整数数字特征 (0 到 n_categories - 1):

```python
from sklearn.preprocessing import OrdinalEncoder

X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]

encoder = OrdinalEncoder()
encoder.fit(X)

# will get: array([[0., 1., 1.]])
encoder.transform([['female', 'from US', 'uses Safari']])
```

注意：这样的整数特征表示并不能在scikit-learn的estimator中直接使用，因为这样的连续输入， estimator会认为类别之间是有序的，但实际却是无序的。(例如：浏览器的类别数据则是任意排序的)。

### OneHotEncoder

该类把每一个具有 n_categories 个可能取值的 categorical特征变换为长度为 n_categories 的二进制特征向量，里面只有一个地方是1，其余位置都是0。

```python
from sklearn.preprocessing import OneHotEncoder

X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]

encoder = OrdinalEncoder()
encoder.fit(X)

# will get: array([[1., 0., 0., 1., 0., 1.], [0., 1., 1., 0., 0., 1.]])
encoder.transform([['female', 'from US', 'uses Safari']]).toarray()

# 默认情况下，每个特征使用几维的数值可以从数据集自动推断。而且可以在属性 categories_ 中找到:
# [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]
encoder.categories_
```

## 离散化

离散化 (Discretization) (有些时候叫量化(quantization) 或装箱(binning)) 提供了将连续特征划分为离散特征值的方法。 某些具有连续特征的数据集会受益于离散化，因为离散化可以把具有连续属性的数据集变换成只有名义属性(nominal attributes)的数据集。 (译者注： nominal attributes 其实就是 categorical features, 可以译为名称属性，名义属性，符号属性，离散属性等)

One-hot 编码的离散化特征可以使得一个模型更加的有表现力(expressive)，同时还能保留其可解释性(interpretability)。 比如，用离散化器进行预处理可以给线性模型引入非线性。

### K-bins 离散化

KBinsDiscretizer 类使用 k 个等宽的 bins 把特征离散化

```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal')
discretizer.fit(X)
```

默认情况下，输出是被 one-hot 编码到一个稀疏矩阵。而且可以使用参数 encode 进行配置。对每一个特征， bin 的边界以及总数目在 fit 过程中被计算出来，它们将用来定义区间。

### 特征二值化

特征二值化(Feature binarization) 是将数值特征用阈值过滤得到布尔值的过程。这对于pipeline中下游的概率型模型是有用的， 它们假设输入数据是多值伯努利分布(multi-variate Bernoulli distribution) 。例如这个模型 sklearn.neural_network.BernoulliRBM 就用到了多值伯努利分布 。

即使归一化计数(a.k.a. term frequencies) 和 TF-IDF值特征在实践中表现稍好一些， 在文本处理社区中也常常使用二值化特征值(这可能会简化概率推理(probabilistic reasoning))。

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0)
binarizer.fit(X)
```

请注意当 k = 2 ， 或者当 bin edge is at the value threshold 时， Binarizer 与 KBinsDiscretizer 是类似的。

# 缺失值补全

关于缺失值(missing values) 补全的方法和工具的讨论，请看章节：[缺失值处理(Imputation of missing values)](https://www.studyai.cn/modules/impute.html#impute "缺失值处理(Imputation of missing values)")

# 产生多项式特征

通常在机器学习中，通过使用一些输入数据的非线性特征来增加模型的复杂度是有用的。 一个简单通用的办法是使用多项式特征(polynomial features)，这可以获得特征的更高阶次和互相间关系的 项(terms)。 上述功能已经在 PolynomialFeatures 类中实现

```python
from sklearn.preprocessing import PolynomialFeatures

features = PolynomialFeatures(2)
features.fit_transform(X)
```

请注意，当我们使用多项式核核函 的时候， 多项式特征在 kernel methods 中被隐式的使用了。 (比如说, sklearn.svm.SVC, sklearn.decomposition.KernelPCA)。

# 自定义变换器

通常在机器学习中，你可能想要将一个已有的 Python 函数转化为一个变换器(transformer)来协助数据清理或处理。 你可以使用 FunctionTransformer 类从任意函数中实现一个transformer。例如，在一个pipeline中构建一个实现日志转换的transformer， 就可以这样做

```python
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p, validate=True)
transformer.transform(X)
```

# Question

Q：哪些机器学习模型必须进行特征缩放？

	通过梯度下降法求解的模型需要进行特征缩放，这包括线性回归（Linear Regression）、逻辑回归（Logistic Regression）、感知机（Perceptron）、支持向量机（SVM）、神经网络（Neural Network）等模型。此外，近邻法（KNN），K均值聚类（K-Means）等需要根据数据间的距离来划分数据的算法也需要进行特征缩放。主成分分析（PCA），线性判别分析（LDA）等需要计算特征的方差的算法也会受到特征缩放的影响。
	
	决策树（Decision Tree），随机森林（Random Forest）等基于树的分类模型不需要进行特征缩放，因为特征缩放不会改变样本在特征上的信息增益。

Q：进行特征缩放有哪些注意事项？

	需要先把数据拆分成训练集与验证集，在训练集上计算出需要的数值（如均值和标准值），对训练集数据做标准化/归一化处理（不要在整个数据集上做标准化/归一化处理，因为这样会将验证集的信息带入到训练集中，这是一个非常容易犯的错误），然后再用之前计算出的数据（如均值和标准值）对验证集数据做相同的标准化/归一化处理。

更多例子，请查看：

[预处理数据](https://www.studyai.cn/modules/preprocessing.html "预处理数据")

[Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html "Preprocessing data")