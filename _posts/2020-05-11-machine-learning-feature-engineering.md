---
title: auto-machine-learning-01
date: 2020-05-11 21:45:00 +0800
categories: [machine-learning]
tags: [sklearn]
comments: true
toc: true
---


# Foreword

在机器学习领域，整个行业朝着准入门槛更低、研发流程更敏捷方向发展，Auto-ML的出现意味着整个流程更加自动化。Auto-ML主要集中在特征工程自动化，模型选择自动化，模型训练及参数优化自动化，数据偏移检测自动化等。以下以python sklearn等模块，示例各个阶段如何进行自动化数据数据。

# Example Data

略。

# Feature Engineering

特征工程自动化指通过一些方法手段，自动从原始数据中提取出候选特征进行模型训练。

## feature tools

官方网站：https://github.com/FeatureLabs/featuretools

### 原理介绍

    生产环境中我们遇到的大部分数据都是按照结构化存储的。存储的数据类型可简单分为：数据实体（Entity）、数据实体关联关系（EntitySet）。不同的数据实体有不同的ID（EntityID），包括命名规则以及值域等。EntitySet则描述了不同EntityID的关联关系。
    
    Featuretools提供了基于DFS深度特征提取算法，自动发现EntitySet路径中不同实体的关联关系。在遍历数据时，使用预先定义的Aggregation和Transform，对Entity数据进行计算，生成合成特征。
    
    Featuretools侧重于特征计算处理，其最大的优点是定义了一套标准的特征处理元语：Aggregation、Transform，可以用于处理相同类型的数据。

官方例子已经比较详细，因此这里不再添加额外例子，可访问查看：https://docs.featuretools.com/en/stable/

## boruta

官方网站：https://github.com/scikit-learn-contrib/boruta_py

### 原理介绍

    Boruta主要用于探索特征变量相关性。生产环境建模中，不仅关注特征如何构建，还关注新增、删除特征后，模型评价指标的变化。通常对特征选择有以下两类结果：

> 1、删除某些特征后，导致模型评价指标变差，说明这些特征对于模型评价指标很重要；
> 2、删除某些特征后，模型评价指标无明显变化，说明这些特征对于模型评价指标不重要。

    以上结论中，结论1一定正确，结论2不一定正确，因为模型评价指标可能同因变量关系不大。Boruta提供了评估自变量（特征）同因变量重要关系的方法，从而能够更高效发现重要特征。
    
    Boruta将原特征进行随机shuffle从而构造出影子特征，将原始特征与影子特征拼接成新特征后，加入到训练数据中，作为特征矩阵参与模型训练，最后以影子特征的feature importance得分做为参考base, 从原始特征中选出与因变量真正相关的特征集合。

官方例子已经比较详细，因此这里不再添加额外例子，可访问查看：https://pypi.org/project/Boruta/

## tsfresh

官方网站：https://github.com/blue-yonder/tsfresh

### 原理介绍

    tsfresh适用于时间序列特征提取工具。该包包含多种特征提取方法和鲁棒特征选择算法，同时还可以对提取出的特征做重要性选择。
    
    tsfresh提供了以下处理能力：返回时序数据的绝对能量（平方和）、返回时序数据的一阶差分结果的绝对值之和、返回时序数据的各阶差分值之间的聚合（方差、均值）统计特征等等，时序数据处理中通用的特征处理计算方法。

官方例子已经比较详细，因此这里不再添加额外例子，可访问查看：https://tsfresh.readthedocs.io/en/latest/

## sklearn

官方网站：https://scikit-learn.org/

### 原理介绍

    sklearn提供了一整套涵盖从数据清洗、特征处理、特征选择、模型训练、评估以及可视化等流程功能，网上介绍资料比较多，这里不再赘述。

官方例子已经比较详细，因此这里不再添加额外例子，可访问查看：https://scikit-learn.org/stable/
