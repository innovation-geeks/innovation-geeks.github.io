---
title: machine-learning-sklearn-search-cv
date: 2020-05-11 21:45:00 +0800
categories: [machine-learning]
tags: [sklearn]
comments: true
toc: true
---


# Foreword

sklearn模块提供了自动调参功能，这里调整的参数是模型的超参，而非训练过程中特征的权重参数。这些超参数需要在训练前先验的设置，因此可以使用对应方法进行自动调参。

官方网站：https://scikit-learn.org/stable/modules/grid_search.html