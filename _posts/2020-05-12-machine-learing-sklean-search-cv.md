---
title: machine-learning-sklearn-search-cv
date: 2020-05-11 21:45:00 +0800
categories: [machine-learning]
tags: [sklearn]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAyMC0wNi0wOQ==
---


# Foreword

sklearn模块提供了自动调参功能，这里调整的参数是模型的超参，而非训练过程中特征的权重参数。这些超参数需要在训练前先验的设置，因此可以使用对应方法进行自动调参。

官方网站：https://scikit-learn.org/stable/modules/grid_search.html

# GridSearchCV

GridSearchCV包含了两个概念，网格搜索（Grid search）和CV（Cross Validation）。网格搜索是指对给定的超参值集合，遍历每一个可选的超参值，在超参集合中找到模型评价指标最好的超参数。CV将训练集合划分为K份（K-Fold），依次取其中一份做为测试集，其余做为训练集，进行模型评价指标评估，取K次训练的平均评价指标为模型的评价指标。

用法：

```python
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 导入训练数据
train_data = pd.read_csv("train.csv", sep=',')
train_data.set_index('id', drop=True, inplace=True)

train_x = train_data
train_y = train_data['y']

del train_x['y']

print("train data:", train_x.shape, train_y.shape)

# 分类器使用xgboost
model = xgb.XGBClassifier()

# 设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
    'colsample_bytree': np.linspace(0.5, 0.99, 5),
    'learning_rate': np.linspace(0.01, 1, 5),
    'max_depth': range(1, 15, 3),
    'min_child_weight': range(1, 9, 3),
    'n_estimators': range(10, 20, 3),
    'subsample': np.linspace(0.7, 0.9, 5),
}

scoring = 'neg_log_loss'

# GridSearchCV参数说明
# param_dist字典类型，放入参数搜索范围
# scoring = neg_log_loss，精度评价方式
# n_jobs = -1，使用所有的CPU进行训练
# cv = 10，使用10折交叉验证
search_result = GridSearchCV(model, param_dist, cv=10, scoring=scoring, n_jobs=-1, verbose=1)

print("begin search ...")
search_time_start = time.time()

# 在训练集上训练
search_result.fit(train_x.values, np.ravel(train_y.values))

print("search time:", time.time() - search_time_start)

# 详细cv结果
cv_results = search_result.cv_results_

# 最优scoring
best_score = search_result.best_score_
print("best score: {}".format(best_score))

# 最优params
best_params = search_result.best_params_
for param_name in sorted(best_params.keys()):
    print('bast params: key=[%s], value=[%r]' % (param_name, best_params[param_name]))
```

# RandomizedSearchCV

如前所述，GridSearchCV采用遍历超参集合方案，当待搜索超参数目过多，效率十分低下，因此可以考虑使用RandomizedSearchCV方法。RandomizedSearchCV在给定的参数空间中，进行随机采样，如果输入的超参列表有连续变量时会被当作一个分布进行采样，这样整个搜索效率就提升很快。

用法：

```python
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# 导入训练数据
train_data = pd.read_csv("train.csv", sep=',')
train_data.set_index('id', drop=True, inplace=True)

train_x = train_data
train_y = train_data['y']

del train_x['y']

print("train data:", train_x.shape, train_y.shape)

# 分类器使用xgboost
model = xgb.XGBClassifier()

# 设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
    'colsample_bytree': np.linspace(0.5, 0.99, 5),
    'learning_rate': np.linspace(0.01, 1, 5),
    'max_depth': range(1, 15, 3),
    'min_child_weight': range(1, 9, 3),
    'n_estimators': range(10, 20, 3),
    'subsample': np.linspace(0.7, 0.9, 5),
}

scoring = 'neg_log_loss'

# GridSearchCV参数说明
# param_dist字典类型，放入参数搜索范围
# scoring = neg_log_loss，精度评价方式
# n_jobs = -1，使用所有的CPU进行训练
# cv = 10，使用10折交叉验证
search_result = RandomizedSearchCV(model, param_dist, cv=2, scoring=scoring, n_jobs=-1, verbose=1)

print("begin search ...")
search_time_start = time.time()

# 在训练集上训练
search_result.fit(train_x.values, np.ravel(train_y.values))

print("search time:", time.time() - search_time_start)

# 详细cv结果
cv_results = search_result.cv_results_

# 最优scoring
best_score = search_result.best_score_
print("best score: {}".format(best_score))

# 最优params
best_params = search_result.best_params_
for param_name in sorted(best_params.keys()):
    print('bast params: key=[%s], value=[%r]' % (param_name, best_params[param_name]))
```

对比搜索时间，相同搜索空间下，RandomizedSearchCV可以比GridSearchCV快10倍左右，意味着可以使用RandomizedSearchCV做更精细粒度参数搜寻。
