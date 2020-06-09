---
title: machine-learning-nni
date: 2020-05-11 21:45:00 +0800
categories: [machine-learning]
tags: [nni]
comments: true
toc: true
sitemap:
  lastmod: !!binary |
    MjAyMC0wNi0wOA==
---


# Foreword

自动调参可以极大的提升模型构建自动化程度，NNI是微软提供的优秀框架，帮助用户自动的进行特征工程，神经网络架构搜索，超参调优以及模型压缩。

官方网站：https://github.com/microsoft/nni/blob/master/README_zh_CN.md

# 原理介绍

开始之前一些术语介绍：

| 概念           | 说明      |
| :----------- | :------ |
| Experiment | 实验表示一次任务，用来寻找模型的最佳超参组合，或最好的神经网络架构等。由Trial和自动机器学习算法所组成。|
| 搜索空间 | 是模型调优的范围。例如，超参的取值范围。 |
| Configuration | 配置是来自搜索空间的实例，每个超参都会有特定的值。 |
| Trial | 是一次独立的尝试，它会使用某组配置（例如，一组超参值，或者特定的神经网络架构）。Trial会基于提供的配置来运行。 |
| Tuner | 调优器，自动机器学习算法，会为下一个Trial生成新的配置。新的Trial会使用这组配置来运行。 |
| Assessor | 评估器分析Trial的中间结果（例如，定期评估数据集上的精度），来确定Trial是否应该被提前终止。 |
| 训练平台 | Trial的执行环境。根据Experiment的配置，可以是本机，远程服务器组，或其它大规模训练平台（如，OpenPAI，Kubernetes）。 |

NNI主要运行流程如下：

```text
输入: 搜索空间, Trial代码, 配置文件
输出: 一组最佳的超参配置

1: For t = 0, 1, 2, ..., maxTrialNum,
2:      hyperparameter = 从搜索空间选择一组参数
3:      final result = run_trial_and_evaluate(hyperparameter)
4:      返回最终结果给 NNI
5:      If 时间达到上限,
6:          停止实验
7: return 最好的实验结果
```

# 自动超参数调优

如果需要使用 NNI 来自动训练模型，找到最佳超参，需要根据代码，进行如下三步改动：

https://github.com/microsoft/nni/tree/master/examples/trials/mnist-tfv1

第一步：定义 JSON 格式的搜索空间文件，包括所有需要搜索的超参的名称和分布（离散和连续值均可）。

```python
-   params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-   'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+ {
+     "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
+     "conv_size":{"_type":"choice","_value":[2,3,5,7]},
+     "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
+     "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
+     "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
+ }
```

第二步：修改 Trial 代码来从 NNI 获取超参，并返回 NNI 最终结果。

```python
+ import nni

  def run_trial(params):
      mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)

      mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'], channel_2_num=params['channel_2_num'], conv_size=params['conv_size'], hidden_size=params['hidden_size'], pool_size=params['pool_size'], learning_rate=params['learning_rate'])
      mnist_network.build_network()

      with tf.Session() as sess:
          mnist_network.train(sess, mnist)
          test_acc = mnist_network.evaluate(mnist)

+         nni.report_final_result(test_acc)

  if __name__ == '__main__':

-     params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-     'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+     params = nni.get_next_parameter()
      run_trial(params)
```

第三步：定义 YAML 格式的配置文件，其中声明了搜索空间和 Trial 文件的路径。 它还提供其他信息，例如调整算法，最大 Trial 运行次数和最大持续时间的参数。

```yaml
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: local
# 搜索空间文件
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
# 运行的命令，以及 Trial 代码的路径
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
```

然后，运行：

```shell
nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

更多信息，可以查看官方文档：https://nni.readthedocs.io/zh/latest/Tutorial/QuickStart.html

# 神经网络架构搜索

通过两个 NAS API LayerChoice 和 InputChoice 来定义神经网络模型，而不需要编写具体的模型。 本质上还是遍历。

# 模型压缩

NNI 提供了几种压缩算法，包括剪枝和量化算法：剪枝算法通过删除冗余权重或层通道来压缩原始网络，从而降低模型复杂性并解决过拟合问题；量化算法通过减少表示权重或激活所需的精度位数来压缩原始网络，这可以减少计算和推理时间。

# 特征工程

试验阶段，待后续完善后可使用。
