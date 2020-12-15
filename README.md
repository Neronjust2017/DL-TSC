# keras-project 
<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>
<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>
#### 项目简介：

一个使用Keras来构建和训练深度学习模型的项目：

1. 本项目采用数据类型为时序数据（Time Series Data），目标是在对时序数据进行分类或预测

2. 本项目使用Keras构建和训练用于两种任务（classification和regression）的深度学习模型

   ###### classification模型

   mlp、cnn、mcdcnn、encoder、tlenet、fcn、resnet、resnet_v2、resnext、inceptiontime

   ###### regression模型

   CNN、DeepConvLSTM、DeepConvLSTM_2、DeepResBiLSTM、LSTM、TCN、TCN_2、 TCN_3

3. 本项目中的模型训练流程划分为：加载数据、构建模型、训练、评估

4. 采用Tensorboard及Comet.ml可视化训练过程

5. 加入Microsoft NNI等超参数优化工具

# 目录

- [项目介绍](#项目介绍)

    - [项目结构](#项目结构)
    - [文件结构](#文件结构)
    - [主要组成部分](#主要组成部分)

- [项目运行](#项目运行)

- [Comet.ml设置](#Comet.ml设置)

- [数据集](#数据集)

- [Future Work](#future-work)

- [Example Projects](#example-projects)

- [Contributing](#contributing)

- [Acknowledgements](#acknowledgements)

    

  ​    



# 项目介绍

## 项目结构

<div align="center">
<img src="https://github.com/lixiaoyu0575/keras-project/blob/master/img/keras.png" alt="keras" style="zoom:60%;" />

</div>

## 文件结构

```
keras-project-master
├─ base						- 本目录包含几个重要组成部分的抽象类
│	├─ base_data_loader.py	- data_loader的抽象类
│	├─ base_evaluater.py	- evaluater的抽象类
│	├─ base_model.py		- model的抽象类
│	└─ base_trainer.py		- trainer的抽象类
|
├─ data_loader							  - 本目录包含data_loader类，加载相应任务数据	
│	├─ uts_classification_data_loader.py  - UtsClassificationDataLoader类
│	└─ uts_regression_data_loader.py	  - UtsRegressionDataLoader类
|
├─ models						- 本目录包含model类以及所有模型各自的类
│	├─ classification				- 本目录包含classification任务的所有model类
│	│	├─ cnn.py						- cnn模型
│	│	├─ encoder.py					- encoder模型
│	│	├─ fcn.py						- fcn模型
│	│	├─ inception.py					- inceptiontime模型
│	│	├─ mcdcnn.py					- mcdcnn模型
│	│	├─ mlp.py						- mlp模型
│	│	├─ resnet.py					- resnet模型
│	│	├─ resnet_v2.py					- resnet_v2模型
│	│	├─ resnext.py					- resnext模型
│	│	└─ tlenet.py					- tlenet模型
|   |
│	├─ regression					- 本目录包含regression任务的所有model类
│	│	├─ CNN.py						- CNN模型
│	│	├─ DeepConvLSTM.py				- DeepConvLSTM模型
│	│	├─ DeepConvLSTM_2.py			- DeepConvLSTM_2模型
│	│	├─ DeepResBiLSTM.py				- DeepResBiLSTM模型
│	│	├─ LSTM.py						- LSTM模型
│	│	├─ TCN.py						- TCN模型
│	│	├─ TCN_2.py						- TCN_2模型
│	│	├─ TCN_3.py						- TCN_3模型
|   |
│	├─ uts_classification_model.py		- UtsClassificationModel类
│	└─ uts_regression_model.py			- UtsRegressionModel类
|
├─ trainers								- 本目录包含trainer类
│	├─ uts_classification_trainer.py		- UtsClassificationTrainer类
│	└─ uts_regression_trainer.py			- UtsRegressionModelTrainer类
|
├─ evaluater							- 本目录包含evaluater类						
│	├─ uts_classification_evaluater.py		- UtsClassificationEvaluater类
│	└─ uts_regression_evaluater.py			- UtsRegressionEvaluater类
|
├─ configs								- 本目录包含所有config文件	
│	├─ custom								- 本目录包含所有添加了comet api key的config文件
│	│	├─ uts_classification.json				- classification任务
│	│	└─ uts_regression.json					- regression任务
|   |
│	├─ simple_mnist_config.json				- 示例
│	└─ simple_mnist_from_config.json		- 示例
|
├─ utils					- 本目录包含所有工具类
│	├─ args.py					- run arguments的处理方法
│	├─ config.py				- json文件的处理方法，解析参数并生成实验保存目录
│	├─ dirs.py					- create_dirs方法，创建文件夹
|   |
│	├─ uts_classification的		- 本目录包含uts_classification的工具类等
│	│	├─ metric.py				- recall、precision、f1的计算方法
│	│	└─ utils.py					- classification任务的utils
|   |
│	└─ uts_regression			 - 本目录包含uts_regression的工具类等
│	 	├─ data.py					- Data_utility，进行数据的处理
│	 	├─ tools.py					- 自定义的LearnignRateScheduler
│	 	└─ utils.py					- regression任务的utils
|
├─ opt							 - 本目录包含Optimization相关内容
│	├─ BayesianOptimization			- 本目录包含BayesianOptimization相关内容
│	│	└─ ...
│	└─ nni							- 本目录包含nni相关内容
│	 	└─ ...
|
├─ datasets						 - 本目录包含所有数据集
│    ├─ mts_data					- 本目录包含所有mts多变量时序数据集
│    └─ uts_data					- 本目录包含所有uts单变量时序数据集
|
├─ experiments					 - 本目录包含所有实验结果
│    ├─ 2019-12-01					- 按日期保存
│    ├─ ...
│    ├─ 2020-01-04
│    └─ ...
|
├─ main.py							- 示例
├─ main_classification.py			- classification任务的main方法
├─ main_forecasting.py				- forecasting/regression任务的main方法	
├─ nni_config.yml					- nni的配置文件
├─ nni_main_classification.json		- nni-classification超参数优化的config文件
├─ nni_main_classification.py		- nni进行classification任务超参数优化
├─ nni_search_space.json			- nni超参数搜索空间
└─ vis_training_size.py				- 可视化评价指标随训练集大小的变化情况

```

## 主要组成部分

### Data Loaders

两个Data Loader类：**UtsClassificationDataLoader** 、 **UtsRegressionDataLoader**，特点是：

1. 所有的DataLoader都继承自**BaseDataLoader**
2. 重写了BaseDataLoader中的 ***get_train_data()*** 和***get_test_data()***方法，以得到各自相应的训练数据和测试数据
3. 在构造函数中完成主要的逻辑，包括按不同数据集读取数据，并处理成模型的标准输入形式

### Models

两个Model类：**UtsClassificationModel**、**UtsRegressionModel**，特点是：

1. 所有的Model都继承于**BaseModel**
2. 重写了BaseModel中的 ***build_model*** 方法，以实现各自相应模型的构建
3. 在构造函数中调用 ***build_model*** 方法来构建模型，该方法从config中获得所要构造的模型名称，并交由相应的模型类来构建模型

### Trainers

两个Model类：**UtsClassificationTrainer**、**UtsRegressionModelTrainer**，特点是：

1. 所有的Model都继承自**BaseTrainer**
2. 重写了BaseTrainer中的 ***train*** 方法，以实现各自不同的训练方法
3. 在构造函数中初始化训练过程中的各种指标值的list，并调用***init_callbacks()***方法初始化***callback***函数
4. 在***train***方法中将***callback***数组传递给模型对象的***fit***/***fit_generator***函数
5. ***train***方法实现模型的训练，并使用***save_training_logs***方法保存training logs

### Evaluaters

两个Evaluater类：**UtsClassificationEvaluater**、**UtsRegressionEvaluater**，特点是：

1. 所有的Model都继承自**BaseEvaluater**
2. 重写了BaseEvaluater中的 ***evluate***方法，以实现各自不同的评估方法
3. 在构造函数中接受**DataLoader**传过来的测试数据集与**Trainer**传过来的***best model***
4. 在***evaluate***方法中在测试数据集上对模型进行评估，并使用**save_evaluating_result**方法保存评估结果

### Configs

每一个任务/实验对应一个.json文件，其中包含了实验和模型的配置，如exp_name(实验名)、dataset_name(数据集名)、model_name(模型名)、learning_rate(学习率)、num_epochs(epoch数目)等。通过改变json文件中参数的值，就可以选择不同的数据、模型以及训练参数，方便地进行不同的实验。

**例子：mts_classification.json**

```
{
  "exp": {
    "name": "mts_classification"		- 实验名称
  },
  "dataset":{							- 数据集
    "type": "mts",							- 类型：多变量时序数据
    "name": "UCI_HAR_Dataset"				- 名称：UCI_HAR_Dataset
  },
  "model":{								- 模型
    "name":"resnet",						- 名称：resnet
    "learning_rate": 0.0001,				- 学习率：0.0001
    "optimizer": "adam"						- 优化器：adam
  },
  "trainer":{							- 训练
    "num_epochs": 100,						- epoch数目：100
    "batch_size": 108,						- batch大小：108
    "validation_split":0.25,				- 验证集划分比例：0.25
    "verbose_training": true				- 显示训练过程日志：true
  },
  "callbacks":{							- callback函数
    "checkpoint_monitor": "val_loss",		- ModelCheckpoint参数
    "checkpoint_mode": "min",				- ...
    "checkpoint_save_best_only": true,		- ...
    "checkpoint_save_weights_only": true,	- ...
    "checkpoint_verbose": true,				- ...
    "tensorboard_write_graph": true			- Tensorboard参数
  },
  "comet_api_key": "..."					- comet api key
```

### Main

整个流程pipeline的构建：

1. 解析config文件
2. 创建一个dataLoader的实例
3. 创建一个model的实例
4. 创建一个trainer的实例
5. 调用trainer对象的***train()***方法进行训练
6. 创建一个evaluater的实例
7. 调用evaluater对象的***evluate()***方法进行评估

**例子：main_classification**

```
def main():

    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    config = process_config_UtsClassification(args.config)
    
    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir,
                 config.log_dir, config.result_dir])

    print('Create the data generator.')
    data_loader = UtsClassificationDataLoader(config)

    print('Create the model.')

    model = UtsClassificationModel(config, data_loader.get_inputshape(), data_loader.get_nbclasses())

    print('Create the trainer')
    trainer = UtsClassificationTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Create the evaluater.')
    evaluater = UtsClassificationEvaluater(trainer.best_model, data_loader.get_test_data(), data_loader.get_nbclasses(), config)

    print('Start evaluating the model.')
    evaluater.evluate()
    print('done')
```



# 项目运行
#### 例1.时序数据上的classification任务：

1. 编辑uts_classification.json文件，配置实验参数
2. 运行项目:

```shell
python main_classification.py -c configs/uts_classification.json
```

2. 查看训练log、learning cruves、metrics、confusion matrix等
3. 利用Tensorboard可视化训练过程：
```shell
tensorboard --logdir experiments/.../tensorboard_logs
```

<div align="center">
<img align="center" width="600" src="https://github.com/lixiaoyu0575/keras-project/blob/master/img/Tensorboard_demo.png">

</div>

#### 例2. 使用NNI进行classification任务超参数优化：

1. 编辑nni_config.yml文件，配置NNI实验的参数
2. 编辑nni_main_classification.json文件，指定classification任务的固定参数
3. 编辑nni_search_space.json文件，指定classification任务中的超参数的搜索空间

2. 在命令行运行NNI:

```shell
nnictl create --config nni_config.yml
```

2. 在浏览器打开NNI Web UI ：

   <img align="center" width="600" src="https://github.com/lixiaoyu0575/keras-project/blob/master/img/nni_cmd.png">

3. 查看实验进行情况：

   <img align="center" width="600" src="https://github.com/lixiaoyu0575/keras-project/blob/master/img/nni_webui.png">


# Comet.ml设置
This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.

Add your API key [in the configuration file](configs/simple_mnist_config.json#L15):


For example:  `"comet_api_key": "your key here"`

You can also link your Github repository to your comet.ml project for full version control.

# 数据集

这里提供两个基本的时序数据集，其他数据集也可以加入到本项目的datasets文件夹下，并在data_loader里添加相应的load代码。

* [uts_data](http://www.timeseriesclassification.com/dataset.php) (Univariate Weka formatted ARFF files and .txt files)
* [mts_data](http://www.timeseriesclassification.com/dataset.php) (Multivariate Weka formatted ARFF files (and .txt files))


# Future Work
完善超参数优化部分

# Example Projects

一个简单的示例项目：

* [Toxic comments classification using Convolutional Neural Networks and Word Embedding](https://github.com/Ahmkel/Toxic-Comments-Competition-Kaggle)

# Contributing

Any contributions are welcome including improving the  project.

# Acknowledgements

This project template is based on [MrGemy95](https://github.com/MrGemy95)'s [Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template).

project template： https://github.com/Ahmkel/Keras-Project-Template
