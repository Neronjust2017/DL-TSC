运行代码之前需要：
0. 安装kerastuner和autokeras(v1.0.0),可直接pip install autokeras==1.0.0（会自动安装kerastuner）
1. 将keras-tuner文件夹下的文件添加（替换）到包文件夹：site-packages/kerastuner/
2. 将auto-keras文件夹下的文件添加（替换）到包文件夹：site-packages/autokeras/
3. executions_per_trial=4(每个trial执行次数)可在 kerastuner/engine/multi_execution_tuner.py中修改
4. 实验记录在结果保存目录下的“log”中
