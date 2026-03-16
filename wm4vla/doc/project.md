# 项目简介
world model 4 vla(wm4vla)
采取双gpu策略，在一张卡上跑world model生成观测值，传给vla生成action，而不需要等待执行完action后再传回观测值，实现机器人一边执行动作，模型一边推理的异步机制。

现在只需要通过仿真环境（libero, kinetix），模拟推理延迟，实现机器人动作异步推理的仿真实验：

根据benchmark，训练一个生成模型，实现传入当前状态，输出下一次的观测信息，不需要等待机器人传回执行完动作后的状态信息，再把生成的观测信息传给vla，生成action给机器人执行。

D:  Inference_delay=[1,2,3,4]
D-1=d 为wm内部delay
1. wm  ：输入obs_{t},  action_{t+D-1}, D-1,  输出 obs_{t+D}
2. vla 输入obs_{t+D}， state_{t+D}，输出action_{t+D, t+D+1,…}
    state_{t+D}用上一次预测的action_{t+D-1}（LIBERO的action与state维度不匹配，仿真模拟一下，直接使用state_{t+D}）
3. action_to_execute重构：D推理时间段用上一次生成的action_to_execute剩下的，后续用新生成的


## 仿真实验1
model: pi0, pi05; 仿真环境：libero

在/home/kyji/storage_net/tmp/lbai/vlash/vlash/eval_libero.py的基础上，改成我们的wm4vla：即使用world model生成观测值，传入vla的模型，其他实验设置保持一致。

world model需要根据现有的cosmos-predict2.5模型在libero数据集上进行post-training
policy不需要根据wm4vla架构进行后训练

## 仿真实验2
model: flow policy; 仿真环境：kinetix

在/home/kyji/storage_net/tmp/lbai/real-time-chunking-kinetix的基础上，改成我们的wm4vla：即使用world model生成观测值，传入vla的模型，其他实验设置保持一致。


## 实现说明
目前teacher模型在cosmos-predict2.5目录下训练，
而蒸馏步骤在wm-cosmos-predict2.5目录下进行训练，

需要teacher部分时参考cosmos-predict2.5目录

## world model
不使用视频格式，只需要生成未来图片
传入condition imgage，生成future iamge 
lerobot-libero有两个视角的图片，组成一个Batch，这样就能同时生成两个视角的图片了