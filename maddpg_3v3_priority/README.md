### Requirement
gym==0.10.5

Pillow==7.1.1

tensorflow==1.14.0

maddpg==0.0.1 在maddpg文件夹下安装:

cd ./maddpg

pip install -e .

multiagent==0.0.1 在multiagent-particle-envs文件夹下安装:

cd ./multiagent-particle-envs

pip install -e .

### 主要添加的功能：
#### 1、智能体死亡退出
#### 2、训练中当所有蓝方死亡后，开始下一轮实验
#### 3、场景是3v3的
#### 4、训练时每隔一定训练次数输出胜率
#### 5、蓝方随机运功

在./maddpg/experiments目录下可运行测试用例：

cd ./maddpg/experiments

训练运行命令:
python train.py --load-dir ./test

测试运行命令:
python train.py --load-dir ./test --display --save-gif

>以下是来自张冠宇代码测试的问题
>
>3v3 初步成功版本存在的问题：红方在两个蓝方之间权衡
>
>V4.0
蓝方采用MADDPG训练，目标是不出边界
红方采用MADDPG训练，能够很好的抓住蓝方
2V2
>
>v5.0
增加观察状态量
3v3 对抗规则胜率80%
>
>V6.0
蓝方加载红方模型
训练红方对抗
结果可以完胜规则和新的蓝方
>
>V7.0
更正攻击角度没有加绝对值的bug
红方多了一个lock_time
>
### Reard Settings
#### 红方
1 distance -0.1*min_dis

2 红方任何人抓住一个蓝方 +10

3 红方离开屏幕，-50

4 抓住所有蓝方 +100


#### 蓝方

1、距离 +0.1 * 距离所有红方的距离

2、被抓住 -10

3、离开屏幕 -5