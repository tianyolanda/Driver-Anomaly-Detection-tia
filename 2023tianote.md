#　向日葵
/usr/local/sunlogin/bin/sunloginclient

# 2024.3.8记录
电脑空间不足了,所以把一些文件剪切备份到其他地方
2023年训练的checkpoint放在了:/home/ubuntu/disk2/wyt-DADcode-backup
数据集的10个压缩包: /media/ubuntu/TU200Pro/DADdataset压缩包

# zed相机实时demo
/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/demo_live_onemodality_mobilenetv2se_zed.py
写了代码,可以运行实时的demo,但是score是nan,无法分辨驾驶行为
猜测原因有:
1. zed相机得到的深度图和训练集用的infineon相机的深度图在角度\距离\像素值都有较大差别;
2. 16帧叠加的顺序目前是最新的在最后,不确定是否正确?

## 读取zed深度图代码
/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/zed_depth.py

# astra相机实时demo
https://developer.orbbec.com.cn/technical_library.html?id=50
详见astra笔记

# 记录sh (已变更,未更新)
tia-run1.sh 训练mobilenetv2se (各种reduction)

tia-run2.sh 训练resnet
 
tia-run3.sh 训练mobilenetv2, 大bs

tia-run4.sh 训练mobilenetv2, 小bs

tia-run5.sh 测试mobilenetv2, 小bs

tia-run6.sh 训练mobilenetv2se (各种reduction)

测试单模态运行速度fps:
python demo_live_onemodality_mobilenetv2se.py

# start from 2023.6.9
##　待办
1. 统计模型参数量
        #print(len(pretrained_dict.keys()))
2. 统计模型fps (已统计)


## mobilenetv2se
1. num_worker = 1 
2. 加了SE之后内存溢出，没解决，只能每70个epoch　resume训练

## 运行Driver-Anomaly-Detection遇到的问题
1. 数据集解压 error
unrar x DAD.part10.rar
Total errors: 4

2. 直接运行demo_live.py 报killed
因为cpu的memory不够,
硬件问题,给16g内存条换了位置就识别出来了

3. --mode test 报错缺图片
# 1
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/datasets/DADdataset/DAD/DAD/Tester20/normal_driving_4/front_depth/img_6333.png'
copy了6332的图

# 2
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/datasets/DADdataset/DAD/DAD/Tester9/normal_driving_4/front_IR/img_5545.png'

# 3
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/datasets/DADdataset/DAD/DAD/Tester23/normal_driving_5/top_IR/img_7006.png'

# 4
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/datasets/DADdataset/DAD/DAD/Tester25/normal_driving_5/top_IR/img_518.png'

## 备忘
1. conda activate torch140


