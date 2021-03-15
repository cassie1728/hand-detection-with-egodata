# hand-detection-with-egodata
使用egohands公开数据集用yolov5模型训练通用手部检测模型


## 1. 解压处理egohands公开数据集
数据集链接：http://vision.soic.indiana.edu/projects/egohands/

使用egohands_dataset_clean.py代码解压。

解压得到数据图片以及对应的csv文件，要适应yolov5的标注格式需要进行转化。


## 2.下载训练模型
百度云链接：链接:https://pan.baidu.com/s/1dXxhw4pF5wwmX9ivoagP5w  密码:xoyu

将下载的模型放到weights文件夹中。

## 3.环境配置
参考yolov5源代码的配置

本项目使用的版本如下：
cuda10.2
python3.8.5
pytorch1.7.0

