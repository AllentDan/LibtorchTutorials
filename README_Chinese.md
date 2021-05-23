[English](https://github.com/AllentDan/LibtorchTutorials/blob/main/README.md) | 中文

<div align="center">

![logo](https://raw.githubusercontent.com/AllentDan/ImageBase/main/detection/LibtorchTutorials.png)  
**一个libtorch教学的代码仓库，包含使用libtorh (or pytorch c++)的教程.**  

</div>

**如果有用，别忘了给我star**
## 环境
- win10
- visual sutdio 2017 或者Qt4.11.0
- Libtorch 1.7
- Opencv4.5

## 配置
[libtorch+Visual Studio](https://allentdan.github.io/2020/12/16/pytorch%E9%83%A8%E7%BD%B2torchscript%E7%AF%87)和[libtorch+QT](https://allentdan.github.io/2021/01/21/QT%20Creator%20+%20Opencv4.x%20+%20Libtorch1.7%E9%85%8D%E7%BD%AE/#more)分别记录libtorch在VS和Qt的开发环境配置。

## 介绍
注意，这个项目不是直接利用torch::jit加载.pt模型预测，而是用c++构建深度学习模型，在c++中训练预测。

本教程旨在教读者如何用c++搭建模型，训练模型，根据训练结果预测对象。为便于教学和使用，本文的c++模型均使用libtorch（或者pytorch c++ api）完成搭建和训练等。目前，国内各大平台似乎没有pytorch在c++上api的完整教学，也没有基于c++开发的完整的深度学习开源模型。可能原因很多：

1. c/c++的深度学习已经足够底层和落地，商用价值较高，开发难度偏大，一般不会开源；
2. 基于python训练，libtorch预测的部署形式足够满足大多数项目的需求，如非产品级应用，不会有人愿意研究如何用c++从头搭建模型，实现模型训练功能；
3. Tensorflow的市场份额，尤其时工业应用的部署下市场占比足够高，导致基于libtorch的开发和部署占比很小，所以未见开源。

本教程提供基于libtorch的c++开发的深度学习视觉模型教学，本教程的开发平台基于Windows环境和Visual Sutdio集成开发环境，实现三大深度视觉基本任务：分类，分割和检测。适用人群为：
- 有c++基础，了解面向对象思维，
- 有pytorch编程经验者。

## 章节安排
本教程分多个章节：
- 第一章：[libtorch+Visual Studio](https://allentdan.github.io/2020/12/16/pytorch%E9%83%A8%E7%BD%B2torchscript%E7%AF%87)和[libtorch+QT](https://allentdan.github.io/2021/01/21/QT%20Creator%20+%20Opencv4.x%20+%20Libtorch1.7%E9%85%8D%E7%BD%AE/#more)开发环境搭建
- 第二章：[张量的常规操作](https://allentdan.github.io/2021/01/16/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%BA%8C%EF%BC%89/)
- 第三章：[简单MLP,CNN,LSTM模型搭建](https://allentdan.github.io/2021/01/16/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%B8%89%EF%BC%89/)
- 第四章：[数据加载模块使用](https://allentdan.github.io/2021/01/18/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E5%9B%9B%EF%BC%89/)
- 第五章：[分类模型搭建，训练和预测](https://allentdan.github.io/2021/01/18/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%BA%94%EF%BC%89/)
- 第六章：[分割模型搭建，训练和预测](https://allentdan.github.io/2021/02/10/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E5%85%AD%EF%BC%89/)
- 第七章：[目标检测模型搭建，训练和预测](https://allentdan.github.io/2021/05/03/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%B8%83%EF%BC%89/)
- 第八章：[总结](https://allentdan.github.io/2021/05/03/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E5%85%AB%EF%BC%89/)

## 相关项目
基于libtorch，我释放了如下开源项目:
- [LibtorchTutorials](https://github.com/AllentDan/LibtorchTutorials)
- [LibtorchSegmentation](https://github.com/AllentDan/LibtorchSegmentation)
- [LibtorchDetection](https://github.com/AllentDan/LibtorchDetection)

挂下自己微信吧，may88happy，最近很多小伙伴有问题。建了个群，加了拉群里直接问，我看到一般都会回复的
