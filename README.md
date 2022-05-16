该程序结合haar小波变换和svdfree完成图像识别任务
程序主体在haar_svdfree_mnist.py，使用了一些辅助工具在tools.py中。
程序中用到的数据集在dataset/mnist/中。

程序主体部分包含主要函数和参数介绍：
0. 参数k表示正则化系数，参数p表示自编码隐层神经元数量，参数d表示输入数据特征维度
1. haar_wavelet()完成图像的哈尔小波变换
2. lightweightPilae()完成svdfree部分，该对象又包含以下几个主要函数
(1)svdfree()计算初始的编码器权重we0，并归一化处理得到we
(2)norm()归一化的实现函数，这里可以通过qr参数选择是否使用qr分解
3. activeFunc()编码器隐层神经元的激活函数，使用带偏移的阶跃函数。这里包含阈值参数epsilon，需要根据隐层激活前对角线和非对角线元素确定。
4.
