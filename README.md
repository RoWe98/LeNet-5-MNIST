# 基于LeNet-5的MNIST手写数字识别

## 项目实验环境

|系统|显卡|处理器|Cuda版本|Tensorflow版本|
|--|--|--|--|--|
|Windows10 Pro|Nvidia RTX2070Super|Intel core i5 9600KF|10.1|Tensorflow-GPU 2.3|


1990 年代，Yann LeCun 等人提出了用于手写数字和机器打印字符图片识别的神经网
络，被命名为 LeNet-5 [4]。LeNet-5 的提出，使得卷积神经网络在当时能够成功被商用，
广泛应用在邮政编码、支票号码识别等任务中。下图是 LeNet-5 的网络结构图，它
接受32 × 32大小的数字、字符图片，经过第一个卷积层得到 28 28 形状的张量，经过
一个向下采样层，张量尺寸缩小到 ，经过第二个卷积层，得到 形状
的张量，同样经过下采样层，张量尺寸缩小到 ，在进入全连接层之前，先将张量
打成 的张量，送入输出节点数分别为 120、84 的 2 个全连接层，得到 8 的张
量，最后通过 Gaussian connections 层。

![LeNet-5](https://s1.ax1x.com/2020/08/08/aIFPqH.png)

现在看来，LeNet-5 网络层数较少(3 个卷积层和 2 个全连接层)，参数量较少，计算代
价较低，尤其在现代 GPU 的加持下，数分钟即可训练好 LeNet-5 网络。
我们在 LeNet-5 的基础上进行了少许调整，使得它更容易在现代深度学习框架上实
现。首先我们将输入𝑿形状由32 × 32调整为28 × 28，然后将 2 个下采样层实现为最大池化
层(降低特征图的高、宽，后续会介绍)，最后利用全连接层替换掉 Gaussian connections
层。下文统一称修改的网络也为 LeNet-5 网络。网络结构如下所示。

![网络结构](https://s1.ax1x.com/2020/08/08/aIFyJx.png)


## 1.网络定义
```python
network = Sequential([
    # 卷积层1
    layers.Conv2D(filters=6,kernel_size=(5,5),activation="relu",input_shape=(28,28,1),padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    
    # 卷积层2
    layers.Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same"),
    layers.MaxPool2D(pool_size=2,strides=2),
    
    # 卷积层3
    layers.Conv2D(filters=32,kernel_size=(5,5),activation="relu",padding="same"),
    
    layers.Flatten(),
    
    # 全连接层1
    layers.Dense(200,activation="relu"),
    
    # 全连接层2
    layers.Dense(10,activation="softmax")    
])
network.summary()
```

## 2.网络结构

这时可以看见我们的网络结构以及参数信息如下所示:
```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 6)         156
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 16)        2416
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 16)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 32)          12832
_________________________________________________________________
flatten (Flatten)            (None, 1568)              0
_________________________________________________________________
dense (Dense)                (None, 200)               313800
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2010
=================================================================
Total params: 331,214
Trainable params: 331,214
Non-trainable params: 0
_________________________________________________________________
```

## 3.加载MNIST数据集

一般机器学习框架都使用MNIST作为入门。就像"Hello World"对于任何一门编程语言一样，要想入门机器学习，就先要掌握MNIST。

<div style="text-align: center;">

![MNIST数据集](https://s1.ax1x.com/2020/08/08/aIkmkR.png)
</div>

该数据集包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，其值为0到1。为简单起见，每个图像都被平展并转换为784(28 * 28)个特征的一维numpy数组。

***加载和处理数据集***
```python
mnist = tf.keras.datasets.mnist
(trainImage, trainLabel),(testImage, testLabel) = mnist.load_data()
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

trainImage = tf.reshape(trainImage,(60000,28,28,1))
testImage = tf.reshape(testImage,(10000,28,28,1))
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)
```

## 4.模型训练和保存

利用keras接口的complie函数和fit函数进训练以及将训练好的模型进行保存

```python
# 模型训练
network.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
network.fit(trainImage,trainLabel,epochs=30,validation_split=0.1)

# 模型保存
network.save('./lenet_mnist.h5')
print('lenet_mnist model saved')
del network
```

此时开始训练，训练30个epoch后得到准确率大概为98% **训练使用GPU训练，tensorflow为GPU版本**

```python
Epoch 1/30
2020-08-08 17:17:14.928899: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cublas64_10.dll
2020-08-08 17:17:15.130821: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cudnn64_7.dll
2020-08-08 17:17:15.771680: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] Internal: Invoking GPU asm compilation is supported on Cuda non-Windows platforms only
Relying on driver to perform ptx compilation.
Modify $PATH to customize ptxas location.
This message will be only logged once.
1688/1688 [==============================] - 4s 2ms/step - loss: 0.2784 - accuracy: 0.9444 - val_loss: 0.0678 - val_accuracy: 0.9793
Epoch 2/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0664 - accuracy: 0.9796 - val_loss: 0.0669 - val_accuracy: 0.9815
Epoch 3/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0525 - accuracy: 0.9834 - val_loss: 0.0508 - val_accuracy: 0.9862
Epoch 4/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0441 - accuracy: 0.9861 - val_loss: 0.0648 - val_accuracy: 0.9835
Epoch 5/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0437 - accuracy: 0.9869 - val_loss: 0.0548 - val_accuracy: 0.9835
···············································································································
Epoch 27/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0317 - accuracy: 0.9941 - val_loss: 0.1511 - val_accuracy: 0.9873
Epoch 28/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0308 - accuracy: 0.9951 - val_loss: 0.1368 - val_accuracy: 0.9852
Epoch 29/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0350 - accuracy: 0.9946 - val_loss: 0.1468 - val_accuracy: 0.9813
Epoch 30/30
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0281 - accuracy: 0.9953 - val_loss: 0.1790 - val_accuracy: 0.9872
lenet_mnist model saved
```

## 5.模型测试

这里随机在MNIST数据集中选取25张图片进行预测，为了检查预测结果和原图是否一致，使用了matplotlib来显示前25张图片

```python
mnist = tf.keras.datasets.mnist
(testImage, testLabel) = mnist.load_data()
 
for i in [testImage,testLabel]:
    print(i.shape)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(testImage[i], cmap='gray')
plt.show()
```

这里得到前25张图片如下所示

<div style="text-align: center;">

![前25张图片](https://s1.ax1x.com/2020/08/08/aIVjg0.png)
</div>

先读取模型后，进行预测，然后得到结果，代码如下所示：
```python
# 读取网络
network = keras.models.load_model('lenet_mnist.h5')
network.summary()

mnist = tf.keras.datasets.mnist
(trainImage, trainLabel),(testImage, testLabel) = mnist.load_data()
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(testImage[i], cmap='gray')
plt.show()

# 改变图片维度
testImage = tf.reshape(testImage,(10000,28,28,1))
# 预测前25张图片结果
result = network.predict(testImage)[0:25]
pred = tf.argmax(result, axis=1)
pred_list=[]
for item in pred:
    pred_list.append(item.numpy())
print(pred_list)
```

**结果如下所示**
```bash
[7, 2, 1, 0, 4, 
 1, 4, 9, 5, 9, 
 0, 6, 9, 0, 1, 
 5, 9, 7, 8, 4, 
 9, 6, 6, 5, 4]
```

**可以看到识别成功率可以达到98%，结果还是很出色的**

## 项目源码

项目源码点击[LeNet-5 MNIST手写数字识别](https://github.com/RoWe98/LeNet-5-MNIST)
