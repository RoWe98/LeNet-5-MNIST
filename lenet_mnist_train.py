import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers

# 加载数据集
mnist = tf.keras.datasets.mnist
(trainImage, trainLabel),(testImage, testLabel) = mnist.load_data()
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

trainImage = tf.reshape(trainImage,(60000,28,28,1))
testImage = tf.reshape(testImage,(10000,28,28,1))
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

# 网络定义
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

# 模型训练 训练30个epoch
network.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
network.fit(trainImage,trainLabel,epochs=30,validation_split=0.1)

# 模型保存
network.save('./lenet_mnist.h5')
print('lenet_mnist model saved')
del network
