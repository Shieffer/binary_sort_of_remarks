from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # 这里的10000指的是每个输入样本数据的维度，不包含样本数量的维度
model.add(layers.Dense(16, activation='relu'))  # 从第二层开始，就不用写输入样本数据的维度，因为自动根据上一层进行调节
model.add(layers.Dense(1, activation='sigmoid'))

# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# 模型编译
# 配置优化器、损失函数和指标
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # 二元交叉熵
              metrics=['accuracy'])

