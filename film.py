from keras.datasets import imdb
import numpy as np
import film_model
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 加载数据
# 所加载的数据集其实是二维列表
# 每条评论是由其单词的整数索引构成的向量
(train_data, train_lables), (test_data, test_lables) = imdb.load_data(num_words=10000)  # 10000意思是只保留最常出现的前10000单词

word_index = imdb.get_word_index()  # word_indexs是一个将单词映射为整数索引的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # dict.items(),返回的是字典键值对的列表，目的是为了可以遍历

# 准备数据（合适的格式）
# 因为神经网络需要传入张量的数据结构，所以要将数据向量化，将所有评论都向量化到一样的维度此处化到10000维，因为根据最常用的前10000的单词，单词所对应的序号最大为9999
# 对列表进行one-hot编码，将其转化为0和1组成的向量


def vectorize_sequences(sequences, dimension=10000):  # 这里的 sequence 其实是传入的（训练集/测试集)
    results = np.zeros((len(sequences), dimension))   # 创建零矩阵（二维张量）
    for i, sequence in enumerate(sequences):
        # 这里其实有点桶排的味道，将下标索引为该值的元素置1
        results[i, sequence] = 1.                       # [1.]表示此处数据类型为浮点
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 将标签也向量化，因为标签本来就是一维的向量了，所以直接转化成张量的数据结构就可了
y_train = np.asarray(train_lables).astype('float32')
y_test = np.asarray(test_lables).astype('float32')

# 将训练集的前10000个取出来做验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 开始训练
# history = film_model.model.fit(partial_x_train,  # 训练集
#                                partial_y_train,  # 训练集标签
#                                epochs=20,        # 轮次
#                                batch_size=512,   # 批量
#                                validation_data=(x_val, y_val)  # 验证集
#                                )

history = film_model.model.fit(x_train,  # 训练集
                               y_train,  # 训练集标签
                               epochs=4,        # 轮次
                               batch_size=512,   # 批量
                               )

# 绘制训练损失和验证损失
'''
调用model.fit()返回了一个History对象。这个对象有一个成员history
它是一个字典，包含训练过程中的所有数据[val_acc,acc,val_loss,loss]
'''
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_value = history_dict['val_loss']
#
# epochs = range(1, len(loss_values)+1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')  # 'bo'表示蓝色圆点
# plt.plot(epochs, val_loss_value, 'b', label='Validation loss')  # 'b'表示蓝色实线
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 绘制训练精度和验证精度
# plt.clf()  # 清空图像
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Training Acc')  # 'bo'表示蓝色圆点
# plt.plot(epochs, val_acc, 'b', label='Validation Acc')  # 'b'表示蓝色实线
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# 在测试集上评估
result = film_model.model.evaluate(x_test, y_test)
print(result)

# 在测试集上预测
print(film_model.model.predict(x_test))
