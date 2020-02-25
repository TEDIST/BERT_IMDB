# 从pos以及neg样例中共抽取25000个样本
import os
imdb_dir = '/Users/ted/Desktop/NLP/IMDB-BERT/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# print(len(texts))

# 通过BERT将文本数据转换成向量
import numpy as np
from bert.extract_feature import BertVector

## 将原始数据集打乱，并分成训练集和验证集（本次实验从原始样本中，选取200个作为训练集，10000个作为验证集）

idxs = np.random.randint(0,len(texts),size = 10200)

X = []
y = []
for id in idxs:
    X.append(texts[id])
    y.append(labels[id])

X_VEC = []
## 使用BERT进行向量化
print("star encoding...")
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=100)
for text in X:
    X_VEC.append(bert_model.encode([text])["encodes"][0])

X_VEC_CLS = []
for vec in X_VEC:
    X_VEC_CLS.append(vec[0])

x_train = np.array(X_VEC_CLS[:8000])
x_test = np.array(X_VEC_CLS[8000:])
y_train = np.array(y[:8000])
y_test  = np.array(y[8000:])

# 训练集测试集构建完成，开始准备构建模型

print("star training...")
from keras.models import Sequential
from keras.layers import  Dense

model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
 loss='binary_crossentropy',
 metrics=['acc'])
history = model.fit(x_train, y_train,
 epochs=10,
 batch_size=8,
 validation_data=(x_test, y_test))

# 绘制结果
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()