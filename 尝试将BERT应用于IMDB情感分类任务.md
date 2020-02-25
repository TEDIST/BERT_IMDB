# 前言
之前在学习《Python深度学习》这本书的时候记得在书中有一个 使用GloVe词嵌入的方式进行IMDB数据集的情感分类任务，而最近在网上学习了一个使用BERT进行关系抽取的项目，于是乎考虑尝试用BERT来再次尝试对IMDB数据集的情感分类任务。同样的这次也是采取随机的200个数据集作为训练集，10000个数据集作为测试集。全部代码在[github](https://github.com/TEDIST/BERT_IMDB)上已经给出。

# 步骤
本次实验的大致步骤同书中的例子一致：
分词->向量化+Dense层

## 数据集

从http://mng.bz/0tIo,下载到的原始IMDB数据集，我们使用解压后文件中的aclimdb文件夹中的train数据集，其中包含12500个neg样本和12500个pos样本。对其进行如下处理：

```c
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
```
## 使用BERT使其向量化
由于之前我们处理数据的时候得到的数据集前12500个是neg样本后12500个是pos样本，因此我们需要将其随机打乱：

```c
idxs = np.random.randint(0,len(texts),size = 10200)

X = []
y = []
for id in idxs:
    X.append(texts[id])
    y.append(labels[id])
```
得到打乱后的数据集后我们使用BERT对其进行向量化，BERT部分的具体代码在github上有给出，这里我们直接使用就行了，这里由于样本是英文的因此我们需要提前下载uncased_L-12_H-768_A-12模型，最后和书中相同我们取句子的最大长度为100.
对于文本分类任务，BERT模型在文本前插入一个[CLS]符号，并将该符号对应的输出向量作为整篇文本的语义表示，用于文本分类，可以理解为：与文本中已有的其它字/词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。因此我们做如下处理：

```c
X_VEC = []
## 使用BERT进行向量化
print("star encoding...")
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=100)
for text in X:
    X_VEC.append(bert_model.encode([text])["encodes"][0])

X_VEC_CLS = []
for vec in X_VEC:
    X_VEC_CLS.append(vec[0])

x_train = np.array(X_VEC_CLS[:200])
x_test = np.array(X_VEC_CLS[200:])
y_train = np.array(y[:200])
y_test  = np.array(y[200:])
```

## 模型
由上面得到的训练集的向量化表示再简单接上一个Dense层就完成了模型的最终构建：

```c
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
```

## Performance
最终我们仅仅使用200个样本的训练集在10000个样本的验证集上得到了将近0.7的acc，远高于书中GloVe的效果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200225113932585.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzcyOTcy,size_16,color_FFFFFF,t_70)
