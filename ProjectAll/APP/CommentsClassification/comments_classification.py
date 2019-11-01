import pandas as pd
import os
import numpy as np
np.random.seed(42)
import gc
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D,LSTM
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, GRU, Bidirectional
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D,concatenate
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from gensim.models import KeyedVectors

import pickle
import jieba
import re
from hanziconv import HanziConv


class StringFliterZH():
    '''
    这是一个用于中文分词，过滤标点，停用词的类
    '''

    def __init__(self, stopwords=None):
        if stopwords:
            self.__stopwords = set(stopwords)
        else:
            self.__stopwords = pickle.load(open('stopwords.pickle', 'rb'))

    def to_simlified(self, string):
        '''繁体字转简体字'''
        string = HanziConv.toSimplified(string)
        return string

    def to_tokens(self, string, method=None, stopwords=None, flag=1):
        '''分词，去停用词等处理，返回分词结果，以空格分隔，类似英文文本格式
           @string：self入文本
           @method: 分词API，默认None，使用jieba分词
           @stopwords:停用词，默认None，使用默认停用词集
           @flag:是否启用停用词，默认启动，不启用时，保留所有文字字母文本
        '''
        if stopwords: self.__stopwords = set(stopwords)
        retu_string = ''
        if flag:
            for word in jieba.cut(string):
                if word not in self.__stopwords:
                    retu_string += word + ' '
        else:
            pattern = re.compile(r'\w+?')
            for word in jieba.cut(''.join(pattern.findall(string))):
                retu_string += word + ' '
        return retu_string.strip()

    def apply_all(self, string, method=None, stopwords=None):
        return self.to_tokens(self.to_simlified(string), method, stopwords)


def onehot(labels):
    labels = (np.arange(-2, 2) == labels.reshape(-1, 1)).astype(np.float32)
    return labels.flatten()


my_fliter = StringFliterZH()
def data_extract(cm):
    y_label = cm.iloc[:, 2:len(cm.columns)].values
    y_label_level1 = np.stack((np.any(y_label[:, 0:3] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 3:7] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 7:10] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 10:14] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 14:18] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 18:20] != -2, axis=1).astype(np.float32))).T
    y_label_level2 = np.zeros((y_label.shape[0], 80))
    for i in range(y_label_level2.shape[0]):
        y_label_level2[i] = onehot(y_label[i])
    y = np.hstack((y_label_level2, y_label_level1))

    corpus = cm['content'].apply(my_fliter.to_tokens).values

    return corpus, y

class CorpusSequence:
    def __init__(self):
        self.embed_size = 500 ##对应embedding向量的维度
        self.max_features = 35000 # 保留词个数
        self.maxlen = 400 #Rnn网络序列指定长度
        self.embedding_path = r"E:/MYGIT/model/wiki_stopwords/wiki_word2vec.kv"

    def to_sequence(self, *args):
        '''返回序列化文本，和embbeding词典'''
        tokenizer = text.Tokenizer(num_words=self.max_features)

        corpus = []
        for i in args:corpus += list(i)
        tokenizer.fit_on_texts(corpus)

        datas = []
        for i in args:
            X_data = tokenizer.texts_to_sequences(i)
            x_data = sequence.pad_sequences(X_data, maxlen=self.maxlen)
            datas.append(x_data)

        model_wv = KeyedVectors.load(self.embedding_path, mmap='r')
        word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words, self.embed_size))
        for word, i in word_index.items():
            if i >= self.max_features: continue
            try:
                embedding_vector = model_wv[word]
            except:
                embedding_vector = None
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        del model_wv
        _ = gc.collect()

        datas.append(embedding_matrix)
        return tuple(datas)

class Attention(Layer):
    '''自定义attention层，使用的是Transformer的attention机制，用于特征提取'''
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

class LossHistory(Callback):
    # 继承自Callback类,用于记录训练时候的loss等变化
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
    def on_batch_end(self, batch, logs={}):
        keys = list(logs.keys())
        self.losses['batch'].append(logs.get(keys[0]))
        self.accuracy['batch'].append(logs.get(keys[1]))
        self.val_loss['batch'].append(logs.get(keys[2]))
        self.val_acc['batch'].append(logs.get(keys[3]))
    def on_epoch_end(self, batch, logs={}):
        keys = list(logs.keys())
        self.losses['epoch'].append(logs.get(keys[0]))
        self.accuracy['epoch'].append(logs.get(keys[1]))
        self.val_loss['epoch'].append(logs.get(keys[2]))
        self.val_acc['epoch'].append(logs.get(keys[3]))

class Model_Level:
    def __init__(self):
        self.HIDDNE_SIZE_1 = 128
        self.HIDDNE_SIZE_2 = 64
        self.max_features = 35000
        self.maxlen = 400
        self.embed_size = 500
        self.epochs = 2
        self.batch_size = 256
        self.model = None

    def __get_model(self, embedding_matrix):
        ###embedding 和textcnn一样处理
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)

        # conv_0 = Conv1D(num_filters, kernel_size=kernel_size, strides=1)(x_emb)#这里也可以用conv1D，因为在embed_size
        # 等于词向量维度大小，故在列方向相当于没有做卷积操作，使用Conv2D的效果和Conv1D一样
        GRU1 = Bidirectional(GRU(self.HIDDNE_SIZE_1, return_sequences=True, recurrent_dropout=0.2,
                                 input_shape=(self.maxlen, self.embed_size)))(x)

        GRU2 = Bidirectional(GRU(self.HIDDNE_SIZE_1, return_sequences=False, recurrent_dropout=0.2,
                                 input_shape=(self.maxlen, self.HIDDNE_SIZE_1)))(GRU1)

        z = Dropout(0.2)(GRU2)

        fully1 = Dense(self.HIDDNE_SIZE_2, activation='relu')(z)

        outp = Dense(6, activation='sigmoid')(fully1)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def start_train(self, x,y, x1, y1, embedding_matrix):
        self.model = self.__get_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x, y[:, 80:], batch_size=self.batch_size,
                  epochs=self.epochs,validation_data=(x1, y1[:, 80:]),callbacks=[history])
        return history

class Model_Cnn_Attention:
    def __init__(self):
        self.filter_sizes = [1,2,3,5]
        self.num_filters = 32
        self.max_features = 35000
        self.maxlen = 400
        self.embed_size = 500
        self.epochs = 3
        self.batch_size = 256
        self.model = None

    def __get_model(self, embedding_matrix):
        ###embedding 和textcnn一样处理
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Reshape((self.maxlen, self.embed_size, 1))(x)  ###变成Conv2D输入格式

        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_3 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[3], self.embed_size), kernel_initializer='normal',
                        activation='elu')(x)

        z = Concatenate(axis=1)([conv_0, conv_1, conv_2, conv_3])
        myshape = (int(z.shape[1]), int(z.shape[-1]))
        z1 = Reshape((-1, myshape[1]))(z)

        outputs = []
        for i in range(20):
            z = Attention(myshape[0])(z1)
            z = Dropout(0.1)(z)
            z = Dense(64, activation="relu")(z)
            out = Dense(4, activation="softmax")(z)
            outputs.append(out)
        output = Concatenate()(outputs)
        model = Model(inputs=inp, outputs=output)

        def loss_a(y_true, y_pred):
            loss_sum = 0
            # print(y_pred.shape,y_true.shape)
            for i in range(0, 80, 4):
                loss_sum += categorical_crossentropy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return loss_sum

        def myacc(y_true, y_pred):
            rights = 0
            for i in range(0, 80, 4):
                rights += categorical_accuracy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return rights / 20

        model.compile(loss=loss_a,
                      optimizer='adam',
                      metrics=[myacc])
        return model

    def start_train(self, x,y, x1, y1, embedding_matrix):
        self.model = self.__get_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x, y[:, :80], batch_size=self.batch_size,
                  epochs=self.epochs,validation_data=(x1, y1[:, :80]),callbacks=[history])
        return history

class Model_GRU_Attention:
    '''GRU+Attention机制实现文本分类'''
    def __init__(self):
        self.filter_sizes = [1,2,3,5]
        self.num_filters = 32
        self.max_features = 35000
        self.maxlen = 400
        self.embed_size = 500
        self.epochs = 3
        self.batch_size = 128
        self.model = None

    def __get_model(self, embedding_matrix):
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x1 = Bidirectional(GRU(128, return_sequences=True))(x)

        outputs = []
        for i in range(20):
            x = Attention(self.maxlen)(x1)
            z = Dropout(0.1)(x)
            z = Dense(64, activation="relu")(z)
            out = Dense(4, activation="softmax")(z)
            outputs.append(out)
        output = Concatenate()(outputs)
        model = Model(inputs=inp, outputs=output)

        def loss_a(y_true, y_pred):
            loss_sum = 0
            # print(y_pred.shape,y_true.shape)
            for i in range(0, 80, 4):
                loss_sum += categorical_crossentropy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return loss_sum

        def myacc(y_true, y_pred):
            rights = 0
            for i in range(0, 80, 4):
                rights += categorical_accuracy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return rights / 20

        model.compile(loss=loss_a,
                      optimizer='adam',
                      metrics=[myacc])
        return model

    def start_train(self, x,y, x1, y1, embedding_matrix):
        self.model = self.__get_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x, y[:, :80], batch_size=self.batch_size,
                  epochs=self.epochs,validation_data=(x1, y1[:, :80]),callbacks=[history])
        return history



if __name__ == '__main__':
    paths = [
        'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
        'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv',
        'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv']

    ##文本处理，标签onehot编码
    corpus_train, y_train = data_extract(pd.read_csv(paths[0]))
    corpus_valid, y_valid = data_extract(pd.read_csv(paths[1]))
    corpus_test, _ = data_extract(pd.read_csv(paths[2]))##测试集没有标签

    ##序列化文本
    my_sequence = CorpusSequence()
    x_train, x_valid, x_test, embedding_matrix = my_sequence.to_sequence(corpus_train, corpus_valid,corpus_test)

    # pickle_file = 'E:\MYGIT\Project\ProjectAll\APP\CommentsClassification\comments_pickle'
    # with open(pickle_file, 'rb') as f:
    #     save = pickle.load(f)
    #     x_train = save['X_tra']
    #     y_train = save['y_tra']
    #     x_valid = save['X_val']
    #     y_valid = save['y_val']
    #     x_test = save['X_test']
    #     y_test = save['y_test']
    #     embedding_matrix = save['embedding_matrix']
    #     del save  # hint to help gc free up memory


    ##6大类检测
    # model_level1 = Model_Level()
    # model_level1.start_train(x_train, y_train, x_valid, y_valid, embedding_matrix)
    # pred_test_level1 = model_level1.model.predict(x_test)

    ##20小类情感颗粒度分类
    # model_level2 = Model_Cnn_Attention()
    model_level2 = Model_GRU_Attention()
    model_level2.start_train(x_train, y_train, x_valid, y_valid, embedding_matrix)
    #pred_test_level2 = model_level2.model.predict(x_test)


