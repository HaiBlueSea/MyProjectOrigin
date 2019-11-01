import pandas as pd
import os
import numpy as np 
import jieba
import pickle
import re
from hanziconv import HanziConv
import time
from bert_serving.client import BertClient

import pandas as pd
import os
import numpy as np
np.random.seed(42)
import gc
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
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



class StringFliterZH():
    '''
    这是一个用于中文分词，过滤标点，停用词的类
    '''
    def __init__(self, stopwords=None):
        if stopwords:self.__stopwords = set(stopwords)
        else:self.__stopwords = pickle.load(open('stopwords.pickle', 'rb'))
        
    def to_simlified(self, string):
        '''繁体字转简体字'''
        string = HanziConv.toSimplified(string)
        return string
    
    def to_tokens(self, string, method =None,stopwords=None,flag=0):
        '''分词，去停用词等处理，返回分词结果，以空格分隔，类似英文文本格式
           @string：self入文本
           @method: 分词API，默认None，使用jieba分词
           @stopwords:停用词，默认None，使用默认停用词集
           @flag:是否启用停用词，默认启动，不启用时，保留所有文字字母文本
        '''
        if stopwords:self.__stopwords = set(stopwords)
        retu_string = ''
        if flag:
            for word in jieba.cut(string):
                if word not in self.__stopwords:
                    retu_string += word + ' '
        else:
            pattern = re.compile(r'[\u4e00-\u9fa5]+')
            for word in jieba.cut(' '.join(pattern.findall(string))):
                if word not in self.__stopwords:
                    retu_string += word + ' '
        return retu_string.strip()
    
    def apply_all(self,string, method=None, stopwords=None):
        return self.to_tokens(self.to_simlified(string), method, stopwords)


def onehot(labels):
    labels = (np.arange(-2,2) == labels.reshape(-1,1)).astype(np.float32)
    return labels.flatten()

my_fliter = StringFliterZH()
def data_extract(cm):
    y_label = cm.iloc[:, 2:len(cm.columns)].values
    y_label_level1 =  np.stack((np.any(y_label[:,0:3]!=-2, axis=1).astype(np.float32), 
                  np.any(y_label[:,3:7]!=-2, axis=1).astype(np.float32),
                  np.any(y_label[:,7:10]!=-2, axis=1).astype(np.float32),
                  np.any(y_label[:,10:14]!=-2, axis=1).astype(np.float32),
                  np.any(y_label[:,14:18]!=-2, axis=1).astype(np.float32),
                  np.any(y_label[:,18:20]!=-2, axis=1).astype(np.float32))).T
    y_label_level2 = np.zeros((y_label.shape[0], 80))
    for i in range(y_label_level2.shape[0]):
        y_label_level2[i] = onehot(y_label[i])
    y = np.hstack((y_label_level2,y_label_level1))
    
    corpus = cm['content'].apply(my_fliter.to_tokens).values
    
    return corpus, y

def get_coefs(word, *arr):
	return word, np.asarray(arr, dtype='float32')


def main():
	p1 = time.time()
	paths = [
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv',
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv']

	corpus_train, y_train = data_extract(pd.read_csv(paths[0]))
	corpus_valid, y_valid = data_extract(pd.read_csv(paths[1]))
	corpus_test, y_test = data_extract(pd.read_csv(paths[2]))
	print('deal_corpus : {:.3f} s'.format(time.time() - p1))	

	max_features = 35000
	maxlen = 400
	embed_size = 300
	##把corpus序列化，保存前100000个词作为字典,会分词过滤标点等，只适用于英文
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(corpus_train) + list(corpus_test) + list(corpus_valid))

	X_train = tokenizer.texts_to_sequences(corpus_train)
	#padding使得所有序列一样长,不够的往前填充0，多的保留后200个
	x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

	X_valid = tokenizer.texts_to_sequences(corpus_valid)
	x_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

	X_test = tokenizer.texts_to_sequences(corpus_test)
	x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	paths = ['E:/MYGIT/model/word2vec_models/sgns.weibo.bigram',
			 'E:/MYGIT/model/word2vec_models/sgns.renmin.bigram',
			 'E:/MYGIT/model/word2vec_models/sgns.target.word-word.baidubaike.dim300',
			 'E:/MYGIT/model/word2vec_models/sgns.target.word-ngram.1-2.baidubaike.dim300']
	my_matrix = []

	for i in range(4):
		EMBEDDING_FILE = paths[i]

		model_wv = {}
		with open(EMBEDDING_FILE, encoding='utf-8') as f:
			line = f.readline()
			while line != '':
				line = f.readline()
				key, value = get_coefs(*line.rstrip().rsplit(' '))
				model_wv[key] = value
		empty = []
		word_index = tokenizer.word_index
		nb_words = min(max_features, len(word_index))
		embedding_matrix = np.zeros((nb_words, embed_size))

		for word, i in word_index.items():
			if i >= max_features: continue
			try:
				embedding_vector = model_wv[word]
			except:
				empty.append(word)
				embedding_vector = None
			if embedding_vector is not None: embedding_matrix[i] = embedding_vector
		print(empty)
		del (model_wv)
		_ = gc.collect()

		my_matrix.append(embedding_matrix)

	pickle_file = 'comments_pickle2'
	try:
		with open(pickle_file, 'wb') as f:
			save = {
				'X_tra': x_train,
				'y_tra': y_train,
				'X_val': x_valid,
				'y_val': y_valid,
				'X_test': x_test,
				'y_test': y_test,
				'matrixs': my_matrix,
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)



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

class Model_GRU_Attention:
    '''GRU+Attention机制实现文本分类'''
    def __init__(self):
        self.filter_sizes = [1,2,3,5]
        self.num_filters = 32
        self.max_features = 30000
        self.maxlen = 300
        self.embed_size = 300
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


def main1():
	p1 = time.time()
	paths = [
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv',
		'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv']

	# corpus_train, y_train = data_extract(pd.read_csv(paths[0]))
	corpus_valid, y_valid = data_extract(pd.read_csv(paths[1]))
	# corpus_test, y_test = data_extract(pd.read_csv(paths[2]))
	print('deal_corpus : {:.3f} s'.format(time.time() - p1))

	max_features = 30000
	maxlen = 300
	embed_size = 300
	##把corpus序列化，保存前100000个词作为字典,会分词过滤标点等，只适用于英文
	tokenizer = text.Tokenizer(num_words=max_features)
	# tokenizer.fit_on_texts(list(corpus_train) + list(corpus_test) + list(corpus_valid))
	tokenizer.fit_on_texts(list(corpus_valid))

	# X_train = tokenizer.texts_to_sequences(corpus_train)
	# padding使得所有序列一样长,不够的往前填充0，多的保留后200个
	# x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

	X_valid = tokenizer.texts_to_sequences(corpus_valid)
	x_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

	# X_test = tokenizer.texts_to_sequences(corpus_test)
	# x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	paths = ['E:/MYGIT/model/word2vec_models/sgns.weibo.bigram',
			 'E:/MYGIT/model/word2vec_models/sgns.renmin.bigram',
			 'E:/MYGIT/model/word2vec_models/sgns.target.word-word.baidubaike.dim300',
			 'E:/MYGIT/model/word2vec_models/sgns.target.word-ngram.1-2.baidubaike.dim300']

	EMBEDDING_FILE = paths[-1]
	model_wv = {}
	with open(EMBEDDING_FILE, encoding='utf-8') as f:
		line = f.readline()
		while line != '':
			line = f.readline()
			key, value = get_coefs(*line.rstrip().rsplit(' '))
			model_wv[key] = value
	empty = []
	word_index = tokenizer.word_index
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.zeros((nb_words, embed_size))

	for word, i in word_index.items():
		if i >= max_features: continue
		try:
			embedding_vector = model_wv[word]
		except:
			empty.append(word)
			embedding_vector = None
		if embedding_vector is not None: embedding_matrix[i] = embedding_vector
	print(empty)
	del (model_wv)
	_ = gc.collect()

	x1,x2,y1,y2 = train_test_split(x_valid,y_valid,test_size=0.2)
	model= Model_GRU_Attention()
	hist = model.start_train(x1,y1,x2,y2,embedding_matrix)


def get_bert_vec():
    pass

if __name__ == '__main__':
    main1()
    # paths = [
    #     'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
    #     'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv',
    #     'E:/MYGIT/DataSources/comments_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv']
    # bc = BertClient(ip='39.100.3.165')
    # cm = pd.read_csv(paths[1])
    # corpus = cm['content'].values
    # corpus = corpus.tolist()
    # # for i in range(58):
    # #     p1 = time.time()
    # #     x_valid = bc.encode(corpus[i*256:i*256+256])
    # #     np.save('npdata/'+str(i),x_valid)
    # #     print('#############{}#########{:.2f}'.format(i,(time.time()-p1)/60))
    #
    # x_valid = bc.encode(corpus[0:2])
    # print(x_valid)
    # bc.close()
