from collections import defaultdict
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import SentenceSplitter
from pyltp import Parser
from gensim.models import KeyedVectors
import numpy as np
import jieba
import os
import re
import pickle
from APP.SpeechExtraction.myconfig import Myconfig
import gc
jieba.initialize()


class SentenceEmbedding:
    #句子向量化类
    def __init__(self):
        self.word_frequence = self.__get_word_frequence()

    def get_sentences_vec(self, model_wv, sent_list):
        # 句子向量化处理
        a = 0.001
        row = model_wv.vector_size
        col = len(sent_list)
        sent_mat = np.zeros((row, col))
        for i, sent in enumerate(sent_list):
            length = len(sent)
            sent_vec = np.zeros(row)
            for word in sent:
                pw = self.word_frequence[word]
                w = a / (a + pw)
                try:
                    vec = np.array(model_wv[word])
                    sent_vec += w * vec
                except:
                    pass
            sent_mat[:, i] += sent_vec
            sent_mat[:, i] /= length

        # PCA处理
        sent_mat = np.mat(sent_mat)
        u, s, vh = np.linalg.svd(sent_mat)
        sent_mat = sent_mat - u * u.T * sent_mat
        return sent_mat

    def __get_word_frequence(self):
        # 这里不做停用次处理，直接在计算句子向量时候，如果找不到该词，直接跳过
        path = Myconfig.get_path('frequency.txt')
        assert path
        with open(path, 'rb') as f:
            word_frequence = pickle.load(f)
        return word_frequence

    # 计算余弦相似度
    def cos_similarity(self,v1, v2):
        assert isinstance(v1, np.ndarray)
        assert isinstance(v2,np.ndarray)
        # 输入向量维度不一致
        if len(v1) != len(v2):
            return 0
        return np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 返回句子向量矩阵中各列向量与第一列向量的相似度
    def __calcu_similarity(self, sent_mat):
        assert (isinstance(sent_mat, np.ndarray) or isinstance(sent_mat, np.matrix))
        # 采用点积的方法计算
        first = np.array(sent_mat[:, 0]).flatten()
        col = sent_mat.shape[1]
        sims = []
        for i in range(1, col):
            vec = np.array(sent_mat[:, i]).flatten()
            sims.append(self.cos_similarity(first, vec))
        return sims

    # 获取相似度结果#输入句子中每一句和首句的相似度
    def get_similarity_result(self, model_wv, sent_list):
        sent_mat = self.get_sentences_vec(model_wv,sent_list)
        sim = self.__calcu_similarity(sent_mat)
        return sim

class Model:
    #模型类
    def __init__(self):
        self.segmentor = None
        self.postagger = None
        self.recognizer = None
        self.parser = None
        self.model_wv = None

    def load_model(self):
        """返回分词，词性标注，命名实体识别，依存解析等实例对象"""
        LTP_DATA_DIR = Myconfig.get_path('ltp_data')
        LTP_TEMP_DIR = Myconfig.get_path('temp_file')
        assert LTP_DATA_DIR
        assert LTP_TEMP_DIR
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        cut_temp_path = os.path.join(LTP_TEMP_DIR, 'cut_external_dict/cut_external_dict')
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load_with_lexicon(cws_model_path, cut_temp_path)  # 加载模型


        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        pos_temp_path = os.path.join(LTP_TEMP_DIR, 'pos_external_dict/pos_external_dict')
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load_with_lexicon(pos_model_path, pos_temp_path)  # 加载模型

        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.recognizer = NamedEntityRecognizer()  # 初始化实例
        self.recognizer.load(ner_model_path)  # 加载模型

        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        self.parser = Parser()  # 初始化实例
        self.parser.load(par_model_path)  # 加载模型

        fname = Myconfig.get_path('vec.kv') #或取模型目录
        assert fname
        # model_wv.save(fname)
        self.model_wv = KeyedVectors.load(fname, mmap='r')

    def release_all_model(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()
        self.parser.release()
        del(self.model_wv)
        _ = gc.collect()
        _ = gc.collect()

class SpeechExtraction:
    def __init__(self):
        self.Sen_Embedding = SentenceEmbedding()
    def __cut_sentence(self,string):
        """@string contain many sentence"""
        sents = SentenceSplitter.split(string)  # 分句
        return list(sents)

    # 获取人名或机构名
    def __get_name(self,word_list, prase, ner_list, pos_list, say_index):
        # 合并说话对象名字
        index = -1
        for arc in prase:
            if arc.relation == "SBV" and arc.head == say_index + 1:
                index = prase.index(arc)
                # break
                # 第二种情况
        if index == -1:
            for arc in prase:
                if arc.relation == "SBV" and arc.head == prase[say_index].head and prase[say_index].relation == "COO":
                    index = prase.index(arc)
        # 两种情况都没找到
        if index == -1: return ''

        Entity = (
        'S-Nh', 'S-Ni', 'S-Ns', 'B-Nh', 'B-Ni', 'B-Ns', 'I-Nh', 'I-Ni', 'I-Ns', 'E-Nh', 'E-Ni', 'E-Ns')  # 命名实体标记
        # Entity = ('S-Nh', 'S-Ni')  # 命名实体标记
        name = ''
        cur = index
        ret_flag = False
        while cur >= 0:
            if ner_list[cur] in Entity or cur == index:
                name = word_list[cur] + name
                if ner_list[cur] in Entity: ret_flag = True
                cur -= 1
            else:
                break

        if ret_flag: return name
        if pos_list[index] not in ('n', 'nh', 'ws'): return ''

        pre = word_list[:index]  # 前半部分
        pos = word_list[index + 1:]  # 后半部分
        while pre:
            w = pre.pop(-1)
            w_index = word_list.index(w)
            # if prase[w_index].relation == 'ADV': continue
            if prase[w_index].relation in ('WP', 'ATT', 'SVB') and (w not in ('，', '。', '、', '）', '（')):
                name = w + name
            else:
                pre = False
        while pos:
            w = pos.pop(0)
            w_index = word_list.index(w)
            if prase[w_index].head - 1 == index or \
                    prase[prase[w_index].head - 1].head - 1 == index or \
                    prase[prase[prase[w_index].head - 1].head - 1].head - 1 == index:
                if prase[w_index].relation in ('WP', 'LAD', 'COO', 'RAD', 'ATT') and w_index < say_index and (
                        w not in ('，', '。', '、', '）', '（')):
                    name = name + w  # 向后拼接
                else:  # 中断拼接直接返回
                    pos = False
            else:
                pos = False
        return name

        if pos_list[index] not in ('n', 'nh', 'ws'): return name
        name = word_list[index]

        pre = word_list[:index]  # 前半部分
        pos = word_list[index + 1:]  # 后半部分
        while pre:
            w = pre.pop(-1)
            w_index = word_list.index(w)
            # if prase[w_index].relation == 'ADV': continue
            if prase[w_index].relation in ('WP', 'ATT', 'SVB') and (w not in ('，', '。', '、', '）', '（')):
                name = w + name
            else:
                pre = False
        while pos:
            w = pos.pop(0)
            w_index = word_list.index(w)
            if prase[w_index].head - 1 == index or \
                    prase[prase[w_index].head - 1].head - 1 == index or \
                    prase[prase[prase[w_index].head - 1].head - 1].head - 1 == index:
                if prase[w_index].relation in ('WP', 'LAD', 'COO', 'RAD', 'ATT') and w_index < say_index and (
                        w not in ('，', '。', '、', '）', '（')):
                    name = name + w  # 向后拼接
                else:  # 中断拼接直接返回
                    pos = False
            else:
                pos = False
        return name

    def __get_under_node(self,parent_node, parse, relation):
        index = []
        for arc in parse:
            if arc.relation == relation and arc.head == parent_node + 1:
                index.append(parse.index(arc))
        try:
            return index
        except NameError:
            return -1

        # 找出输入所有节点中最小的索引和最大的索引，然后返回两者间的句子

    def __node_under_sentence(self, parent_point, den_parsing_list, word_list):
        words_index = []
        # 搜索子节点
        if parent_point == [] or parent_point[0] < 0: return ''
        for point in parent_point:
            start = [point]
            words_index.append(start[0])
            while start != []:
                cur = start.pop(0)
                for i, arc in enumerate(den_parsing_list):
                    if arc.head == cur + 1:
                        words_index.append(i)
                        start.append(i)
        words_index = sorted(words_index)
        # print(words_index, len(word_list))
        return ''.join(word_list[words_index[0]:words_index[-1] + 1])

    def find_opinion_of_someone(self,input_news, say_related, models):
        # 输入文本进行分句
        sentence_list = self.__cut_sentence(input_news)
        sentence_list = [sen for sen in sentence_list if len(sen) > 3]
        Entity = ['S-Nh', 'S-Ni', 'B-Nh', 'B-Ni', 'I-Nh', 'I-Ni', 'E-Nh', 'E-Ni']  # 命名实体标记
        pro_news_dict = defaultdict(list)  # 用来存储可能是多句的句子
        news_dict = defaultdict(list)


        word_list_all = [list(models.segmentor.segment(sentence)) for sentence in sentence_list]  # 分词
        # name_list = [] #用于保存所有人名
        for _i, sentence in enumerate(sentence_list):
            # print('\n——————————句子:{} 处理过程——————————'.format(_i + 1))
            # time_point_1 = time.time()
            word_list = word_list_all[_i]  #
            pos_list = list(models.postagger.postag(word_list))  # 词性分析
            ner_list = list(models.recognizer.recognize(word_list, pos_list))  # 命名实体提取
            den_parsing_list = list(models.parser.parse(word_list, pos_list))  # 依存关系

            # 获取命名实体和说相关词,同时获取索引值
            pro_say_word = [(a, i) for i, a in enumerate(word_list) if a in say_related]
            if pro_say_word == []: continue

            # 找到说的主语
            for say in pro_say_word:
                if say[0] in (':', '：') and den_parsing_list[say[1]].relation == 'WP':
                    say = (say[0], den_parsing_list[say[1]].head - 1)
                name = self.__get_name(word_list, den_parsing_list, ner_list, pos_list, say[1])
                #print('第{}句子, sayword:{} name:{}'.format(_i, say, name))
                if name != '':
                    say_underword = []
                    index = self.__get_under_node(say[1], den_parsing_list, 'VOB')
                    _flag = 999
                    if index != -1:
                        _flag = len(index)
                        say_underword += index
                        index = self.__get_under_node(say[1], den_parsing_list, 'COO')
                        if index != -1: say_underword += index

                    # 特殊情况处理1
                    if len(say_underword) > _flag:
                        for _node in range(_flag):
                            if pos_list[say_underword[_node]] in ['n', 'nh']:
                                say_underword.pop(0)

                    # 特殊情况处理2
                    saying = self.__node_under_sentence(say_underword, den_parsing_list, word_list)
                    if not saying:
                        if (den_parsing_list[say[1]].relation == 'POB' and den_parsing_list[
                            den_parsing_list[say[1]].head - 1].relation == 'ADV'):
                            saying = ''.join(word_list[say[1] + 1:])
                            saying = saying.strip('，')
                        if not saying:
                            if _i > 0:
                                quotations = re.findall(r'“(.+?)”', sentence_list[_i - 1])
                                if quotations and len(quotations[-1]) > 6: saying = quotations[-1]
                    if saying != '':
                        if saying[-5:] in sentence[-9:]:
                            words1 = jieba.lcut(saying)
                            _word_list = sentence_list[_i + 1:_i + 4]  # 切片的话_i+4如果超出则取到末尾
                            _word_list.insert(0, words1)
                            sim = self.Sen_Embedding.get_similarity_result(models.model_wv,_word_list)
                            #print(sim)
                            for i_sim, _sim in enumerate(sim):
                                if _sim > 0.85:
                                    pro_news_dict[_i] += [sentence_list[_i + i_sim + 1]]
                                else:
                                    break
                        news_dict[_i] = [name, say[0], saying]
                        break

                        # 对多句添加进行判断，如果下一句里没说，或没提取出来，则拼接到上一句，避免重复
        for key in pro_news_dict.keys():
            for _i in range(len(pro_news_dict[key])):
                if key + _i + 1 not in news_dict.keys():
                    news_dict[key][1] += pro_news_dict[key][_i]
                else:
                    break
        return news_dict


class My_Extractor:
    # 外部接口类，把本文件功能全部集成在该类
    def __init__(self):
        self.related_words = self.__get_relatedwords()
        self.models = Model()
        self.models.load_model()
        self.extractor = SpeechExtraction()
    def __get_relatedwords(self):
        temp_path = Myconfig.get_path('temp_file')
        assert temp_path
        path = os.path.join(temp_path, 'saywords')
        relatedwords_list = []
        with open(path, encoding='utf-8') as f:
            line_str = f.readline()
            while line_str != '':
                relatedwords_list.append(line_str.strip('\n'))
                line_str = f.readline()
        relatedwords_list = tuple(relatedwords_list)
        return relatedwords_list

    def get_results(self, text):
        result = self.extractor.find_opinion_of_someone(text,self.related_words,self.models)
        return result

    def release(self):
        self.models.release_all_model()
        _ = gc.collect()
