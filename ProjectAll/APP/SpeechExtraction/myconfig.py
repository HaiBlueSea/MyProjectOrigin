import pickle
class MyConfig:
    def __init__(self):
        self.paths = [
            'E:/MYGIT/Project/ltp_data',#pyltp模型目录
            './temp_file',#
            'E:/MYGIT/model/wiki_stopwords/wiki_word2vec.kv',#word2vec词向量目录
            'E:/MYGIT/Word2Vec/wikiextractor-master/zhwiki/AA/wiki_frequency.txt'
        ]

    def get_path(self, name):
        pathreturn = None
        for path in self.paths:
            if path.endswith(name):
                pathreturn = path
        return pathreturn

Myconfig = MyConfig()

# print(Myconfig.get_path('vec.kv'))
# print(Myconfig.get_path('ltp_data'))
# print(Myconfig.get_path('temp_file'))