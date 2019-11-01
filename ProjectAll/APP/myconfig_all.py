import pickle
class MyConfig:
    def __init__(self):
        self.paths = [
            'E:/MYGIT/model/wiki_stopwords/wiki_word2vec.kv',#word2vec词向量目录
            'E:/MYGIT/Word2Vec/wikiextractor-master/zhwiki/AA/wiki_frequency.txt',
            'E:/MYGIT/MyProjectOrigin/ProjectAll/temp_file/stopwords.txt',
            'E:/MYGIT/MyProjectOrigin/ProjectAll/APP/CommentsClassification/datas_pickle',
        ]

    def get_path(self, name):
        pathreturn = None
        for path in self.paths:
            if path.endswith(name):
                pathreturn = path
        return pathreturn

myconfig = MyConfig()

# print(Myconfig.get_path('vec.kv'))
# print(Myconfig.get_path('ltp_data'))
# print(Myconfig.get_path('temp_file'))