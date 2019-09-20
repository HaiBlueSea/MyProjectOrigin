# coding:utf-8

from flask import request, render_template, jsonify,Blueprint
from threading import Timer

from APP.SpeechExtraction.news_extraction import My_Extractor

app_extraction = Blueprint("news_extraction", __name__, static_folder='static',template_folder='templates')


Extractor = None
def load_extractor():
    global Extractor
    Extractor = My_Extractor() # 删除该对象时限调用对象的release方法

def release_model():
    global Extractor
    if not Extractor:
        Extractor.release()
        Extractor  =None
        print('模型被释放')

class DelayRelease:
    #用来延迟释放模型
    def timer_start(self):
        #使用后5分钟后释放模型
        self.t = Timer(300, release_model)
        self.t.start()
    def timer_stop(self):
        self.t.cancel()

Mytimer = DelayRelease()

@app_extraction.route("/", methods=["GET"])
def index():
    """定义的视图函数"""
    t = Timer(2, load_extractor)
    t.start()
    return render_template("pro1.html")

@app_extraction.route("/solve", methods=["POST"])
def solve():
    text = request.data
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = text.replace('\u3000', '')
    text = text.replace('\\n', '')
    text = text.replace(' ', '')
    # print(text)
    try:
        data = Extractor.get_results(text)
    except:
        return 0

    try:Mytimer.timer_stop()
    except:pass
    Mytimer.timer_start()
    return jsonify(data)

    # data = { '小明':['小明','也许不是这样的'],
    #          '大米':['大米','不知道呢？？']
    # }