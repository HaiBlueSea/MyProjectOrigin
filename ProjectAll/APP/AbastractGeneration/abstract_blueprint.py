# coding:utf-8

from flask import request, render_template, jsonify,Blueprint
from threading import Timer
from APP.AbastractGeneration.abstract_extraction import My_Summrazation
import gc
import pymysql
import random
import re
app_summarization = Blueprint("autosummarization", __name__, static_folder='static',template_folder='templates')


Summary = None
engine =None

def connet_sql():
    conn = pymysql.connect(
        host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
        user='root',
        password='AI@2019@ai',
        db='stu_db',
        charset='utf8'
    )
    return conn

def load_extractor():
    global Summary
    global engine
    if not Summary:
        Summary = My_Summrazation() # 删除该对象时限调用对象的release方法
        print('模型载入')
    if not engine:
        engine = connet_sql()

def release_model():
    global Summary
    global engine
    if Summary:
        Summary.release()  
        Summary  = None
        _=gc.collect()
        _=gc.collect()
        print('模型被释放')
    if engine:
        engine.colse()
        engine = None
        _=gc.collect()
        _=gc.collect()

class DelayRelease:
    #用来延迟释放模型
    def timer_start(self):
        #使用后5分钟后释放模型
        self.t = Timer(300, release_model)
        self.t.start()
    def timer_stop(self):
        self.t.cancel()

Mytimer = DelayRelease()
@app_summarization.route("/", methods=["GET"])
def index():
    """定义的视图函数"""
    t = Timer(1, load_extractor)
    t.start()
    return render_template("pro2.html")

@app_summarization.route("/solve", methods=["POST"])
def solve():
    data = request.json
    text = data['text']
    num = int(data['num'])
    title = data['title']
    if len(title.strip()) < 8:
        title = None
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = text.replace('\u3000', '').replace('\n','').replace('\\n','')
    if Summary:
        # try:
        data = Summary.get_results(text, num,title = title)
        # except:
            #return jsonify({'code':0})
        try:Mytimer.timer_stop()
        except:pass
        Mytimer.timer_start()
        return jsonify(data)
    else:
        return jsonify({'code': 0})

@app_summarization.route("/mysql", methods=["GET"])
def get_data_mysql():
    if engine:
        rnd = random.randint(33000,36000)
        sql = "select content from news_chinese_01 where id="+str(rnd)
        cur = engine.cursor()
        cur.execute(sql)
        ss = cur.fetchall()[0][0].replace('\\n','')
        for flag in ('乐讯','报讯','快讯','技讯','日电','日讯','(组图)', '(图)','（组图）','（图）'):
            title =  ss.find(flag)
            if title != -1:
                title = (title, title+len(flag))
                break
        if title ==-1:
            title = re.search(r'[\(（]\w*?记者.*?[\)）]', ss)
            if title:
                title =title.span()
            else:
                title = -1
        if title != -1:
            content = ss[title[1]:]
            head = ss[:title[0]]
        else:
            content =ss
            head =''
        data = {'content':content, 'title':head}
        return jsonify(data)
