# coding:utf-8

from flask import request, render_template,Blueprint,jsonify
from threading import Timer
import numpy as np
import random
import gc
import re
import  pickle
from APP.myconfig_all import myconfig
app_classification = Blueprint("classification_extraction", __name__, static_folder='static',template_folder='templates')


data = None
def release_data():
    global data
    data = None
    _ = gc.collect()

class DelayRelease:
    #用来延迟释放模型
    def timer_start(self):
        #使用后5分钟后释放模型
        self.t = Timer(300, release_data)
        self.t.start()
    def timer_stop(self):
        self.t.cancel()

Mytimer = DelayRelease()

@app_classification.route("/", methods=["GET"])
def index():
    """定义的视图函数"""
    global data
    path = myconfig.get_path('pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)  #dict_keys(['barcharts', 'scores', 'metrics', 'texts', 'pred'])
    return render_template("pro3.html")

@app_classification.route("/getcomments", methods=["GET"])
def get_comments():
    id = random.randint(0,14990)
    if data:
        result = {'text':data['texts'][id],'pred':data['pred'][id]}
    else:
        result = {'code':'0'}
    return jsonify(result)

@app_classification.route("/getevaluate", methods=["GET"])
def get_evaluate():
    try:
        Mytimer.timer_stop()
        Mytimer.timer_start()
    except:pass
    id = request.args.get("id")
    try:id = int(id)-1
    except:id = 0
    if data:
        result = {'barchart':data['barcharts'][id],
                  'score':[data['scores'][id]],
                  'metric':data['metrics'][id]}
    else:
        result = {'code':'0'}
    return jsonify(result)


