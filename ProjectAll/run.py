# coding:utf-8

from flask import Flask, render_template
from APP.SpeechExtraction.news_blueprint import  app_extraction

app = Flask("__main__")


# 分隔视图方法2：注册蓝图，url_prefix="/goods"定义前缀 这里变成 /goods/get_goods
app.register_blueprint(app_extraction,url_prefix="/SpeechExtraction")
#app.register_blueprint(app_extraction, url_prefix="/goods")

@app.route("/", methods=["GET"])
def index():
    """定义的视图函数"""
    return render_template("home.html")

if __name__ == '__main__':
    # 启动flask程序
    app.run()
    # app.run(host="0.0.0.0", port=5000, debug=True)