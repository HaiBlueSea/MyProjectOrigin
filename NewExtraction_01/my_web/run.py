# coding:utf-8

from flask import Flask, request, render_template,jsonify

from news_extraction import  *

app = Flask("__main__")

# # 3. 直接操作config的字典对象
#app.config["DEBUG"] = True

@app.route("/",methods=["GET"])
def index():
    """定义的视图函数"""
    return render_template("index2.html")


@app.route("/solve",methods=["POST"])
def solve():
    text = request.data
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = text.replace('\u3000', '')
    text = text.replace('\\n', '')
    #print(text)
    #print(text == '')
    #print(text)
    data = find_opinion_of_someone(text, relatedwords_list, model_list)
    # data = { '小明':['小明','也许不是这样的'],
    #          '大米':['大米','不知道呢？？']
    # }
    return jsonify(data)

if __name__ == '__main__':
    # 启动flask程序
    model_list = load_all_model()
    relatedwords_list = get_relatedwords()
    app.run()
    #pass
    # app.run(host="0.0.0.0", port=5000, debug=True)