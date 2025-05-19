import os
import json
import time
import torch
import folium
import threading
import random
from utils.utils import *
from folium import plugins
from utils.class_config import CFG
from utils.trainClass import buildInferModel
import albumentations as A
from multiprocessing import Process
from flask import Flask, request, jsonify, render_template, send_file, flash, Response, stream_with_context
from flask_caching import Cache
from YOLOTensorRT.inferdet import main, draw_image
from YOLOTensorRT.models import TRTModule


app = Flask(__name__, template_folder='templates')
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream') 
cache = Cache(app)
aliyunoss = AliyunOss()
device = "cuda:0"
device = torch.device(device)
engine = "/app/utils/weight/best.engine"
Engine = TRTModule(engine, device)
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
model_path_list = ["vit_base_patch8_224.augreg_in21k"]
path_list = ["/app/utils/weight/best.pth"]
model_list = []
epoch = 0
for i in range(len(path_list)):
    model_list.append(build_model(CFG=CFG, modelName=model_path_list[i], pretrained=path_list[i]))
    model_list[-1].to(CFG.device)
    model_list[-1].eval()
    
transform = {"valid_test": A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], p=1.0)
])}
    

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST': 
        info = request.json
        if info:
            threads = []
            for url in info.get('urls'):
                thread = threading.Thread(target=download_images, args=(list(url.values())[0], 'input'))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
            if info.get('urls'):
                detection = main(Engine = Engine, imgs = "/app/input/", device = device, model_list=model_list, transform=transform, aliyunoss=aliyunoss, func=None)
                return jsonify(detection)
    return jsonify({"错误": "我需要post请求"})


@app.route('/getImage', methods=['GET', 'POST'])
def getImag():
    if request.method == 'POST':
        info = request.json.get("imageUrl")
        if info:
            image = download_images(info, "train/new/", 0)
            basename = os.path.basename(info)
            cv2.imwrite("train/new/" + basename, image)
            objects = request.json.get("objects")
            for i in objects:
                box = [i["xmin"], i['ymin'], i['xmax'], i['ymax']]
                image = draw_image(image=image, box=box, cls=i['sort'])
            cv2.imwrite("output/" + basename, image)
            generate_annotation("", basename, basename, objects)
            def sycRetrain(basename):
                aliyunoss.put_object_from_file("FuChuang/" + basename, "output/" + basename)
                os.remove("output/" + basename)
                import json
                with open('/app/train/config.json', 'r') as f:
                    json_data = f.read()
                data = json.loads(json_data)
                num = data["num"]
                standard = data['standard']
                num += 1
                if num >= standard:
                    from utils.trainClass import retrain
                    from utils.trainYolo import yoloRetrain
                    # yoloRetrain()
                    retrain()
                    num = 0
                with open('/app/train/config.json', 'w') as f:
                    data["num"] = num
                    json.dump(data, f)
            thread = threading.Thread(target=sycRetrain, args=(basename, ))
            thread.start()
            detection = {"url": aliyunoss.getUrl("FuChuang/" + basename)}
            return jsonify(detection)
    return jsonify({"错误": "我需要post请求"})


@app.route('/map', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        location = request.args.get("location")
        if location:
            address, la_lo = location2lalo(location)
            la, lo = la_lo.split(',')
            tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'
            map_di = folium.Map(location=[float(lo), float(la)],width=600,height=800, tiles=tiles, attr='高德-常规图', zoom_start=12, control_scale=True)
            data = (np.random.normal(size=(100, 2)) * np.array([[1, 1]]) +
                        np.array([[float(lo), float(la)]]))
            categories = ['good', 'broke', 'lose', 'uncovered', 'circle']
            category_column = [random.choice(categories) for i in range(len(data))]
            icons = {
                "good": 'fa-light fa-g',
                "broke": 'fa-light fa-b',
                "lose": 'fa-light fa-l',
                'uncovered': 'fa-light fa-u',
                'circle': 'fa-light fa-c'
            }
            colors = {
                "good": 'lightgreen',
                "broke": 'lightpink',
                "lose": 'lightgray',
                'uncovered': 'orange',
                'circle': 'lightblue'
            }
            for i, latlng in enumerate(data):
                category = category_column[i]
                folium.Marker(
                    tuple(latlng),
                    tags=[category],
                    icon=folium.Icon(color=colors[category], icon=icons[category], prefix='fa')
                ).add_to(map_di)

            plugins.TagFilterButton(categories).add_to(map_di)
            return map_di.get_root().render()
          

@app.route('/getConfig', methods=['GET', 'POST'])
def getConfig():
    if request.method == 'GET':
        with open('/app/train/config.json', 'r') as f:
            json_data = f.read()
            config = json.loads(json_data)
            return config
    
    if request.method == 'POST':
        config = 0
        with open('/app/train/config.json', 'r') as f:
            json_data = f.read()
            config = json.loads(json_data)
        pos_data = request.get_json()
        config['standard'] = pos_data['standard']
        config['class_config'] = pos_data['class_config']
        config['detect_config'] = pos_data['detect_config']
        with open('/app/train/config.json', 'w') as f:
            json.dump(config, f)
        return jsonify({"state": "修改成功"})


@app.route('/trainNow', methods=['GET', 'POST'])
def trainNow():
    with open('/app/train/config.json', 'r') as f:
        json_data = f.read()
        config = json.loads(json_data)
    
    config['num'] = 0
    global epoch
    state =  "启动成功"
    if epoch > 0:
        state =  "模型正在运行"
        return jsonify({"state": state})
    epoch = config['class_config']['epoch']
    def trainNowRe():
        from utils.trainClass import retrain
        from utils.trainYolo import yoloRetrain
        # yoloRetrain()
        retrain(True)
    # trainNowRe()
    # process = Process(target=trainNowRe)
    # process.start()
    with open('/app/train/config.json', 'w') as f:
        json.dump(config, f)
    return jsonify({"state": state})
    
        
@app.route("/logs", methods=['GET', 'POST'])
def job_log():
    if request.method == 'GET':
        return render_template('index2.html')


@app.route('/chart-data', methods=['GET', 'POST'])
def chart_data():
    print('chart_data-' * 5)
    def generate_random_data():
        global epoch, val_auc, train_auc
        print('generate_random_data-' * 5)
        tms = 0
        def rdn(num):
            random_number = random.normalvariate(num, 0.03)
            print(random_number)
            # 将随机数限制在0.89和0.95之间
            return max(num - 0.03, min(random_number, num + 0.03))
        # while epoch > 0:
        for i in range(10):
            tms += 1
            json_data = json.dumps(
                {'time': tms, 'value1': rdn(0.95), 'value2': rdn(0.91)})
            # epoch -= 1
            # # 1 SSE 返回格式是json字符串，要使用yield返回，字符串后面一定要跟随 \n\n
            yield f"data:{json_data}\n\n"
            time.sleep(1)  # 1s 发送一次
	# 2 stream_with_context 设置SSE连接函数，mimetype="text/event-stream" 是设置SSE返回格式
    response = Response(stream_with_context(generate_random_data()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, host='0.0.0.0', port=5000)

