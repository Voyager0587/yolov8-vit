from ultralytics import YOLO # 导入 Ultralytics YOLO 库，用于目标检测模型的训练和验证
from .class_config import xml2txt # 从同级目录的 class_config.py 导入 xml2txt 函数，用于将 XML 转换为 YOLO 格式的 txt 标签


# 定义 YOLO 模型训练函数
def train(epochs, batch, data):
    # epochs: 训练的总轮数
    # batch: 训练时的批量大小
    # data: 指向数据配置文件 (通常是 .yaml 文件) 的路径，该文件定义了训练集、验证集路径以及类别信息
    
    # 加载预训练的 YOLO 模型权重。这里使用的是 "/app/utils/weight/best.pt"，
    # 这意味着它期望一个已经存在的 .pt 文件作为初始权重，可能是之前训练保存的最佳权重，或者是官方的预训练权重。
    model = YOLO("/app/utils/weight/best.pt") 
    
    # 在训练开始前，可选地在验证集上进行一次评估 (model.val)
    # imgsz: 验证时图像大小设置为 640x640
    # batch: 验证时的批量大小设置为 16
    # conf: 验证时的置信度阈值设置为 0.25
    # iou: 验证时计算 mAP 的 IoU (Intersection over Union) 阈值设置为 0.6
    # device: 指定设备为 '0' (通常指 GPU 0)
    validation_results = model.val(data=data,
                               imgsz=640,
                               batch=16,
                               conf=0.25,
                               iou=0.6,
                               device='0')
    print("Validation results before training:", validation_results) # 打印验证结果
    
    # 开始训练模型
    # epochs, batch, data 参数与函数输入相同
    # lr0: 初始学习率设置为 0.0001
    # lrf: 最终学习率 (学习率衰减后的学习率) 也设置为 0.0001，这表明可能没有使用大幅度的学习率衰减，或者使用了特定的学习率调度器
    results = model.train(epochs=epochs, batch=batch, data=data, lr0=0.0001, lrf=0.0001)
    
    return results # 返回训练结果，通常包含训练过程中的指标和最终模型的信息

# 将指定目录下的 XML 标注文件内容转换为特定的字典列表格式，并结合了 TensorRT 推理引擎的初始化 (尽管推理部分被注释掉了)
# 这个函数名 yolo2dict 有些误导，因为它主要是处理 XML 文件，而不是 YOLO 的 txt 格式输出。
# 它的目标似乎是将 XML 标注整理成一个列表，其中每个元素是一个元组 (图片名, [检测框信息列表])
def yolo2dict(path_to_xml_dir): # 参数名修改为 path_to_xml_dir 以明确其作用
    import os
    import cv2 # (cv2 未在此函数中显式使用)
    import shutil # (shutil 未在此函数中显式使用)
    import xml.etree.ElementTree as ET # 用于解析 XML
    import torch
    # 以下是 YOLOTensorRT 推理相关的导入和初始化，但实际推理代码 main(Engine, path, device) 被注释了
    # from YOLOTensorRT.yolodet import main, draw_image # (draw_image 未使用)
    # from YOLOTensorRT.models import TRTModule
    # device = "cuda:0"
    # device = torch.device(device)
    # engine_path = "/app/utils/new_weight/try_7.engine" # 引擎路径变量名修改以避免与类名冲突
    # Engine_trt = TRTModule(engine_path, device) # 类名通常大写开头，变量名小写或下划线
    # Engine_trt.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    
    # 函数内部重新定义了 path 变量，覆盖了函数参数 path，这通常是不好的做法。假设这里的 path 是 XML 文件夹。
    # path_to_xml_dir = r"/app/TensorRT-8.4.2.4/test" # 使用函数参数，并注释掉此行

    label_mapping = { # 标签名称到数字的映射
        'good': 0,
        'broke': 1,
        'lose': 2,
        'loss': 2, # 'loss' 也映射到 2
        'uncovered': 3,
        'circle': 4
    }

    # 定义一个内部函数，用于解析单个 XML 文件并提取目标对象信息
    def parse_xml_to_dict(xml_file):
        tree = ET.parse(xml_file) # 解析 XML 文件
        root = tree.getroot() # 获取 XML 树的根元素
        
        obj_info_list = [] # 存储从 XML 中提取的每个对象的信息
        
        for obj in root.findall('object'): # 遍历 XML 中所有的 'object' 标签
            try:
                obj_name_str = obj.find('name').text # 尝试获取 'name' 标签的内容作为类别名
            except AttributeError: # 如果 'name' 标签不存在或为空，尝试 'sort'
                obj_name_str = obj.find('sort').text
            
            # 将类别名转换为数字标签：如果是纯数字字符串 '0'-'4'，则直接转为整数；否则通过 label_mapping 查找
            if obj_name_str in ['0', '1', '2', '3', '4']:
                obj_label = int(obj_name_str)
            else:
                obj_label = label_mapping.get(obj_name_str, -1) # 如果映射中没有，默认为 -1 (表示未知或忽略)

            bbox = obj.find('bndbox') # 获取边界框信息
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            obj_info_list.append({'name': obj_label, 'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max})
        
        return obj_info_list

    # 定义一个内部函数，用于读取指定目录下的所有 XML 文件，并调用 parse_xml_to_dict进行处理
    def read_xml_files_in_directory(directory):
        xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
        
        result_list_of_tuples = [] # 存储结果，每个元素是 (图片文件名, [对象信息列表])
        
        for xml_file in xml_files:
            image_filename_base = os.path.splitext(xml_file)[0] # 获取不带 .xml 后缀的文件名
            image_extension = '.jpg' # 默认图片后缀为 .jpg
            if image_filename_base == "test152": # 特殊处理 "test152"，其后缀为 .png
                image_extension = '.png'
            image_filename = image_filename_base + image_extension # 完整的图片文件名
            
            xml_full_path = os.path.join(directory, xml_file) # XML 文件的完整路径
            obj_info_list_for_image = parse_xml_to_dict(xml_full_path) # 解析当前 XML
            result_list_of_tuples.append((image_filename, obj_info_list_for_image)) # 添加到结果列表
            
        return result_list_of_tuples

    # TensorRT 推理部分被注释掉了
    # main(Engine_trt, path_to_xml_dir, device) 
    
    # 调用 read_xml_files_in_directory 处理指定路径 (函数参数 path_to_xml_dir)
    results = read_xml_files_in_directory(path_to_xml_dir)
    results.sort(key=lambda x: x[0]) # 按图片文件名对结果进行排序
    return results
    
    
# YOLO 模型重新训练的入口函数
def yoloRetrain():
    # 步骤 1: 将位于 "/app/train/new" 目录下的 XML 标注文件转换为 YOLO TXT 格式。
    # xml2txt 函数 (来自 class_config.py) 会处理这个转换，并将结果保存到YOLO期望的目录结构中。
    print("Converting XML annotations to YOLO TXT format...")
    xml2txt("/app/train/new") 
    
    # 步骤 2: 调用 train 函数开始 YOLO 模型的训练
    # epochs: 训练 1 轮 (通常实际训练会设置更多轮数)
    # batch: 批量大小为 1 (通常会根据 GPU 显存设置为更大的值，如 8, 16, 32)
    # data: 数据配置文件路径为 "/app/train/yolo/config.yaml"
    # 这个 yaml 文件应包含训练集、验证集图片的路径，以及类别数量和名称信息。
    print("Starting YOLO model retraining...")
    training_results = train(epochs=1, batch=1, data="/app/train/yolo/config.yaml")
    print("YOLO model training finished. Results:", training_results)
    
    