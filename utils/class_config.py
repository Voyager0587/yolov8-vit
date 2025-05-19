import torch
from sklearn.model_selection import StratifiedKFold
import xml.etree.ElementTree as ET # 用于解析 XML 文件
from PIL import Image # 用于图像处理
import numpy as np
import shutil # 用于文件操作，如复制
import random
import os


# 定义一个配置类，用于存储训练和模型相关的超参数
class CFG:
        seed = 42 # 随机种子，保证实验可复现性
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置设备，优先使用 CUDA GPU
        img_size = [224, 224] # 图像尺寸
        train_bs = 1 # 训练时的批量大小
        valid_bs = train_bs * 2 # 验证时的批量大小
        num_classes = 5 # 数据集中的类别数量
        epoch = 10 # 训练的总轮数
        lr = 1e-4 # 学习率
        modelName = "vit_base_patch8_224.augreg_in21k" # 使用的模型名称
        pretrained = '/app/utils/weight/best.pth' # 预训练权重的路径
        train_path = ["/app/train/new_train", "/app/train/circle", "/app/train/2024/train_xmls", "/app/train/new"] # 训练数据路径列表
        valid_path = ["/app/train/2024/valid_xmls", "/app/train/new_valid"] # 验证数据路径列表


# 将边界框坐标从 (xmin, ymin, xmax, ymax) 格式转换为 YOLO 格式 (center_x, center_y, width, height)，并进行归一化
def convert(box,dw,dh):
    # box: (xmin, ymin, xmax, ymax)
    # dw: 图像原始宽度
    # dh: 图像原始高度
    x=(box[0]+box[2])/2.0 # 计算中心点 x 坐标
    y=(box[1]+box[3])/2.0 # 计算中心点 y 坐标
    w=box[2]-box[0] # 计算边界框宽度
    h=box[3]-box[1] # 计算边界框高度

    x=x/dw # 归一化中心点 x 坐标
    y=y/dh # 归一化中心点 y 坐标
    w=w/dw # 归一化宽度
    h=h/dh # 归一化高度

    return x,y,w,h


# 复制图片文件到指定文件夹
def copy_image(source_path, destination_folder):
    # source_path: 源图片文件路径
    # destination_folder: 目标文件夹路径
    if not os.path.exists(destination_folder): # 如果目标文件夹不存在
        os.makedirs(destination_folder) # 创建目标文件夹

    image_file_name = os.path.basename(source_path) # 获取源图片文件名

    destination_path = os.path.join(destination_folder, image_file_name) # 构建目标图片文件完整路径

    shutil.copy(source_path, destination_path) # 复制图片


# 创建用于交叉验证的文件夹结构
def mkdir(number):
    # number: 折叠 (fold) 的编号
    fold = f"./fold{number}" # 定义当前折叠的文件夹名称
    images_dir = os.path.join(fold, "images") # 图片存放路径
    label_dir = os.path.join(fold, "labels") # 标签存放路径
    if not os.path.exists(fold): # 如果当前折叠的文件夹不存在
        os.mkdir(fold) # 创建当前折叠的文件夹
        os.mkdir(images_dir) # 创建图片文件夹
        os.mkdir(label_dir) # 创建标签文件夹
        os.mkdir(os.path.join(images_dir, "train")) # 创建训练集图片子文件夹
        os.mkdir(os.path.join(label_dir, "train")) # 创建训练集标签子文件夹
        os.mkdir(os.path.join(images_dir, "val")) # 创建验证集图片子文件夹
        os.mkdir(os.path.join(label_dir, "val")) # 创建验证集标签子文件夹


# 将解析后的目标对象信息写入 YOLO 格式的 txt 文件
def writeTxt(path, objects):
    # path: 输出 txt 文件的路径 (不含 .txt 后缀)
    # objects: 包含图像宽高和目标对象列表的字典
    txt_o = open(f"{path}.txt", 'w') # 打开（或创建）txt 文件用于写入
    for box in objects['objects']: # 遍历图像中的所有目标对象
        # 将边界框转换为 YOLO 格式并归一化
        x, y, w, h = convert((box['xmin'], box['ymin'], box['xmax'], box['ymax']), objects["width"], objects["height"])
        # 格式化输出字符串：类别标签 中心点x 中心点y 宽度 高度
        write_t = "{} {:.5f} {:.5f} {:.5f} {:.5f}\\n".format(box["label"], x, y, w, h)
        txt_o.write(write_t) # 写入文件


# 从指定目录读取 XML 标注文件，解析其中的目标对象信息，并将其转换为 YOLO 格式的 txt 文件，同时复制相应的图片
def xml2pd(directory):
    # directory: 包含 XML 标注文件的目录路径
    label_mapping = { # 类别名称到数字标签的映射
        'good': 0,
        'broke': 1,
        'lose': 2,
        'loss': 2, # 'loss' 和 'lose' 映射到同一个标签
        'uncovered': 3,
        'circle': 4
    }
    # 移除Jupyter Notebook产生的检查点文件夹（如果存在）
    if os.path.exists(f"{os.path.join(directory, '.ipynb_checkpoints')}"):
        os.rmdir(f"{os.path.join(directory, '.ipynb_checkpoints')}")
    width = 0 # 初始化图像宽度
    height = 0 # 初始化图像高度
    objects = [] # 用于存储所有图像及其标注信息的列表
    for root, dirs, files in os.walk(directory): # 遍历指定目录下的所有文件和子目录
        for file in files: # 遍历当前目录下的所有文件
            if file.endswith(".xml"): # 如果文件是 XML 文件
                temp = [] # 临时列表，用于存储当前 XML 文件中的所有目标对象信息
                path = os.path.join(directory, file) # 构建 XML 文件的完整路径
                tree = ET.parse(path) # 解析 XML 文件
                root = tree.getroot() # 获取 XML 树的根元素
                # 获取图像文件的路径，通常在 XML 文件中指定
                # os.path.normpath 用于规范化路径表示
                dataPath = os.path.normpath(os.path.join(os.path.dirname(path), root.find('path').text))
                # 获取图像的宽度和高度
                with Image.open(dataPath) as img: # 打开图像文件
                # 获取图像的长和宽
                    # 优先从 XML 文件中读取宽高信息
                    if int(root.find('size/width').text) and int(root.find('size/height').text):
                        width, height = int(root.find('size/width').text), int(root.find('size/height').text)
                    else: # 如果 XML 中没有宽高信息，则从图像文件本身获取
                        width, height = img.size
                for obj in root.findall('.//object'): # 遍历 XML 文件中所有的 'object' 标签（即目标对象）
                        try:
                            sort = obj.find('name').text # 尝试获取 'name' 标签下的类别名称
                        except:
                            sort = obj.find('sort').text # 如果没有 'name' 标签，尝试获取 'sort' 标签下的类别名称
                        temp.append({ # 将目标对象信息存入字典
                                'name': sort, # 原始类别名称
                                'label': label_mapping[sort], # 映射后的数字标签
                            'xmin': int(obj.find('.//xmin').text), # 边界框左上角 x 坐标
                            'ymin': int(obj.find('.//ymin').text), # 边界框左上角 y 坐标
                            'xmax': int(obj.find('.//xmax').text), # 边界框右下角 x 坐标
                            'ymax': int(obj.find('.//ymax').text), # 边界框右下角 y 坐标
                        })
                file_name, file_extension = os.path.splitext(root.find('filename').text) # 获取不带后缀的文件名
                objects.append({'path': dataPath,'objects': temp, "width": width, "height": height, "name": file_name}) # 将当前图像及其所有目标对象信息存入列表
    # 遍历所有解析到的图像信息
    for i in objects:
        # 随机分配图像到训练集或验证集 (80% 训练集, 20% 验证集)
        if random.random() > 0.2: # 80% 的概率分到训练集
            dis_image = "/app/train/yolo/fold0/images/train" # 训练集图片目标路径
            dis_labels = "/app/train/yolo/fold0/labels/train" # 训练集标签目标路径
        else: # 20% 的概率分到验证集
            dis_image = "/app/train/yolo/fold0/images/val" # 验证集图片目标路径
            dis_labels = "/app/train/yolo/fold0/labels/val" # 验证集标签目标路径
        copy_image(i["path"], dis_image) # 复制图片到目标路径
        writeTxt(f"{dis_labels}/{i['name']}", i) # 将目标对象信息写入 YOLO 格式的 txt 文件

                    
# 将指定路径下的 XML 标注文件转换为 YOLO 格式的 txt 文件
def xml2txt(path):
    # path: 包含 XML 标注文件的目录路径
    xml2pd(path) # 调用 xml2pd 函数进行处理


        
