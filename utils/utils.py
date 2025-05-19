import os
import requests
import cv2
import oss2
import numpy as np
import timm
import torch
import torch.nn as nn
from flask_sse import sse
import xml.etree.ElementTree as ET

def download_images(url, save_folder, save_flag = True):
    # url: 图片的URL地址
    # save_folder: 如果 save_flag 为 True，图片将保存到此文件夹
    # save_flag: 布尔值，决定是否将下载的图片保存到文件 (True) 或仅返回图像数据 (False)
    try:
        response = requests.get(url, timeout=10) # 发送 GET 请求下载图片，设置超时时间为10秒
        response.raise_for_status() # 如果请求失败 (状态码不是 2xx), 则抛出 HTTPError 异常
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False # 下载失败返回 False

    nparr = np.frombuffer(response.content, np.uint8) # 将响应内容 (图片字节流) 转换为 NumPy 数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 将 NumPy 数组解码为 OpenCV 图像格式 (BGR)
    
    if image is None: # 检查图像是否成功解码
        print(f"Error decoding image from {url}")
        return False

    if save_flag: # 如果需要保存图片
        # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 这行重复了，可以移除
        image_filename = os.path.basename(url) # 从 URL 中提取文件名
        # 处理可能导致文件名无效的查询参数
        image_filename = image_filename.split('?')[0]
        if not image_filename: # 如果文件名为空（例如 URL 以 '/' 结尾）
            # 尝试从 Content-Disposition header 获取文件名，或生成一个唯一文件名
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                import re
                fname = re.findall("filename*?=([^;]+)", content_disposition, flags=re.IGNORECASE)
                if fname:
                    image_filename = fname[0].strip('"\' ')
            if not image_filename: # 如果还是没有文件名，生成一个基于时间的
                import time
                image_filename = f"downloaded_image_{int(time.time())}.jpg"
        
        save_path = os.path.join(save_folder, image_filename) # 构建完整的保存路径
        os.makedirs(save_folder, exist_ok=True) # 确保保存文件夹存在
        try:
            cv2.imwrite(save_path, image) # 将图像写入文件
            return save_path # 返回保存后的文件路径
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
            return False
    else: # 如果不需要保存图片
        return image # 直接返回解码后的 OpenCV 图像对象


class Network_Wrapper(nn.Module):
    def __init__(self, model, num_class):
        super(Network_Wrapper, self).__init__()
        self.model = model # 预训练模型
        hidden_units = 128 # 自定义全连接层的隐藏单元数
        self.fc = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(1000, hidden_units), # 假设预训练模型输出维度为 1000
                    nn.ReLU(),
                    nn.Linear(hidden_units, num_class) # 输出层，输出维度为类别数
                )
        
    def forward(self, x):
        return self.fc(self.model(x))
    
    
def build_model(CFG, modelName, pretrained_path):
    # CFG: 配置对象，应包含 num_classes 和 device 属性
    # modelName: 要从 timm 加载的模型名称
    # pretrained_path: 自定义训练好的 Network_Wrapper 模型的权重路径
    
    # 从 timm 创建基础模型，不加载 timm 的预训练权重 (pretrained=False)，因为我们之后会加载自己的权重
    # num_classes=1000 是 timm 模型的常见默认输出（如 ImageNet）
    model = timm.create_model(modelName, pretrained=False, num_classes=1000)

    net = Network_Wrapper(model, CFG.num_classes) # 使用 Network_Wrapper 包装基础模型
    # 加载我们自己训练好的权重到 Network_Wrapper 实例中
    net.load_state_dict(torch.load(pretrained_path, map_location=torch.device(CFG.device)))
    return net


class AliyunOss(object):

    def __init__(self):
        # 从环境变量或配置文件中读取这些敏感信息是更安全的做法
        self.access_key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID", "LTAI5tBDpqYK4utyCFEejFHj")
        self.access_key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET", "xtttuoSistE41T3BlMcxGZrQnKKhAb")
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret) # 授权
        self.bucket_name = "xiaowenjie" # OSS Bucket 名称
        self.endpoint = "oss-cn-beijing.aliyuncs.com" # OSS Endpoint
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name) # 创建 Bucket 操作对象

    # 上传本地文件到 OSS
    def put_object_from_file(self, name, file_path): # 参数名修改为 file_path 更清晰
        # name: 在 OSS 上存储的对象名称 (包含路径)
        # file_path: 本地文件的路径
        try:
            self.bucket.put_object_from_file(name, file_path)
            print(f"Successfully uploaded {file_path} to OSS as {name}")
            return True
        except Exception as e:
            print(f"Failed to upload {file_path} to OSS: {e}")
            return False
    
    # 获取 OSS 上对象的公开访问 URL
    def getUrl(self, name):
        # name: OSS 上的对象名称
        return "https://{}.{}/{}".format(self.bucket_name, self.endpoint, name)
    
    # 删除 OSS 上的对象
    def delete_object(self, name):
        # name: OSS 上的对象名称
        try:
            self.bucket.delete_object(name)
            print(f"Successfully deleted {name} from OSS.")
            return True
        except oss2.exceptions.NoSuchKey:
            print(f"Object {name} not found in OSS. Nothing to delete.")
            return False # 或者根据需求返回 True，因为目标状态（不存在）已达到
        except oss2.exceptions.OssError as e:
            print(f"Error deleting object {name} from OSS: {e}")
            return False


def generate_annotation(folder_name, image_filename, image_path, objects_data, save_dir="train/new/"):
    # folder_name: XML 中 <folder> 标签的内容
    # image_filename: XML 中 <filename> 标签的内容 (e.g., "image1.jpg")
    # image_path: XML 中 <path> 标签的内容 (图像的实际路径)
    # objects_data: 一个包含多个目标对象信息的列表，每个对象是一个字典，
    #               例如 {'sort': 'good', 'xmin': 10, 'ymin': 20, 'xmax': 100, 'ymax': 120}
    # save_dir: XML 文件保存的目录
    
    root = ET.Element("annotation") # 创建根元素 <annotation>

    # 添加 <folder> 子节点
    folder_node = ET.SubElement(root, "folder")
    folder_node.text = folder_name

    # 添加 <filename> 子节点
    filename_node = ET.SubElement(root, "filename")
    filename_node.text = image_filename

    # 添加 <path> 子节点
    path_node = ET.SubElement(root, "path")
    path_node.text = image_path

    # 添加 <source> 子节点及其子节点 <database>
    source_node = ET.SubElement(root, "source")
    database_node = ET.SubElement(source_node, "database")
    database_node.text = "Unknown"

    # 添加 <size> 子节点，包含图像的宽、高、深度
    # 注意：这里的宽高硬编码为 "0"，实际应用中应该从图像文件读取真实宽高
    size_node = ET.SubElement(root, "size")
    width_node = ET.SubElement(size_node, "width")
    width_node.text = "0" # TODO: 应替换为图像真实宽度
    height_node = ET.SubElement(size_node, "height")
    height_node.text = "0" # TODO: 应替换为图像真实高度
    depth_node = ET.SubElement(size_node, "depth")
    depth_node.text = "3" # 通常彩色图像深度为 3

    # 添加 <segmented> 子节点
    segmented_node = ET.SubElement(root, "segmented")
    segmented_node.text = "0" # 0 表示非分割任务
    
    label_mapping = { # 类别名称到数字标签字符串的映射 (用于写入 XML)
        'good': '0',
        'broke': '1',
        'lose': '2',
        'loss': '2', # 'loss' 也映射到 '2'
        'uncovered': '3',
        'circle': '4'
    }

    for obj_data in objects_data: # 遍历每个目标对象的信息
        object_node = ET.SubElement(root, "object") # 创建 <object> 节点
        
        name_node = ET.SubElement(object_node, "sort") # 类别名节点 (XML中用了 <sort>)
        # 如果 obj_data['sort'] 是整数，直接转为字符串；否则通过 label_mapping 转换
        # 假设 sort 可能是数字标签也可能是字符串标签
        sort_value = obj_data['sort']
        if isinstance(sort_value, int):
            name_node.text = str(sort_value)
        elif isinstance(sort_value, str):
            name_node.text = label_mapping.get(sort_value, str(sort_value)) # 如果映射不到，用原始字符串
        else:
            name_node.text = "unknown" # 未知类型处理
        
        pose_node = ET.SubElement(object_node, "pose")
        pose_node.text = "Unspecified"
        truncated_node = ET.SubElement(object_node, "truncated")
        truncated_node.text = "0"
        difficult_node = ET.SubElement(object_node, "difficult")
        difficult_node.text = "0"
        
        bndbox_node = ET.SubElement(object_node, "bndbox") # 边界框节点
        xmin_node = ET.SubElement(bndbox_node, "xmin")
        xmin_node.text = str(obj_data['xmin'])
        ymin_node = ET.SubElement(bndbox_node, "ymin")
        ymin_node.text = str(obj_data['ymin'])
        xmax_node = ET.SubElement(bndbox_node, "xmax")
        xmax_node.text = str(obj_data['xmax'])
        ymax_node = ET.SubElement(bndbox_node, "ymax")
        ymax_node.text = str(obj_data['ymax'])

    tree = ET.ElementTree(root) # 创建 XML 树

    indent(root) # 调用 indent 函数格式化 XML (添加缩进和换行)
    
    os.makedirs(save_dir, exist_ok=True) # 确保保存目录存在
    # 构建输出的 XML 文件名，使用 image_filename 的主名 + .xml
    output_xml_filename = os.path.join(save_dir, f"{os.path.splitext(image_filename)[0]}.xml")
    
    try:
        tree.write(output_xml_filename, encoding="utf-8", xml_declaration=False) # 将 XML 树写入文件
        print(f"Annotation XML saved to {output_xml_filename}")
        return output_xml_filename
    except Exception as e:
        print(f"Error writing XML to {output_xml_filename}: {e}")
        return None

def indent(elem, level=0):
    # elem: 当前 XML 元素
    # level: 当前缩进级别
    i = "\n" + level * "  " # 缩进字符串 (换行 + level*2个空格)
    if len(elem): # 如果元素有子元素
        if not elem.text or not elem.text.strip(): # 如果元素的文本内容为空或只有空白
            elem.text = i + "  " # 设置元素的文本为一个缩进
        if not elem.tail or not elem.tail.strip(): # 如果元素的尾部内容为空或只有空白
            elem.tail = i # 设置元素的尾部为一个缩进 (元素闭合标签后的换行和缩进)
        for sub_elem in elem: # 遍历所有子元素
            indent(sub_elem, level + 1) # 递归调用 indent，增加缩进级别
        if not sub_elem.tail or not sub_elem.tail.strip(): # 处理最后一个子元素的尾部
            sub_elem.tail = i
    else: # 如果元素没有子元素 (叶子节点)
        if level and (not elem.tail or not elem.tail.strip()): # 如果不是根元素且尾部为空
            elem.tail = i # 设置尾部为一个缩进
            
            
def location2lalo(location):
    # location: 地理位置描述字符串 (例如 "北京市朝阳区阜通东大街6号")
    # 需要在高德开放平台申请 Key (1d06fc8c365d6f6c720b89c14d21a9bf 是示例Key)
    # 强烈建议将 Key 存储在环境变量或安全配置文件中，而不是硬编码
    api_key = os.environ.get("AMAP_API_KEY", "1d06fc8c365d6f6c720b89c14d21a9bf")
    if api_key == "1d06fc8c365d6f6c720b89c14d21a9bf":
        print("Warning: Using a sample Amap API Key. Please replace with your own key.")
        
    parameters = {'address': location, 'key': api_key}
    base_url = 'https://restapi.amap.com/v3/geocode/geo' # 高德地理编码 API 地址
    try:
        response = requests.get(base_url, params=parameters, timeout=5) # 发送 GET 请求
        response.raise_for_status() # 检查请求是否成功
        answer = response.json() # 解析 JSON 响应
        
        if answer.get('status') == '1' and answer.get('geocodes'):
            formatted_address = answer['geocodes'][0]['formatted_address']
            longitude_latitude = answer['geocodes'][0]['location'] # "经度,纬度"
            return formatted_address, longitude_latitude
        else:
            print(f"Error from Amap API: {answer.get('info', 'Unknown error')}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error requesting Amap API: {e}")
        return None, None
    except (KeyError, IndexError) as e:
        print(f"Error parsing Amap API response: {e}. Response: {answer}")
        return None, None


def log(log_queue_obj, message, *args):
    # log_queue_obj: 一个队列对象 (例如 multiprocessing.Queue)，用于存储日志消息
    # message: 日志消息模板 (使用 %s, %d 等占位符)
    # *args: 格式化消息模板的参数
    try:
        formatted_message = message % args # 格式化日志消息
        if hasattr(log_queue_obj, 'put'):
            log_queue_obj.put(formatted_message) # 将消息放入队列
        else:
            print("Warning: log_queue_obj does not have a 'put' method.")
        sse.publish({'message': formatted_message}, type='log') # 通过 SSE 发布日志消息
    except Exception as e:
        print(f"Error in log function: {e}")

