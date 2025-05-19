import os
import json
import cv2
from .class_config import * # 从同级目录的 class_config 文件中导入所有内容，主要是 CFG 配置类
import time
import timm # PyTorch Image Models (timm) 库，包含了大量预训练模型
import random
import numpy as np
import albumentations as A # 图像增强库
import xml.etree.ElementTree as ET # 用于解析 XML 文件

from .trainYolo import train # 从同级目录的 trainYolo 文件导入 train 函数 (这里可能存在命名冲突或误用，因为本文件也有一个 train 函数)
from tqdm import tqdm # 用于显示进度条
from torch.utils.data import Dataset, DataLoader # PyTorch 数据集和数据加载器
import numpy as np
import pandas as pd
import torch.nn as nn # PyTorch 神经网络模块
import torch.nn.functional as F # PyTorch 神经网络函数库
from sklearn.metrics import confusion_matrix # 用于计算混淆矩阵

from tqdm import tqdm
from PIL import Image # Python Imaging Library (PIL) 的一个分支 Pillow，用于图像处理


# 定义一个网络包装器，在预训练模型的输出后添加自定义的全连接层
class Network_Wrapper(nn.Module):
    def __init__(self, model, num_class):
        super(Network_Wrapper, self).__init__()
        self.model = model # 预训练模型
        hidden_units = 128 # 自定义全连接层的隐藏单元数
        # 定义全连接层序列：ReLU -> Linear(1000, hidden_units) -> ReLU -> Linear(hidden_units, num_class)
        # 假设预训练模型输出特征维度为 1000
        self.fc = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(1000, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, num_class) # 输出层，输出维度为类别数
                )
        
    def forward(self, x):
        # 前向传播：输入 x 先通过预训练模型，然后通过自定义的全连接层
        return self.fc(self.model(x))


# Focal Loss 实现，用于处理类别不平衡问题
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # alpha 参数，用于平衡正负样本的重要性
        self.gamma = gamma # gamma 参数，用于调整易分类样本的权重，使其更关注难分类样本
        self.reduction = reduction # 指定损失的计算方式：'mean', 'sum', or 'none'

    def forward(self, inputs, targets):
        # 计算二元交叉熵损失 (Binary Cross Entropy)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        p_t = torch.exp(-bce_loss) # 计算预测概率
        # 计算 Focal Loss: alpha * (1 - p_t)^gamma * bce_loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss) # 返回平均损失
        elif self.reduction == 'sum':
            return torch.sum(focal_loss) # 返回总损失
        else:
            return focal_loss # 返回每个样本的损失
        

# 从图像中裁剪出目标区域，并可选择性地在训练时进行小幅度的随机扩展
def crop_image(image_path, x_min, y_min, x_max, y_max, training=False):
    # image_path: 原始图像路径
    # x_min, y_min, x_max, y_max: 目标区域的边界框坐标
    # training: 是否为训练阶段，训练阶段会进行随机扩展
    original_image = Image.open(image_path).convert('RGB') # 打开图像并转换为 RGB 格式
    # 计算边界框宽高的 1/10 作为扩展量的基准
    dis_x = (x_max - x_min) // 10
    dis_y = (y_max - y_min) // 10
    if training: # 如果是训练阶段
        width, height = original_image.size
        # 在原始边界框基础上随机扩展，但不超出图像边界
        x_max = min(width, x_max + random.randint(0, dis_x))
        x_min = max(0, x_min - random.randint(0, dis_x))
        y_max = min(height, y_max + random.randint(0, dis_y))
        y_min = max(0, y_min - random.randint(0, dis_y))
    else: # 如果是验证或测试阶段
        width, height = original_image.size
        # 在原始边界框基础上固定扩展一半的基准量，但不超出图像边界
        x_max = min(width, x_max + dis_x // 2)
        x_min = max(0, x_min - dis_x // 2)
        y_max = min(height, y_max + dis_y // 2)
        y_min = max(0, y_min - dis_y // 2)
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max)) # 裁剪图像
    return cropped_image
    
    
# 余弦退火学习率调度函数
def cosine_anneal_schedule(t, nb_epoch, lr):
    # t: 当前训练步数或轮数 (0-indexed)
    # nb_epoch: 总的训练轮数 (一个周期内的轮数)
    # lr: 初始学习率
    cos_inner = np.pi * (t % (nb_epoch))  # 计算余弦函数的内部参数
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1 # 计算余弦值并调整范围到 [0, 2]

    return float(lr / 2 * cos_out) # 返回当前轮数对应的学习率


# 计算模型预测的正确数量和混淆矩阵
def getCorrect(output_concat, target):
    # output_concat: 模型输出的 logits
    # target: 真实标签 (one-hot 编码)
    _, predicted = torch.max(output_concat.data, 1) # 获取预测类别 (概率最大的类别索引)
    _, targets = torch.max(target, 1) # 获取真实类别索引
    targets = targets.int()
    equal = predicted.eq(targets).cpu() # 比较预测和真实标签是否相等
    # 计算混淆矩阵，标签范围从 0 到 CFG.num_classes - 1
    return equal, confusion_matrix(targets.cpu(), predicted.cpu(), labels=range(CFG.num_classes))


# 在验证集上评估模型性能 (一个 epoch)
@torch.no_grad() # 不计算梯度，节省显存和计算资源
def valid_one_epoch(net, criterion, testloader):
    net.eval() # 设置模型为评估模式
    use_cuda = torch.cuda.is_available() # 检查 CUDA 是否可用
    test_loss = 0 # 初始化测试损失
    correct = 0 # 初始化正确预测数量
    total = 0 # 初始化总样本数量
    idx = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置设备
    # 初始化总的混淆矩阵
    total_cm = np.zeros((CFG.num_classes, CFG.num_classes), dtype=int)
    for batch_idx, (inputs, targets, path) in tqdm(enumerate(testloader), total=len(testloader), desc='valid '):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device).float() # 将数据移到 GPU (如果可用)
        outputs = net(inputs) # 模型前向传播

        loss = criterion(outputs, targets) # 计算损失

        test_loss += loss.item() # 累加损失
        predicted_eq, cm = getCorrect(outputs.data, targets.data) # 获取当前 batch 的正确预测情况和混淆矩阵
        total_cm += cm # 累加混淆矩阵
        total += targets.size(0) # 累加样本总数
        correct += predicted_eq.sum() # 累加正确预测数

        # 打印当前 batch 的评估结果
        print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
        batch_idx, test_loss / (batch_idx + 1),
        100. * float(correct) / total, correct, total))

    test_acc_en = 100. * float(correct) / total # 计算整体准确率
    test_loss = test_loss / (idx + 1) # 计算平均损失
    accuracies = [] # (此变量未使用)
    # 计算归一化的混淆矩阵 (按行归一化，表示每个真实类别的预测分布)
    normalized_cm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    print(normalized_cm) # 打印归一化的混淆矩阵

    return test_acc_en, test_loss


# Label Smoothing Cross Entropy Loss 实现
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0 # 平滑因子 smoothing 必须在 (0, 1) 之间
        self.smoothing = smoothing
        self.confidence = 1. - smoothing # 置信度因子

    def forward(self, x, targets):
        # x: 模型输出的 logits
        # targets: 真实标签 (one-hot 编码)
        _, target = torch.max(targets, 1) # 获取真实类别索引
        y_hat = torch.softmax(x, dim=1) # 对模型输出计算 softmax 得到概率分布
        
        # 计算标准的交叉熵损失 (只考虑真实类别对应的概率)
        cross_loss = self.cross_entropy(y_hat, target)
        # 计算平滑项 (所有类别的负对数似然的平均值)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        # Label Smoothing 损失 = confidence * cross_loss + smoothing * smooth_loss
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean() # 返回平均损失

    # 辅助函数，计算标准交叉熵 (选取真实标签对应的预测概率的负对数)
    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])


# 设置随机种子，用于保证实验的可复现性
def set_seed(seed=42):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) # 如果使用 CUDA，也设置 CUDA 的随机种子

    
# 构建图像增强的变换 (transforms)
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([ # 训练集的图像增强序列
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0), # 调整图像大小，使用最近邻插值
            A.HorizontalFlip(p=0.5), # 50% 概率水平翻转
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], p=1.0), # 归一化到 [-1, 1]
            A.Compose([ # 以 0.25 的概率应用以下组合增强
                A.RandomCrop(height=200, width=200, p=1.0), # 随机裁剪到 200x200
                A.PadIfNeeded(min_height=CFG.img_size[0], min_width=CFG.img_size[1], value=[0, 0, 0], p=1.0), # 如果裁剪后尺寸小于目标尺寸，则填充
            ], p=0.25),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.25), # 随机平移、缩放、旋转
            A.ChannelShuffle(p=0.5), # 50% 概率通道混洗
            A.OneOf([ # 从以下增强中随机选择一个应用，概率为 0.25
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0), # 网格失真
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0), # 光学失真 (被注释掉了)
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0) # 弹性变换
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20, # Cutout/随机擦除
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5), # 50% 概率应用
            ]),
        
        "valid_test": A.Compose([ # 验证集和测试集的图像变换序列 (通常只做必要的调整和归一化)
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0), # 调整图像大小
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], p=1.0) # 归一化
            ])
        }
    return data_transforms


# 自定义 PyTorch 数据集类
class build_dataset(Dataset):
    def __init__(self, objects, objects_circle, val=False, train_val_flag=True, transforms=None):
        # objects: 主要类别的对象列表
        # objects_circle: 'circle' 类别（或其他特定类别）的对象列表，可能用于特殊处理或采样
        # val: 是否为验证集模式
        # train_val_flag: 标记是用于训练/验证 (True) 还是仅用于推理 (False，此时不返回标签)
        # transforms: 应用于图像的变换
        self.objects = objects
        self.objects_circle = objects_circle
        self.train_val_flag = train_val_flag
        self.transforms = transforms
        self.lenth_cir = len(self.objects_circle)
        self.lenth = len(self.objects)
        # 计算 'circle' 类别在总样本中的比例，用于训练时的采样
        self.rate = self.lenth_cir / (self.lenth + self.lenth_cir) if (self.lenth + self.lenth_cir) > 0 else 0
        self.val = val
        if self.val: # 如果是验证集，则合并两类对象
            self.dataset = objects + objects_circle
            
    def __len__(self):
        # 数据集的总长度
        return (len(self.objects_circle) + len(self.objects))

    def __getitem__(self, index):
        if not self.val: # 如果是训练集
            # 根据之前计算的 rate，按比例从 objects 或 objects_circle 中采样
            if random.random() > self.rate:
                obj = self.objects[index % self.lenth if self.lenth > 0 else 0]
            else:
                obj = self.objects_circle[index % self.lenth_cir if self.lenth_cir > 0 else 0]
        else: # 如果是验证集
            obj = self.dataset[index]
            
        # 从图像路径和边界框裁剪出目标区域，训练时进行随机扩展 (1 - self.val)
        img = crop_image(obj["path"], obj['objects']["xmin"], obj['objects']["ymin"], 
                            obj['objects']["xmax"], obj['objects']["ymax"], training=(1 - self.val == 1))
        
        if self.train_val_flag: # 如果是训练或验证模式
            data = self.transforms(image=np.array(img)) # 应用图像变换
            img = np.transpose(data['image'], (2, 0, 1)) # 将图像从 HWC 转为 CHW (PyTorch格式)
            # 将标签转换为 one-hot 编码
            label = F.one_hot(torch.tensor(obj["objects"]["label"]), num_classes=CFG.num_classes)
            return torch.tensor(img), torch.from_numpy(np.array(label).astype(int)), obj["path"]
        else: # 如果是推理模式 (train_val_flag=False)
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1))
            return torch.tensor(img), obj["path"] # 只返回图像和路径，不返回标签 (id 变量未定义，应为 obj["path"])
        

# 从 XML 标注文件中解析数据，类似于 class_config.py 中的 xml2pd，但这里区分了 objects 和 objects_circle
def xml2pd(directory):
    label_mapping = { # 标签映射
        'good': 0,
        'broke': 1,
        'lose': 2,
        'loss': 2,
        'uncovered': 3,
        'circle': 4
    }
    width = 0 # (未使用)
    height = 0 # (未使用)
    objects = [] # 存储非 'circle' 类的对象
    objects_circle = [] # 存储 'circle' 类的对象
    for i in directory: # directory 是一个路径列表
        for root_dir, dirs, files in os.walk(i): # 遍历每个路径下的文件
            for file in files:
                if file.endswith(".xml"):
                    path = os.path.join(i, file) # XML 文件路径 (这里应该用 root_dir 替换 i 来正确处理子目录)
                    # path = os.path.join(root_dir, file) # 修正：应使用 root_dir
                    tree = ET.parse(path)
                    root = tree.getroot()
                    dataPath = os.path.normpath(os.path.join(os.path.dirname(path), root.find('path').text))
                    if os.path.basename(dataPath) == 'well5_0011.jpg': # 跳过特定的图片
                        continue
                    for obj_xml in root.findall('.//object'): # XML 中找到的每个对象
                        try:
                            sort = obj_xml.find('name').text
                        except:
                            sort = obj_xml.find('sort').text
                        temp = { # 存储对象信息
                                'name': sort,
                                'label': label_mapping[sort],
                                'xmin': int(obj_xml.find('.//xmin').text),
                                'ymin': int(obj_xml.find('.//ymin').text),
                                'xmax': int(obj_xml.find('.//xmax').text),
                                'ymax': int(obj_xml.find('.//ymax').text),
                            }
                        file_name, file_extension = os.path.splitext(root.find('filename').text)
                        # 根据标签将对象分别存入 objects 和 objects_circle列表
                        if temp["label"] != 4: # 非 'circle' 类别
                            objects.append({'path': dataPath,'objects': temp, "width": width, "height": height, "name": file_name})
                        elif temp['label'] == 4: # 'circle' 类别
    #                         temp['label'] = 1 # (这行代码被注释掉了，原本可能想将 circle 映射到 broke)
                            objects_circle.append({'path': dataPath,'objects': temp, "width": width, "height": height, "name": file_name})
    random.shuffle(objects) # 打乱列表顺序
    random.shuffle(objects_circle)
    return objects, objects_circle


# 构建训练和验证数据加载器 (DataLoader)
def build_dataloader(objects, objects_circle, valid_objects, valid_objects_circle, data_transforms):
    # objects, objects_circle: 训练数据
    # valid_objects, valid_objects_circle: 验证数据
    # data_transforms: 图像变换
    train_dataset = build_dataset(objects, objects_circle, val=False, train_val_flag=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_objects, valid_objects_circle, val=True, train_val_flag=True, transforms=data_transforms['valid_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                                drop_last=False) # drop_last=False 意味着最后一个不完整的 batch 也会被使用
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    return train_loader, valid_loader


# 构建模型
def build_model(CFG, pretrained=None, modelName=None):
    # CFG: 配置对象
    # pretrained: 预训练权重路径 (如果为 None，则使用 CFG.pretrained)
    # modelName: 模型名称 (如果为 None，则使用 CFG.modelName)
    if not modelName:
        modelName = CFG.modelName
    if not pretrained:
        pretrained = CFG.pretrained
        
    # 使用 timm 库创建模型，num_classes=1000 是因为 timm 预训练模型通常在 ImageNet (1000类) 上训练
    # pretrained=False 表示不加载 timm 自带的预训练权重，因为我们稍后会加载自定义的权重或从头开始训练然后加载自定义的包装器权重
    model = timm.create_model(modelName, pretrained=False, num_classes=1000)

    # 使用 Network_Wrapper 包装模型，添加自定义的分类头
    net = Network_Wrapper(model, CFG.num_classes)
    # 加载预训练权重 (这里的 pretrained 指的是我们自己训练的 Network_Wrapper 的权重)
    net.load_state_dict(torch.load(pretrained, map_location=torch.device(CFG.device)))
    return net


# 构建损失函数，结合了 Label Smoothing Cross Entropy 和 Focal Loss
def build_loss(x, y):
    # x: 模型输出 logits
    # y: 真实标签 (one-hot)
    smooth_loss_fn = LabelSmoothingCrossEntropy(0.1)
    focal_loss_fn = FocalLoss()
    
    # 组合两种损失，按 1/6 和 5/6 的权重
    loss = smooth_loss_fn(x, y) / 6 + focal_loss_fn(x, y) * 5 / 6
    return loss


# 训练模型一个 epoch
def train_one_epoch(net, netp, trainloader, CELoss, optimizer, lr, 
                    batch_size, epoch, nb_epoch, use_cuda, device):
    # net: 原始模型
    # netp: DataParallel 包装后的模型 (用于多GPU训练)
    # trainloader: 训练数据加载器
    # CELoss: 损失函数
    # optimizer: 优化器
    # lr: 学习率列表 (每个参数组一个)
    # batch_size: 批量大小
    # epoch: 当前轮数 (0-indexed for cosine_anneal_schedule)
    # nb_epoch: 总轮数
    # use_cuda: 是否使用 CUDA
    # device: 设备
    net.train() # 设置模型为训练模式
    train_loss = 0
    correct = 0
    total = 0
    idx = 0  
    for batch_idx, (inputs, targets, path) in tqdm(enumerate(trainloader), total=len(trainloader), desc='Train '):
        idx = batch_idx
        if inputs.shape[0] < batch_size: # 如果当前 batch 大小小于定义的 batch_size，则跳过 (通常在 drop_last=False 时发生)
            continue
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device).float()

        # 更新每个参数组的学习率 (使用余弦退火)
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

        optimizer.zero_grad() # 清空梯度
        output = netp(inputs) # 模型前向传播 (使用 DataParallel 包装的模型)
        loss = CELoss(output, targets) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        # 计算当前 batch 的准确率
        correctBatch, cm = getCorrect(output.data, targets.data)
        total += targets.size(0)
        correct += correctBatch.sum()

        train_loss += loss.item()
        # 打印训练信息
        print(
            'Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, train_loss / (batch_idx + 1),
            100. * float(correct) / total, correct, total))
    return correct # 返回当前 epoch 训练集上的总正确数 (似乎应该是准确率或平均损失)


# 主训练函数
def train(CFG, log=False):
    # CFG: 配置对象
    # log: 是否记录训练日志到 JSON 文件
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}") # 打印 CUDA 可用状态
    data_transforms = build_transforms(CFG)  # 构建图像变换
    # 加载训练和验证数据 (从 XML 文件解析)
    objects, objects_circle = xml2pd(CFG.train_path)
    valid_objects, valid_objects_circle = xml2pd(CFG.valid_path)
    # 构建数据加载器
    train_loader, valid_loader = build_dataloader(objects, objects_circle, valid_objects, valid_objects_circle,  data_transforms)
    
    net = build_model(CFG) # 构建模型并加载预训练权重 (CFG.pretrained)
    netp = torch.nn.DataParallel(net, device_ids=[0]) # 使用 DataParallel 支持多GPU (这里只用了 device_ids=[0])
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device) # 将模型移到设备
    
    # 定义优化器 (SGD)
    optimizer = torch.optim.SGD(net.parameters(), CFG.lr, 
    momentum=0.9, weight_decay=1e-3)
    
    # best_val_acc 初始化为 0.99，这可能是一个较高的初始值，意味着只有超过这个值的模型才会被保存
    # 理想情况下，应该从第一个验证 epoch 的结果初始化，或者从 0 开始
    # valAcc, loss = valid_one_epoch(net, build_loss, valid_loader) # 这行被注释掉了，原本可能用于获取初始验证准确率
    best_val_acc = 0.0 # 修改：初始化为0.0，更合理
    lr = [CFG.lr] # 学习率列表 (因为只有一个参数组，所以只有一个学习率)
    results = {} # 用于存储日志结果

    if log and os.path.exists('/app/train/result.json'): # 如果启用日志且文件存在，则清空
        try:
            with open('/app/train/result.json', 'w') as f:
                json.dump({}, f)
        except IOError:
            print("Error: Could not clear result.json") 

    for epoch_num in range(1, CFG.epoch+1): # PyTorch通常epoch从1开始计数，但train_one_epoch内部的cosine_anneal_schedule期望0-indexed epoch
        print(f'\nEpoch: {epoch_num}')
        start_time = time.time()
        # 训练一个 epoch，注意 epoch 传给 train_one_epoch 是 epoch_num-1 (0-indexed)
        train_correct_count = train_one_epoch(net, netp, train_loader, build_loss, optimizer, lr,
                        CFG.train_bs, epoch_num-1, CFG.epoch, use_cuda, device)
        # train_acc = 100. * float(train_correct_count) / len(train_loader.dataset) # 计算训练准确率
        # (上面的 train_acc 计算方式可能不完全准确，因为 train_one_epoch 返回的是正确数量，且可能跳过最后一个 batch)
        
        val_acc, val_loss = valid_one_epoch(net, build_loss, valid_loader) # 验证一个 epoch
        
        current_train_loss = "N/A" # train_one_epoch 未返回平均损失，这里暂时设为 N/A
        # (如果需要记录训练损失，train_one_epoch 需要修改以返回平均损失)
        
        print(f"Epoch {epoch_num} Summary: Val Acc: {val_acc:.3f}%, Val Loss: {val_loss:.4f}")

        if log: # 如果启用日志
            if not os.path.exists('/app/train/result.json') or os.path.getsize('/app/train/result.json') == 0 : 
                 results = {}
            else:
                try:
                    with open('/app/train/result.json', 'r') as f:
                        results = json.load(f)
                except json.JSONDecodeError:
                    results = {} # 如果文件内容不是有效的JSON，则重置

            results[epoch_num] = {'train_acc': "N/A", 'val_acc': val_acc, 'loss': val_loss} # train_acc 暂时记录为N/A
            try:
                with open('/app/train/result.json', 'w') as f:
                    json.dump(results, f, indent=4)
            except IOError:
                print(f"Error: Could not write to result.json at epoch {epoch_num}")

        is_best = (val_acc > best_val_acc)
        best_val_acc = max(best_val_acc, val_acc)
        if is_best: # 如果当前验证准确率是最佳的
            save_path = f"/app/utils/new_weight/best.pth" # 模型保存路径
            if os.path.isfile(save_path): # 如果已存在同名文件，先删除 (避免意外加载旧的、更大的模型)
                try:
                    os.remove(save_path) 
                except OSError as e:
                    print(f"Error removing old best model: {e}")
            try:
                torch.save(net.state_dict(), save_path) # 保存模型参数
                print(f"New best model saved to {save_path} with Val Acc: {val_acc:.3f}%")
            except IOError as e:
                print(f"Error saving model: {e}")
        
        epoch_time = time.time() - start_time
        print("epoch:{}, time:{:.2f}s, best_val_acc:{:.2f}%\n".format(epoch_num, epoch_time, best_val_acc), flush=True)
    
    # 训练结束后，再次确保 result.json 被清空 (根据原始代码逻辑)
    # 但通常我们可能希望保留最后一个训练周期的日志
    # 如果目的是在 retrain 开始时清空，那么应该在 retrain 函数的开头处理
    # 原始代码在这里清空，意味着训练日志只在训练过程中临时存在，最终会被清空
    # with open('/app/train/result.json', 'w') as json_file:
    #     json.dump({}, json_file)
    # print("result.json has been cleared after training completion.")


# 导出模型到 ONNX 格式
def classExport(CFG, pretrained=None, modelName=None):
    import torch.onnx
    if not modelName:
        modelName = CFG.modelName
    if not pretrained: # 如果没有提供 pretrained 路径，则使用 CFG 中的路径
        pretrained = CFG.pretrained
        
    # 构建模型并加载指定的预训练权重
    # 注意：这里 build_model 内部会加载 pretrained 路径的权重
    net = build_model(CFG, pretrained=pretrained, modelName=modelName) 
    net.eval() # 设置为评估模式
    
    dummy_input = torch.randn(1, 3, CFG.img_size[0], CFG.img_size[1]).to(CFG.device)  # 创建一个符合模型输入的虚拟输入张量
    onnx_path = "/app/utils/weight/class.onnx"  # 保存ONNX模型的路径
    
    print(f"Exporting model to ONNX format at {onnx_path}...")
    try:
        torch.onnx.export(net, dummy_input, onnx_path, verbose=True, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
        print(f"Model exported successfully to {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
    
    
# 构建 ONNX 推理会话
def buildInferModel(path="/app/utils/weight/class.onnx"):
    import onnxruntime
    try:
        onnx_session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"ONNX inference session created successfully from {path} using {onnx_session.get_providers()}")
    except Exception as e:
        onnx_session = None
        print(f"Error creating ONNX inference session: {e}")
    return onnx_session


# 将 new 文件夹中的图片和xml按80/20比例移动到 new_train 和 new_valid 文件夹
def deliver():
    source_dir = "/app/train/new/" # 源数据文件夹
    dest_dir_train = "/app/train/new_train" # 训练数据目标文件夹
    dest_dir_val = "/app/train/new_valid" # 验证数据目标文件夹

    # 确保目标文件夹存在
    os.makedirs(dest_dir_train, exist_ok=True)
    os.makedirs(dest_dir_val, exist_ok=True)

    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(filenames) # 打乱文件顺序以便随机分配

    for filename in tqdm(filenames, desc="Delivering files"): # 添加进度条
        image_file = os.path.join(source_dir, filename)
        # 假设 xml 文件名与图片文件名（除后缀外）相同
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_file = os.path.join(source_dir, xml_filename)

        if not os.path.exists(xml_file):
            print(f"Warning: XML file {xml_file} not found for image {image_file}. Skipping.")
            continue

        # 随机分配到训练集或验证集
        if random.random() > 0.2: # 80% 概率分配到训练集
            dest_dir_current_img = dest_dir_train
            dest_dir_current_xml = dest_dir_train
        else: # 20% 概率分配到验证集
            dest_dir_current_img = dest_dir_val
            dest_dir_current_xml = dest_dir_val
        
        dest_image_file = os.path.join(dest_dir_current_img, filename)
        dest_xml_file = os.path.join(dest_dir_current_xml, xml_filename)

        try:
            shutil.move(image_file, dest_image_file) # 移动图片文件
            shutil.move(xml_file, dest_xml_file) # 移动 XML 文件
        except Exception as e:
            print(f"Error moving file {filename} or its XML: {e}")

    print("Data delivery complete.")

# 重新训练模型的入口函数
def retrain(log=False):
    set_seed(CFG.seed if hasattr(CFG, 'seed') else 42) # 设置随机种子，优先使用CFG中的seed
    
    # 重新组织数据：将 /app/train/new/ 中的数据按比例分配到 /app/train/new_train 和 /app/train/new_valid
    print("Starting data delivery...")
    deliver()
    
    # 清空之前的训练日志 (如果需要，并且启用日志)
    if log:
        results_json_path = '/app/train/result.json'
        try:
            with open(results_json_path, 'w') as f:
                json.dump({}, f)
            print(f"{results_json_path} cleared for new training log.")
        except IOError:
            print(f"Error: Could not clear {results_json_path} for new training log.")

    # 开始训练
    print("Starting training...")
    train(CFG, log=log)
    
    # 导出训练好的模型到 ONNX (使用 CFG.pretrained，这里假设它指向最新保存的 best.pth)
    print("Exporting model to ONNX...")
    # classExport 函数默认使用 CFG.pretrained，这应该是 train 函数中保存的最佳模型路径
    # 但 train 函数保存的是 /app/utils/new_weight/best.pth
    # 而 CFG.pretrained 默认是 /app/utils/weight/best.pth (来自 class_config.py)
    # 需要确保 classExport 使用的是刚训练好的模型
    # 最佳实践是让 train 函数返回最佳模型的路径，或者 classExport 直接接收路径参数
    
    # 假设我们希望导出刚刚训练并保存的模型
    # 注意：CFG.pretrained 在 class_config.py 中定义，可能不是最新模型的路径
    # 为了确保导出最新的模型，我们应该使用 train() 函数中保存的路径
    latest_model_path = "/app/utils/new_weight/best.pth"
    if os.path.exists(latest_model_path):
        classExport(CFG, pretrained=latest_model_path) 
    else:
        print(f"Warning: Latest model at {latest_model_path} not found. Exporting with CFG.pretrained or default.")
        classExport(CFG) # 回退到CFG中定义的预训练路径

    print("Retraining process complete.")


