// 如果运行有错误可以更换为python版本为 3.8
// https://blog.csdn.net/weixin_41883450/article/details/132341040


### 2.3 Vision Transformer分类模块

对于第二阶段的细粒度分类任务，我们采用了Vision Transformer (ViT)模型。具体实现如下：

1. 模型配置:
```python
modelName = "vit_base_patch8_224.augreg_in21k"  # 使用timm预训练模型
img_size = [224, 224]  # 输入尺寸
num_classes = 5        # 分类数量
```

2. 分类头设计:
```python
self.fc = nn.Sequential(
    nn.ReLU(),
    nn.Linear(1000, 128),  # 降维
    nn.ReLU(),
    nn.Linear(128, num_class)  # 分类层
)
```

3. 损失函数:
- Label Smoothing Cross Entropy (权重:1/6)
- Focal Loss (权重:5/6)
```python
loss = smooth_loss(x, y) / 6 + focal_loss(x, y) * 5 / 6
```

## 3. 模型优化与实现细节

### 3.1 EfficientNMS 实现

本项目使用 TensorRT 的 EfficientNMS 插件进行非极大值抑制:

```python
# YOLOTensorRT/models/api.py
plugin_creator = trt.get_plugin_registry().get_plugin_creator('EfficientNMS_TRT', '1')

# NMS 参数配置
iou_threshold = 0.65  # IoU 阈值
conf = 0.25          # 置信度阈值
topk = 100          # 最大输出检测框数
```

### 3.2 自定义NMS实现

为解决同一井盖被多个检测框识别的问题，我们设计了自定义非极大值抑制（NMS）算法。算法流程如下：

1. 置信度过滤：根据预设阈值$\theta_c$（如0.35）过滤低置信度预测框

    $$
    B_{\text{filtered}} = \{b_i | \text{conf}(b_i) > \theta_c, b_i \in B\}
    $$
2. 按面积排序：计算每个预测框面积$A_i = w_i \times h_i$，并按面积降序排列

    $$
    B_{\text{sorted}} = \text{sort}(B_{\text{filtered}}, \text{key}=\text{Area}, \text{reverse}=\text{True})
    $$
3. IoU计算与筛选：依次处理排序后的预测框，保留IoU低于阈值$\theta_{IoU}$的框

    $$
    B_{\text{final}} = \text{NMS}(B_{\text{sorted}}, \theta_{IoU})
    $$

此算法通过PyTorch实现，利用矩阵运算提高计算效率：

```python
def custom_nms(boxes, scores, iou_threshold=0.45):
    # 按分数降序排列的索引
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.size(0) > 0:
        # 取分数最高的框
        i = sorted_indices[0].item()
        keep.append(i)
        
        # 如果只剩一个框，则结束循环
        if sorted_indices.size(0) == 1:
            break
            
        # 计算IoU
        ious = box_iou(boxes[i:i+1], boxes[sorted_indices[1:]])
        
        # 保留IoU低于阈值的框
        mask = ious.squeeze() < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
        
    return keep
```

### 3.3 框膨胀策略

实际实现中,我们采用动态膨胀比例:

```python
def crop_image(image_path, x_min, y_min, x_max, y_max, training=False):
    # 计算膨胀距离
    dis_x = (x_max - x_min) // 10
    dis_y = (y_max - y_min) // 10
    
    if training:
        # 训练时随机膨胀
        x_max = min(width, x_max + random.randint(0, dis_x))
        x_min = max(0, x_min - random.randint(0, dis_x))
        y_max = min(height, y_max + random.randint(0, dis_y))
        y_min = max(0, y_min - random.randint(0, dis_y))
    else:
        # 推理时固定膨胀
        x_max = min(width, x_max + dis_x // 2)
        x_min = max(0, x_min - dis_x // 2)
        y_max = min(height, y_max + dis_y // 2)
        y_min = max(0, y_min - dis_y // 2)
```

这种动态膨胀策略可以更好地适应不同大小的检测框。

### 3.4 数据增强策略

为提高模型鲁棒性,实现了以下数据增强:

```python
transforms = A.Compose([
    A.Resize(*CFG.img_size),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.RandomCrop(200, 200, p=0.25),
    A.ShiftScaleRotate(p=0.25),
    A.ChannelShuffle(p=0.5),
    A.CoarseDropout(p=0.5)
])
```

### 3.5 模型加速与部署优化

本项目采用了多种优化策略来提升模型性能和部署效率：

1. TensorRT 加速
```python
# YOLO检测模型TensorRT加速
Engine = TRTModule(engine_path, device)  
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

# ViT分类模型ONNX导出
def classExport(CFG):
    net = build_model(CFG)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(net, dummy_input, 
                     "/app/utils/weight/class.onnx",
                     verbose=True)
```

2. 批处理优化
```python
CFG:
    train_bs = 1      # 训练批大小
    valid_bs = train_bs * 2  # 验证批大小
```

3. 数据加载优化
- 使用 PIL 和 OpenCV 混合加载策略
- 训练时使用多进程数据加载
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=CFG.train_bs,
    num_workers=0,  # 可根据CPU核心数调整
    shuffle=True,
    pin_memory=True  # 固定内存加速GPU传输
)
```

4. 推理优化
- 使用 TensorRT 的 EfficientNMS 插件替代原生 NMS
- 动态 batch size 适应不同场景需求
- 支持 FP16 推理加速
```python
# 推理配置
background_class = -1  # 无背景类
box_coding = 0        # 边界框编码
score_activation = 0  # 分数激活函数
plugin_version = '1'  # 插件版本
```

5. 内存优化
- 使用 torch.no_grad() 减少推理显存占用
- 及时清理中间变量释放内存
```python
@torch.no_grad()
def valid_one_epoch(net, criterion, testloader):
    net.eval()
    # ... 验证代码
```

这些优化措施显著提升了模型的训练和推理效率，使系统能够满足实际部署需求。