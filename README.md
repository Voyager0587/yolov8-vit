// 如果运行有错误可以更换为python版本为 3.8
// https://blog.csdn.net/weixin_41883450/article/details/132341040


### 2.3 Vision Transformer分类模块

对于第二阶段的细粒度分类任务，我们采用了Vision Transformer (ViT)模型。其处理流程如下：

1. 将YOLOv8检测到的井盖区域裁剪并调整为标准尺寸（224×224像素）
2. 将图像划分为$N×N$个固定大小的patches（如16×16像素）
3. 每个patch通过线性映射转换为嵌入向量，并添加位置编码：

$$
z_0 = [x_{\text{class}}; x_p^1E; x_p^2E; ...; x_p^NE] + E_{pos}
$$

其中$E \in \mathbb{R}^{(P^2·C) \times D}$为嵌入矩阵，$E_{pos}$为位置编码

4. 通过$L$层Transformer Encoder进行特征编码：

$$
z_l' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}, \quad l=1...L
$$

$$
z_l = \text{MLP}(\text{LN}(z_l')) + z_l', \quad l=1...L
$$

其中MSA表示多头自注意力机制，LN表示层归一化，MLP为多层感知机。

5. 最终分类头通过class token输出预测结果：

$$
y = \text{MLP}(\text{LN}(z_L^0))
$$

## 3. 模型优化与实现细节

### 3.1 自定义NMS实现

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

### 3.2 框膨胀策略

为弥补YOLOv8定位误差，我们实现了边界框膨胀策略。给定原始边界框$(x_{min}, y_{min}, x_{max}, y_{max})$，膨胀后的边界框计算如下：

$$
x_{min}' = x_{min} - \alpha \cdot w
$$

$$
x_{max}' = x_{max} + \alpha \cdot w
$$

$$
y_{min}' = y_{min} - \alpha \cdot h
$$

$$
y_{max}' = y_{max} + \alpha \cdot h
$$

其中$\alpha$为膨胀系数（设为0.1），$w$和$h$分别为边界框的宽和高。实现时还需考虑图像边界约束：

$$
x_{min}' = \max(0, x_{min}')
$$

$$
y_{min}' = \max(0, y_{min}')
$$

$$
x_{max}' = \min(W, x_{max}')
$$

$$
y_{max}' = \min(H, y_{max}')
$$

其中$W$和$H$为图像宽度和高度。

### 3.3 模型加速与部署优化