# utils/class_config.py 文件解读

本文档详细分析 `utils/class_config.py` 脚本的功能和内部工作流程。该脚本主要定义了一个配置类 `CFG`，用于存储模型训练和数据处理相关的超参数，并提供了一系列用于将PASCAL VOC XML标注格式转换为YOLO TXT标注格式的辅助函数。

## 1. 主要功能

*   **配置管理**：通过 `CFG` 类集中管理项目中的重要参数，如学习率、图像尺寸、模型名称、数据路径等。
*   **数据格式转换**：提供将XML标注文件转换为YOLO TXT格式的函数，这是训练YOLO模型前重要的数据预处理步骤。
*   **文件与目录操作**：包含复制图像、创建特定目录结构（如交叉验证的fold）的辅助函数。

## 2. 依赖项

脚本主要依赖以下库：

*   `torch`: PyTorch库，主要用于设备选择 (`torch.device`)。
*   `sklearn.model_selection.StratifiedKFold`: (导入但未使用) 用于分层K折交叉验证的数据划分。
*   `xml.etree.ElementTree` (as `ET`): Python内置库，用于解析XML文件。
*   `PIL.Image` (Pillow): 用于图像处理，如此脚本中用于打开图像以获取其尺寸。
*   `numpy`: (导入但未使用) 用于数值计算。
*   `shutil`: Python内置库，用于高级文件操作，如复制文件 (`shutil.copy`)。
*   `random`: Python内置库，用于生成随机数，如此脚本中用于随机划分训练/验证集。
*   `os`: Python内置库，用于操作系统相关功能，如路径操作、文件和目录检查、创建目录等。

## 3. `CFG` 配置类

`CFG` 类是一个简单的Python类，用作一个命名空间来存储全局配置参数。这使得参数管理更加集中和方便修改。

*   `seed = 42`: 随机种子，用于确保实验的可复现性。
*   `device = torch.device(...)`: 根据CUDA是否可用，自动选择GPU (`cuda:0`) 或CPU作为计算设备。
*   `img_size = [224, 224]`: 模型训练时输入图像的目标尺寸。
*   `train_bs = 1`: 训练时的批量大小 (Batch Size)。
*   `valid_bs = train_bs * 2`: 验证时的批量大小。
*   `num_classes = 5`: 数据集中目标类别的数量。
*   `epoch = 10`: 模型训练的总轮数。
*   `lr = 1e-4`: 学习率。
*   `modelName = "vit_base_patch8_224.augreg_in21k"`: 使用的预训练模型名称（可能来自`timm`库）。
*   `pretrained = '/app/utils/weight/best.pth'`: 预训练权重文件的路径。
*   `train_path = [...]`: 包含训练数据（XML文件所在目录）的路径列表。
*   `valid_path = [...]`: 包含验证数据（XML文件所在目录）的路径列表。

## 4. 函数解析

### 4.1. `convert(box, dw, dh)`

*   **功能**：将单个边界框的坐标从 `(xmin, ymin, xmax, ymax)` 格式（PASCAL VOC常用格式）转换为YOLO TXT格式所需的 `(center_x, center_y, width, height)` 格式，并对结果进行归一化。
*   **参数**：
    *   `box`: 一个元组或列表，包含4个整数 `(xmin, ymin, xmax, ymax)`。
    *   `dw`: 原始图像的宽度。
    *   `dh`: 原始图像的高度。
*   **逻辑**：
    1.  计算边界框的中心点 `x = (xmin + xmax) / 2.0` 和 `y = (ymin + ymax) / 2.0`。
    2.  计算边界框的宽度 `w = xmax - xmin` 和高度 `h = ymax - ymin`。
    3.  将计算得到的 `x, y, w, h` 分别除以图像的原始宽度 `dw` 和高度 `dh` 进行归一化，使结果在 `0.0` 到 `1.0` 之间。
*   **返回**：归一化后的 `(x, y, w, h)`。

### 4.2. `copy_image(source_path, destination_folder)`

*   **功能**：将指定的源图像文件复制到目标文件夹。
*   **参数**：
    *   `source_path`: 源图像文件的完整路径。
    *   `destination_folder`: 要将图像复制到的目标文件夹路径。
*   **逻辑**：
    1.  检查目标文件夹是否存在，如果不存在，则使用 `os.makedirs()` 创建它。
    2.  获取源图像的文件名。
    3.  构建完整的目标文件路径。
    4.  使用 `shutil.copy()` 执行文件复制操作。

### 4.3. `mkdir(number)`

*   **功能**：创建一个特定结构的目录，通常用于为交叉验证的特定"折" (fold) 准备数据存储位置。
*   **参数**：
    *   `number`: 折的编号 (整数)。
*   **逻辑**：
    1.  创建一个名为 `./fold{number}` 的主文件夹。
    2.  在主文件夹内创建 `images` 和 `labels` 两个子文件夹。
    3.  在 `images` 和 `labels` 子文件夹内分别创建 `train` 和 `val` 子文件夹。
    *   最终目录结构示例 (假设 `number=0`): 
        ```
        ./fold0/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        ```

### 4.4. `writeTxt(path, objects)`

*   **功能**：将单个图像的多个目标对象标注信息（已经过`convert`函数处理）写入YOLO格式的TXT文件。
*   **参数**：
    *   `path`: 输出TXT文件的基本路径（不含 `.txt` 后缀，例如 `labels/train/image_name`）。
    *   `objects`: 一个字典，包含图像的原始宽度 `objects["width"]`、高度 `objects["height"]`，以及一个对象列表 `objects['objects']`。列表中的每个对象是一个字典，包含数字标签 `box["label"]` 和原始边界框坐标 `box['xmin']`, `box['ymin']`, `box['xmax']`, `box['ymax']`。
*   **逻辑**：
    1.  打开（或创建）一个以 `{path}.txt` 命名的文件用于写入。
    2.  遍历 `objects['objects']` 列表中的每个目标对象：
        *   调用 `convert()` 函数将该对象的边界框坐标转换为YOLO格式并归一化。
        *   将数字类别标签和归一化后的 `x, y, w, h` 格式化成一行字符串：`label_id center_x center_y width height`。
        *   将该行字符串写入TXT文件。

### 4.5. `xml2pd(directory)`

这是脚本中的核心函数，负责批量处理XML标注文件。

*   **功能**：遍历指定目录下的所有PASCAL VOC XML标注文件，解析它们，提取图像路径、图像尺寸以及每个目标对象的类别和边界框信息。然后，将这些信息转换为YOLO TXT格式，并将原始图像和对应的TXT标签文件复制/保存到预设的训练集和验证集目录中 (硬编码为 `/app/train/yolo/fold0/...`)。
*   **参数**：
    *   `directory`: 包含XML标注文件的源目录路径。
*   **逻辑**：
    1.  **标签映射 (`label_mapping`)**: 定义一个字典，将字符串类别名 (如 `'good'`, `'broke'`) 映射到整数类别ID (如 `0`, `1`)。
    2.  **遍历XML文件**: 使用 `os.walk()` 递归遍历 `directory` 目录下的所有文件。
    3.  **解析单个XML文件**: 对于每个 `.xml` 后缀的文件：
        *   使用 `ET.parse()` 解析XML。
        *   从XML中提取图像的路径 (`root.find('path').text`)。
        *   使用 `Image.open()` 打开图像，并从中获取原始宽度和高度。优先尝试从XML的 `<size>` 标签读取宽高，如果不存在，则从图像文件本身获取。
        *   遍历XML中的所有 `<object>` 标签（代表一个标注对象）：
            *   提取类别名称（尝试 `<name>` 标签，若失败则尝试 `<sort>` 标签）。
            *   使用 `label_mapping` 将类别名称转换为数字类别ID。
            *   提取边界框坐标 `xmin, ymin, xmax, ymax`。
            *   将提取到的对象信息（原始类别名、数字标签、边界框坐标）存入一个临时列表 `temp`。
        *   将当前图像的完整信息（图像路径、包含所有对象的 `temp` 列表、图像宽高、不带后缀的文件名）作为一个字典存入 `objects` 列表。
    4.  **分配与写入**: 遍历 `objects` 列表中的每个图像信息：
        *   **随机划分**: 以80%的概率将当前图像分配到训练集，20%的概率分配到验证集。
            *   目标路径被硬编码为：
                *   训练集图片: `/app/train/yolo/fold0/images/train`
                *   训练集标签: `/app/train/yolo/fold0/labels/train`
                *   验证集图片: `/app/train/yolo/fold0/images/val`
                *   验证集标签: `/app/train/yolo/fold0/labels/val`
        *   **复制图像**: 调用 `copy_image()` 将原始图像复制到分配好的训练集或验证集图片目录中。
        *   **写入TXT标签**: 调用 `writeTxt()` 将该图像的标注信息（经过 `convert` 格式化后）写入到分配好的训练集或验证集标签目录中，TXT文件名与原图像名（无后缀）相同。

### 4.6. `xml2txt(path)`

*   **功能**：一个简单的包装函数，直接调用 `xml2pd(path)` 来执行从XML到YOLO TXT的转换过程。
*   **参数**：
    *   `path`: 包含XML标注文件的源目录路径。

## 5. 总结

`utils/class_config.py` 脚本为项目提供了一个集中的配置管理方案 (`CFG` 类) 和一套完整的数据预处理工具，专门用于将PASCAL VOC XML格式的标注数据转换为YOLO模型训练所需的TXT格式。这包括解析XML、坐标转换、归一化、文件复制以及按训练/验证集组织输出文件。对于理解项目的数据准备流程至关重要。 