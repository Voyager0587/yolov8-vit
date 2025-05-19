import torch
from utils.utils import *
from utils.class_config import CFG
import albumentations as A
from YOLOTensorRT.inferdet import main, draw_image
from YOLOTensorRT.models import TRTModule


device = "cuda:0"
device = torch.device(device)
engine = "/app/utils/new_weight/95_np.engine"
Engine = TRTModule(engine, device)
Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
model_path_list = ["vit_base_patch8_224.augreg_in21k"]
path_list = ["/app/utils/new_weight/strong.pth"]
model_list = []
for i in range(len(path_list)):
    model_list.append(build_model(CFG=CFG, modelName=model_path_list[i], pretrained=path_list[i]))
    model_list[-1].to(CFG.device)
    model_list[-1].eval()
    
transform = {"valid_test": A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], p=1.0)
])}


main(Engine = Engine, imgs = "/app/image/", device = device, model_list=model_list, transform=transform, aliyunoss=None, func=generate_annotation)


