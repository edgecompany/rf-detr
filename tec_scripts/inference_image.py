import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
import sys
sys.path.append('.')
sys.path.append('..')

from time import time
import numpy as np
from glob import glob
from pathlib import Path
from shai_inference import SlicedTrackerInference

from datamodules.DetectionDataModule import LitDetectionDataModule
from utils import pil_resize_img

slice_size = 728
conf_threshold = 0.12
class_id_bird = [class_id for class_id in COCO_CLASSES.keys() if COCO_CLASSES[class_id] == 'bird'][0]
print(f"Bird class_id: {class_id_bird}")
    
    

    
image = Image.open(file).convert('RGB')

model = RFDETRBase(resolution=slice_size, threshold=conf_threshold, device='cuda')
detections = model.predict(image)

# 1920*1080 divided by 728px tile = 2.63 * 1.48 = 6 tiles
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

detections = detections[np.isin(detections.class_id, [class_id_bird])]
labels = [
    f"bird {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
print("Saved output")
annotated_image.save(f'output_{Path(file).stem}.jpg')




def main():
    config = {
        "root": '/Datasets/CVAT/',          
        "annFile": 'clearml://{"name":"video classification bird-drone", "project":"BirdVideo-Classification", "file":"*train*.json"}',        
        "labels_path": "../TEC_DatasetTools/labels/Bird-Drone-Classification.csv",
        "use_cache": False,       
        "sort_as_video": True,
        "filter_labels": ['bird']
    }
    datamodule = LitDetectionDataModule(**config)
    datamodule.prepare_data()
    datamodule.setup(stage='train')
    
    dataset = datamodule.get_dataset('train')
    
    c_ = 0
    for item in dataset:
        # DEBUG: item = {'image': <PIL.Image.Image image mode=RGB size=3840x2160 at 0x774B8C131910>, 'targets': tensor([1]), 'class_labels': ['bird'], 'class_ids': tensor([1]), 'track_ids': ['db071400_0'], 'bboxs': tensor([[2487, 1115,   29,   21]], dtype=torch.int32), 'video_ids': ['1000']}

        print({'video_ids':item['video_ids'], 'track_ids':item['track_ids']})
        
        if c_ > 100:
            break
        c_ += 1
        
main()