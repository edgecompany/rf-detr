import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from coco_classes import TEC_COCO_CATEGORIES, COCO_CATEGORIES

model = RFDETRMedium(
    resolution= 728,
    device= "cuda",
    num_classes=len(COCO_CLASSES)
    # num_classes= len(TEC_COCO_CATEGORIES)-1,
    # pretrain_weights='logs/rfdetr_train_output-2025-06-18_21-33-22/checkpoint_best_regular.pth',
)

model.optimize_for_inference()

img_source = "./test_images/cars-people-walking_0.jpg"

image = Image.open(img_source)
detections = model.predict(image, threshold=0.1)

labels = [
    f"{class_id} {TEC_COCO_CATEGORIES[class_id]['name']} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

annotated_image.save('out_rfdetr-medium.jpg')