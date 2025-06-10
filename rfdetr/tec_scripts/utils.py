from PIL import Image
from clearml import Task
import warnings
import numpy as np
import numbers 

def pil_resize_img(img, base_width, bboxs_xywh=None):
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.Resampling.BICUBIC)
    
    if bboxs_xywh:
        bboxs_xywh = bboxs_xywh * wpercent
        return img, bboxs_xywh
    return img



def clearml_log_fit_history(data):
    task = Task.current_task()
    if task is None:
        warnings.warn("|WARNING| Logging train history to ClearML is disabled. No current Task active.")
        return 
    logger = task.get_logger()
    for metric in ['train_loss', 'test_loss']:
        if metric in data: 
            logger.report_scalar(
                title='Train History', series=str(metric), value=data[metric], iteration=data['epoch']
            )
    
    data['avg_precision'] = data['test_coco_eval_bbox'][0]
    data['avg_recall'] = data['test_coco_eval_bbox'][6]
   
    for metric in ['avg_precision', 'avg_recall']:
        if metric in data: 
            logger.report_scalar(
                title='AP metric @ [ IoU=0.50:0.95 | area = all ] ', series=str(metric), value=data[metric], iteration=data['epoch']
            )
            
    iou_metrics = [
        "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", 
        "AP @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "AP @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    ]
    
    for iou_metric, iou_value in zip(iou_metrics, data['test_coco_eval_bbox']):
        if metric in data: 
            logger.report_scalar(
                title='IoU Metrics: AP - AR', series=str(iou_metric), value=iou_value, iteration=data['epoch']
            )


    for metric in ['n_parameters']:
        if metric in data: 
            logger.report_scalar(
                title='Model parameters', series=str(metric), value=data[metric], iteration=data['epoch']
            )
            
    for metric in ['epoch']:
        if metric in data: 
            logger.report_scalar(
                title='Epochs', series=str(metric), value=data[metric], iteration=data['epoch']
            )
            
    for prefix_metric in ['Train', 'Test', 'EMA']:
        for metric in data.keys(): 
            if metric.startswith(prefix_metric.lower()) and isinstance(data[metric], numbers.Number):
                logger.report_scalar(
                    title=prefix_metric, series=str(metric), value=data[metric], iteration=data['epoch']
                )
                
    
    print("Logging data:", data)
    
    