import supervision as sv
from rfdetr import RFDETRBase, RFDETRLarge
import onnx
import os
from pathlib import Path
import torch
from copy import deepcopy
from glob import glob

def export_to_onnx(
    model,
    resolution,
    device,
    dynamic_axes=None,
    output_dir="output",
    infer_dir=None,
    simplify=False,
    backbone_only=False,
    opset_version=17,
    verbose=True,
    force=False,
    shape=None,
    batch_size=1,
    **kwargs,
):
    """Export the trained model to ONNX format"""
    print(f"Exporting model to ONNX format")
    from rfdetr.deploy.export import export_onnx, onnx_simplify, make_infer_image

    model = model.model.model
    model = deepcopy(model.to("cpu"))
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    if shape is None:
        shape = (resolution, resolution)
    else:
        if shape[0] % 14 != 0 or shape[1] % 14 != 0:
            raise ValueError("Shape must be divisible by 14")

    input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
    input_names = ["input"]
    output_names = ["features"] if backbone_only else ["dets", "labels"]
    dynamic_axes = dynamic_axes
    model.eval()
    with torch.no_grad():
        if backbone_only:
            features = model(input_tensors)
            print(f"PyTorch inference output shape: {features.shape}")
        else:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            print(
                f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}"
            )
    model.cuda()
    input_tensors = input_tensors.cuda()

    # Export to ONNX
    output_file = export_onnx(
        output_dir=output_dir,
        model=model,
        input_names=input_names,
        input_tensors=input_tensors,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        backbone_only=backbone_only,
        verbose=verbose,
        opset_version=opset_version,
    )

    print(f"Successfully exported ONNX model to: {output_file}")

    if simplify:
        sim_output_file = onnx_simplify(
            onnx_dir=output_file,
            input_names=input_names,
            input_tensors=input_tensors,
            force=force,
        )
        print(f"Successfully simplified ONNX model to: {sim_output_file}")

    print("ONNX export completed successfully")
    model = model.to(device)



def export(model_name, model, resolution=728, dynamic_axes=True, device='cuda', save_dir='exports', slim=True, clearml_task=None, upload=True):
    _dynamic_axes = None
    if dynamic_axes:
        _dynamic_axes = {
            "input": {0: "batch_size"},  # variable length axes
        }
    
    os.makedirs(save_dir, exist_ok=True)
    output_dir = os.path.join(save_dir, f"export-{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    export_to_onnx(
        model=model,
        resolution=resolution,
        device=device,
        dynamic_axes= _dynamic_axes,
        simplify=False,
        output_dir=output_dir,
        opset_version=19,
    )
    if slim:
        print("Slimming ONNX model ...")
        print(f"exec: onnxslim --model-check {output_dir}/inference_model.onnx {output_dir}/inference_model.slim.onnx")
        os.system(f"onnxslim --model-check {output_dir}/inference_model.onnx {output_dir}/inference_model.slim.onnx")  
    
    if clearml_task and upload:
        for onnx_path in glob(os.path.join(output_dir, '*.onnx')):
            clearml_task.update_output_model(
                model_path=onnx_path, 
                name=f"onnx-rfdetr-{Path(onnx_path).stem}-{clearml_task.id}-FP32",
                model_name=f"onnx-rfdetr-{Path(onnx_path).stem}-{clearml_task.id}-FP32",
                comment=f"ONNX exported model from model id = {clearml_task.id} of type = rfdetr with precison FP32",
                auto_delete_file=False
                )
            print("Model exported sucessfully to ONNX format.")
       
        
    return f"{output_dir}/inference_model.slim.onnx"

if __name__ == '__main__':
    resolution = 728
    model = RFDETRBase(pretrain_weights='./logs/rfdetr_train_output-2025-03-29_13-43-10/checkpoint_best_ema.pth', resolution=resolution, device='cuda', num_queries=100)
    export(model, resolution)