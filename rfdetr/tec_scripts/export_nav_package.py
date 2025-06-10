from typing import Iterable
import argparse
import numpy as np
import torch
import os
from pathlib import Path
import model_navigator as nav
from model_navigator.configuration import Sample
from clearml import Task  # Import ClearML Task if you intend to use it


def get_dataloader(resolution):
    """Returns a random dataloader containing 8 batches of 3xresolutionxresolution tensors"""
    return [torch.randn((8, 3, resolution, resolution)).numpy() for _ in range(8)]


def get_verify_function():
    """Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner: Iterable[Sample], ys_expected: Iterable[Sample]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-2, atol=1.0e-2) for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def main(onnx_path: str, resolution: int = 728, out: str = 'exports/package.nav'):
    """
    Exports and optimizes an ONNX model using Model Navigator.

    Args:
        model_name (str): Name of the model.
        onnx_path (str): Path to the ONNX model file.
        resolution (int, optional): Input resolution for the dataloader. Defaults to 728.
        save_dir (str, optional): Directory to save the exported package. Defaults to 'exports'.
        clearml_project (str, optional): Name of the ClearML project. Defaults to None.
        clearml_task_name (str, optional): Name of the ClearML task. Defaults to None.
        upload (bool, optional): Whether to upload the model to ClearML if a task is initialized. Defaults to True.
    """
    model = Path(onnx_path)

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    package = nav.onnx.optimize(
        model=model,
        dataloader=get_dataloader(resolution),
        # verify_func=get_verify_function(),
        target_formats=(nav.Format.TENSORRT, nav.Format.ONNX),
        custom_configs=[
            nav.TensorRTConfig(precision=nav.TensorRTPrecision.FP16),
            nav.OnnxConfig(model_path=onnx_path, graph_surgeon_optimization=True),
        ],
        verbose=True,
    )
    nav.package.save(package, output_path, override=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export and optimize an ONNX model using Model Navigator.")
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("out", type=str, help="Path to the ONNX model file")
    parser.add_argument("--resolution", type=int, default=728, help="Input resolution for the dataloader")

    args = parser.parse_args()

    main(
        onnx_path=args.onnx_path,
        out=args.out,
        resolution=args.resolution,
    )