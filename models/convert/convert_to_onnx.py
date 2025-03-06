import torch
import argparse
import onnx
from pathlib import Path
from models.yolo import Model  # Chỉnh lại đường dẫn model nếu cần

def load_pytorch_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = Model(cfg=checkpoint['model'].yaml)  
    model.load_state_dict(checkpoint['model'].state_dict())  
    model.eval()
    return model

def convert_to_onnx(ckpt_path, onnx_path, input_size=(3, 640, 640), opset_version=11):
    model = load_pytorch_model(ckpt_path)

    dummy_input = torch.randn(1, *input_size)  
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model converted successfully to {onnx_path}")


    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", type=str, required=True, help="Path to save ONNX model")
    args = parser.parse_args()

    Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
    convert_to_onnx(args.ckpt, args.onnx)
