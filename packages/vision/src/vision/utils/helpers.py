import torch
from pathlib import Path

def _export_onnx(model, sample_input, output_path: Path, input_names=("input",), output_names=("output",), dynamic_axes=None):
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes or {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )