#!/usr/bin/env -S uv run --extra cpu
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fireredvad",
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxsim>=0.4.0",
#     "onnxruntime",
#     "huggingface_hub",
# ]
# ///
"""
Export FireRedVAD PyTorch models to ONNX format.

Downloads models from HuggingFace (FireRedTeam/FireRedVAD) if not already
present, then exports them to ONNX.

Exports:
  - fireredvad_vad.onnx: Non-streaming VAD
      Input: feat [batch, time, 80]  Output: probs [batch, time, 1]
  - fireredvad_aed.onnx: Non-streaming AED
      Input: feat [batch, time, 80]  Output: probs [batch, time, 3]
  - fireredvad_stream_vad.onnx: Streaming VAD (no cache input, caches as output)
      Input: feat [batch, time, 80]  Output: probs [batch, time, 1] + cache tensors
  - fireredvad_stream_vad_with_cache.onnx: Streaming VAD (with cache input/output)
      Input: feat [1, time, 80], caches_in [N, 1, P, L]
      Output: probs [1, time, 1], caches_out [N, 1, P, L]

Usage:
    # Export all models (downloads from HuggingFace automatically)
    python export_onnx.py --all

    # Export a specific model
    python export_onnx.py --task vad
    python export_onnx.py --task stream_vad
    python export_onnx.py --task aed

    # Use a local model directory instead of downloading
    python export_onnx.py --task vad --model-dir pretrained_models/FireRedVAD/VAD
"""

import argparse
import os
import shutil
import sys

import torch
import torch.nn as nn

from fireredvad.core.detect_model import DetectModel

HF_REPO = "FireRedTeam/FireRedVAD"
DEFAULT_MODEL_ROOT = "pretrained_models/FireRedVAD"

TASKS = {
    "vad": {"subdir": "VAD", "streaming": False},
    "stream_vad": {"subdir": "Stream-VAD", "streaming": True},
    "aed": {"subdir": "AED", "streaming": False},
}


class DetectModelNonStreaming(nn.Module):
    """Wrapper for non-streaming ONNX export (no cache inputs/outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat):
        probs, _ = self.model(feat, caches=None)
        return probs


class DetectModelStreamingNoCache(nn.Module):
    """Wrapper for streaming ONNX export without cache inputs.

    Caches are initialized internally (zeros) and returned as outputs.
    Used for the first frame of streaming inference.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat):
        probs, new_caches = self.model(feat, caches=None)
        return (probs, *new_caches)


class DetectModelStreamingWithCache(nn.Module):
    """Wrapper for streaming ONNX export with stacked cache input/output.

    Input:  feat [1, T, 80], caches_in [num_caches, 1, P, lookback_padding]
    Output: probs [1, T, odim], caches_out [num_caches, 1, P, lookback_padding]
    """

    def __init__(self, model, num_caches):
        super().__init__()
        self.model = model
        self.num_caches = num_caches

    def forward(self, feat, caches_in):
        cache_list = [caches_in[i] for i in range(self.num_caches)]
        probs, new_caches = self.model(feat, caches=cache_list)
        return probs, torch.stack(new_caches)


def download_models(model_root=DEFAULT_MODEL_ROOT):
    """Download FireRedVAD models from HuggingFace."""
    if os.path.isdir(model_root):
        missing = [t["subdir"] for t in TASKS.values()
                   if not os.path.isdir(os.path.join(model_root, t["subdir"]))]
        if not missing:
            print(f"Models already downloaded at: {model_root}")
            return model_root

    print(f"Downloading {HF_REPO} to {model_root} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO, local_dir=model_root)
    except ImportError:
        print("Error: huggingface_hub not found. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)

    print(f"Models downloaded to: {model_root}")
    return model_root


def get_num_fsmn_blocks(model):
    """Return the total number of FSMN blocks (1 initial + R-1 in fsmns)."""
    return 1 + len(model.dfsmn.fsmns)


def get_cache_shape(model):
    """Return the shape of a single cache tensor: [1, P, lookback_padding]."""
    fsmn = model.dfsmn.fsmn1
    P = fsmn.lookback_filter.in_channels
    lookback_padding = fsmn.lookback_padding
    return (1, P, lookback_padding)


def export_non_streaming(model, output_path, opset_version=18):
    """Export model in non-streaming mode (no caches)."""
    wrapper = DetectModelNonStreaming(model)
    wrapper.eval()

    dummy_feat = torch.randn(1, 100, 80)

    torch.onnx.export(
        wrapper,
        (dummy_feat,),
        output_path,
        input_names=["feat"],
        output_names=["probs"],
        dynamic_axes={
            "feat": {0: "batch", 1: "time"},
            "probs": {0: "batch", 1: "time"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"  Exported non-streaming model to: {output_path}")


def export_streaming_no_cache(model, output_path, opset_version=18):
    """Export streaming model without cache inputs (caches only as outputs)."""
    num_caches = get_num_fsmn_blocks(model)
    wrapper = DetectModelStreamingNoCache(model)
    wrapper.eval()

    dummy_feat = torch.randn(1, 100, 80)

    output_names = ["probs"] + [f"cache_out_{i}" for i in range(num_caches)]

    torch.onnx.export(
        wrapper,
        (dummy_feat,),
        output_path,
        input_names=["feat"],
        output_names=output_names,
        dynamic_axes={
            "feat": {0: "batch", 1: "time"},
            "probs": {0: "batch", 1: "time"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"  Exported streaming model (no cache input) to: {output_path}")


def export_streaming_with_cache(model, output_path, opset_version=18):
    """Export streaming model with stacked cache input/output."""
    num_caches = get_num_fsmn_blocks(model)
    cache_shape = get_cache_shape(model)

    wrapper = DetectModelStreamingWithCache(model, num_caches)
    wrapper.eval()

    dummy_feat = torch.randn(1, 1, 80)
    dummy_caches = torch.zeros(num_caches, *cache_shape)

    torch.onnx.export(
        wrapper,
        (dummy_feat, dummy_caches),
        output_path,
        input_names=["feat", "caches_in"],
        output_names=["probs", "caches_out"],
        dynamic_axes={
            "feat": {1: "time"},
            "probs": {1: "time"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"  Exported streaming model (with cache) to: {output_path}")
    print(f"  Cache shape: [{num_caches}, {cache_shape[0]}, {cache_shape[1]}, {cache_shape[2]}]")


def simplify_onnx(output_path):
    """Simplify the ONNX model using onnxsim."""
    try:
        import onnx
        import onnxsim
        model = onnx.load(output_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, output_path)
            print(f"  Simplified with onnxsim")
        else:
            print(f"  onnxsim simplification check failed, keeping original")
    except ImportError:
        print("  Skipping simplification (install onnxsim to enable)")
    except Exception as e:
        print(f"  onnxsim failed: {e}")


def verify_onnx(output_path):
    """Verify the exported ONNX model is valid."""
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"  ONNX model verified OK")
    except ImportError:
        print("  Skipping verification (install onnx package to enable)")
    except Exception as e:
        print(f"  ONNX verification failed: {e}")


def print_size(output_path):
    size = os.path.getsize(output_path)
    print(f"  File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")


def export_task(task_name, model_dir, output_dir, opset_version):
    """Export a single task (vad, stream_vad, or aed)."""
    task = TASKS[task_name]

    print(f"\n[{task_name}] Loading model from: {model_dir}")
    model = DetectModel.from_pretrained(model_dir)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    if task["streaming"]:
        # Export both streaming variants
        path1 = os.path.join(output_dir, f"fireredvad_{task_name}.onnx")
        export_streaming_no_cache(model, path1, opset_version)
        simplify_onnx(path1)
        print_size(path1)
        verify_onnx(path1)

        path2 = os.path.join(output_dir, f"fireredvad_{task_name}_with_cache.onnx")
        export_streaming_with_cache(model, path2, opset_version)
        simplify_onnx(path2)
        print_size(path2)
        verify_onnx(path2)
    else:
        path = os.path.join(output_dir, f"fireredvad_{task_name}.onnx")
        export_non_streaming(model, path, opset_version)
        simplify_onnx(path)
        print_size(path)
        verify_onnx(path)


def main():
    parser = argparse.ArgumentParser(
        description="Export FireRedVAD PyTorch models to ONNX format")
    parser.add_argument("--task", choices=list(TASKS.keys()),
                        help="Which model to export (vad, stream_vad, aed)")
    parser.add_argument("--all", action="store_true",
                        help="Export all models (vad, stream_vad, aed)")
    parser.add_argument("--model-dir", default=None,
                        help="Path to model directory (default: auto-download from HuggingFace)")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT,
                        help=f"Root directory for downloaded models (default: {DEFAULT_MODEL_ROOT})")
    parser.add_argument("--output-dir", default="onnx_models",
                        help="Output directory for ONNX files (default: onnx_models)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading models (use existing local files)")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Specify --task or --all")

    # Download models if needed
    if not args.skip_download and args.model_dir is None:
        download_models(args.model_root)

    tasks_to_export = list(TASKS.keys()) if args.all else [args.task]

    for task_name in tasks_to_export:
        task = TASKS[task_name]

        if args.model_dir:
            model_dir = args.model_dir
        else:
            model_dir = os.path.join(args.model_root, task["subdir"])

        export_task(task_name, model_dir, args.output_dir, args.opset)

    # Copy cmvn.ark to output directory (same file for all models)
    first_task = TASKS[tasks_to_export[0]]
    if args.model_dir:
        cmvn_src = os.path.join(args.model_dir, "cmvn.ark")
    else:
        cmvn_src = os.path.join(args.model_root, first_task["subdir"], "cmvn.ark")
    cmvn_dst = os.path.join(args.output_dir, "cmvn.ark")
    if os.path.isfile(cmvn_src):
        shutil.copy2(cmvn_src, cmvn_dst)
        print(f"\nCopied cmvn.ark to: {cmvn_dst}")
    else:
        print(f"\nWarning: cmvn.ark not found at {cmvn_src}")

    print("Done!")


if __name__ == "__main__":
    main()
