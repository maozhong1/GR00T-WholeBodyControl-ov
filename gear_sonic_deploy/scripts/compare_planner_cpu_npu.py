#!/usr/bin/env python3
"""
Compare planner_sonic.onnx inference output between CPU (FP32) and NPU (FP16).

Usage:
    python compare_planner_cpu_npu.py --line 88
    python compare_planner_cpu_npu.py --line 188 --csv planner_input_dump.csv
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import openvino as ov


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU planner inference")
    parser.add_argument("--line", type=int, default=88, help="CSV data line number (1-based, header is line 1)")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "planner_input_dump.csv"),
                        help="Path to planner_input_dump.csv")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "planner", "target_vel", "V2", "planner_sonic.onnx"),
                        help="Path to planner_sonic.onnx")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results")
    parser.add_argument("-q", "--quality", type=str, default="low", choices=["low", "high"],
                        help="NPU precision quality: 'low' (default FP16) or 'high' (higher precision for ReduceSum,Multiply)")
    return parser.parse_args()


def parse_csv_line(csv_path, line_num):
    """Parse a specific line from the planner input dump CSV."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # line 1 is header
        for i, row in enumerate(reader, start=2):
            if i == line_num:
                return header, row
    raise ValueError(f"Line {line_num} not found in CSV (file has fewer lines)")


def build_inputs_from_csv(header, row):
    """
    Build ONNX model input dict from CSV header+row.

    CSV columns (after infer_count, timestamp_us):
      mode, target_vel, target_height,
      movement_direction_0..2, facing_direction_0..2,
      random_seed, has_specific_target,
      specific_target_positions_0..11, specific_target_headings_0..3,
      allowed_pred_num_tokens_0..10,
      context_mujoco_qpos_0..143
    """
    # Skip first two metadata columns (infer_count, timestamp_us)
    values = row[2:]
    col_names = header[2:]

    # Parse all values as floats first, convert int types later
    vals = []
    for v in values:
        vals.append(float(v))

    idx = 0

    # mode: shape=[1], int64
    mode = np.array([int(vals[idx])], dtype=np.int64)
    idx += 1

    # target_vel: shape=[1], float32
    target_vel = np.array([vals[idx]], dtype=np.float32)
    idx += 1

    # target_height (mapped to 'height'): shape=[1], float32
    height = np.array([vals[idx]], dtype=np.float32)
    idx += 1

    # movement_direction: shape=[1,3], float32 (3 columns for 3 elements)
    movement_direction = np.array([[vals[idx], vals[idx+1], vals[idx+2]]], dtype=np.float32)
    idx += 3

    # facing_direction: shape=[1,3], float32
    facing_direction = np.array([[vals[idx], vals[idx+1], vals[idx+2]]], dtype=np.float32)
    idx += 3

    # random_seed: shape=[1], int64
    random_seed = np.array([int(vals[idx])], dtype=np.int64)
    idx += 1

    # has_specific_target: shape=[1,1], int64
    has_specific_target = np.array([[int(vals[idx])]], dtype=np.int64)
    idx += 1

    # specific_target_positions: shape=[1,4,3], float32
    # 12 columns (specific_target_positions_0..11), each is one element in 1x4x3
    stp_vals = vals[idx:idx+12]
    specific_target_positions = np.array(stp_vals, dtype=np.float32).reshape(1, 4, 3)
    idx += 12

    # specific_target_headings: shape=[1,4], float32
    sth_vals = vals[idx:idx+4]
    specific_target_headings = np.array(sth_vals, dtype=np.float32).reshape(1, 4)
    idx += 4

    # allowed_pred_num_tokens: shape=[1,11], int64
    apnt_vals = vals[idx:idx+11]
    allowed_pred_num_tokens = np.array([int(v) for v in apnt_vals], dtype=np.int64).reshape(1, 11)
    idx += 11

    # context_mujoco_qpos: shape=[1,4,36], float32
    # 144 columns (context_mujoco_qpos_0..143), representing 1x4x36
    qpos_vals = vals[idx:idx+144]
    context_mujoco_qpos = np.array(qpos_vals, dtype=np.float32).reshape(1, 4, 36)
    idx += 144

    inputs = {
        "mode": mode,
        "target_vel": target_vel,
        "height": height,
        "movement_direction": movement_direction,
        "facing_direction": facing_direction,
        "random_seed": random_seed,
        "has_specific_target": has_specific_target,
        "specific_target_positions": specific_target_positions,
        "specific_target_headings": specific_target_headings,
        "allowed_pred_num_tokens": allowed_pred_num_tokens,
        "context_mujoco_qpos": context_mujoco_qpos,
    }

    return inputs


def run_inference(model_path, inputs, device, npu_quality="low"):
    """Run inference using OpenVINO on the specified device."""
    core = ov.Core()
    model = core.read_model(model_path)

    # Force FP32 precision on CPU (default may use BF16 on AVX512/AMX-capable CPUs)
    config = {}
    if device == "CPU":
        config["INFERENCE_PRECISION_HINT"] = "f32"
    elif device == "NPU" and npu_quality == "high":
        config["NPU_COMPILATION_MODE_PARAMS"] = "compute-layers-with-higher-precision=ReduceSum,Multiply"

    compiled = core.compile_model(model, device, config)
    infer_request = compiled.create_infer_request()

    # Set inputs
    for name, data in inputs.items():
        input_tensor = ov.Tensor(data)
        infer_request.set_tensor(name, input_tensor)

    infer_request.infer()

    # Get outputs
    outputs = {}
    for output in compiled.outputs:
        name = output.get_any_name()
        outputs[name] = infer_request.get_tensor(output).data.copy()

    return outputs


def save_output(outputs, filepath):
    """Save inference outputs to a text file."""
    with open(filepath, "w") as f:
        for name, data in outputs.items():
            f.write(f"# {name} shape={data.shape} dtype={data.dtype}\n")
            flat = data.flatten()
            for i, v in enumerate(flat):
                f.write(f"{v}\n")
            f.write("\n")
    print(f"  Saved: {filepath}")


def plot_comparison(cpu_outputs, npu_outputs, output_path, line_num):
    """Plot CPU vs NPU outputs and their difference (3 subplots)."""
    # Use the main output: mujoco_qpos [1, 64, 36]
    cpu_data = cpu_outputs["mujoco_qpos"].flatten()
    npu_data = npu_outputs["mujoco_qpos"].flatten()
    delta = cpu_data - npu_data

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(cpu_data, color="tab:blue", linewidth=0.5)
    axes[0].set_title(f"CPU (FP32) - mujoco_qpos output (line {line_num})")
    axes[0].set_xlabel("index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(npu_data, color="tab:orange", linewidth=0.5)
    axes[1].set_title(f"NPU (FP16) - mujoco_qpos output (line {line_num})")
    axes[1].set_xlabel("index")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(delta, color="tab:green", linewidth=0.5)
    axes[2].set_title(f"Delta (CPU - NPU) | max={np.max(np.abs(delta)):.6f}, mean={np.mean(np.abs(delta)):.6f}")
    axes[2].set_xlabel("index")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Plot saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== Planner CPU vs NPU Comparison ===")
    print(f"  Model: {args.model}")
    print(f"  CSV:   {args.csv}")
    print(f"  Line:  {args.line}")
    print()

    # Parse input data
    header, row = parse_csv_line(args.csv, args.line)
    inputs = build_inputs_from_csv(header, row)

    print("Input shapes:")
    for name, data in inputs.items():
        print(f"  {name}: {data.shape} {data.dtype}")
    print()

    # Run CPU inference (FP32)
    print("Running CPU (FP32) inference...")
    cpu_outputs = run_inference(args.model, inputs, "CPU")
    cpu_file = os.path.join(args.output_dir, f"planner_cpu_out_{timestamp}.txt")
    save_output(cpu_outputs, cpu_file)

    # Run NPU inference (FP16)
    print(f"Running NPU (FP16, quality={args.quality}) inference...")
    npu_outputs = run_inference(args.model, inputs, "NPU", npu_quality=args.quality)
    npu_file = os.path.join(args.output_dir, f"planner_npu_{args.quality}_out_{timestamp}.txt")
    save_output(npu_outputs, npu_file)

    # Print summary statistics
    cpu_qpos = cpu_outputs["mujoco_qpos"].flatten()
    npu_qpos = npu_outputs["mujoco_qpos"].flatten()
    delta = cpu_qpos - npu_qpos
    print(f"\n=== Accuracy Summary (mujoco_qpos) ===")
    print(f"  Max absolute error:  {np.max(np.abs(delta)):.8f}")
    print(f"  Mean absolute error: {np.mean(np.abs(delta)):.8f}")
    print(f"  RMS error:           {np.sqrt(np.mean(delta**2)):.8f}")
    print(f"  CPU range: [{cpu_qpos.min():.4f}, {cpu_qpos.max():.4f}]")
    print(f"  NPU range: [{npu_qpos.min():.4f}, {npu_qpos.max():.4f}]")

    # Plot comparison
    plot_path = os.path.join(args.output_dir, f"planner_cpu_npu_diff_line{args.line}_{timestamp}.png")
    plot_comparison(cpu_outputs, npu_outputs, plot_path, args.line)

    print("\nDone!")


if __name__ == "__main__":
    main()
