#!/usr/bin/env python3
"""
Batch compare planner_sonic.onnx: CPU (FP32) vs NPU (FP16, high precision) for lines 1-200.

Usage:
    python compare_planner_batch.py
    python compare_planner_batch.py --csv planner_input_dump.csv --output cpu_npu_all_compare.csv
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import openvino as ov


def parse_args():
    parser = argparse.ArgumentParser(description="Batch compare CPU(FP32) vs NPU(FP16 high) planner inference")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "planner_input_dump.csv"),
                        help="Path to planner_input_dump.csv")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "planner", "target_vel", "V2", "planner_sonic.onnx"),
                        help="Path to planner_sonic.onnx")
    parser.add_argument("--output", type=str, default="cpu_npu_all_compare.csv",
                        help="Output CSV file")
    parser.add_argument("--start", type=int, default=2, help="Start line (2 = first data line after header)")
    parser.add_argument("--end", type=int, default=201, help="End line (exclusive)")
    return parser.parse_args()


def load_all_rows(csv_path):
    """Load all data rows from CSV, return header and list of rows."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def build_inputs_from_row(header, row):
    """Build model input dict from a CSV row."""
    values = [float(v) for v in row[2:]]
    idx = 0

    mode = np.array([int(values[idx])], dtype=np.int64); idx += 1
    target_vel = np.array([values[idx]], dtype=np.float32); idx += 1
    height = np.array([values[idx]], dtype=np.float32); idx += 1
    movement_direction = np.array([values[idx:idx+3]], dtype=np.float32); idx += 3
    facing_direction = np.array([values[idx:idx+3]], dtype=np.float32); idx += 3
    random_seed = np.array([int(values[idx])], dtype=np.int64); idx += 1
    has_specific_target = np.array([[int(values[idx])]], dtype=np.int64); idx += 1
    specific_target_positions = np.array(values[idx:idx+12], dtype=np.float32).reshape(1, 4, 3); idx += 12
    specific_target_headings = np.array(values[idx:idx+4], dtype=np.float32).reshape(1, 4); idx += 4
    allowed_pred_num_tokens = np.array([int(v) for v in values[idx:idx+11]], dtype=np.int64).reshape(1, 11); idx += 11
    context_mujoco_qpos = np.array(values[idx:idx+144], dtype=np.float32).reshape(1, 4, 36); idx += 144

    return {
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


def infer(compiled_model, inputs):
    """Run inference and return mujoco_qpos output."""
    infer_request = compiled_model.create_infer_request()
    for name, data in inputs.items():
        infer_request.set_tensor(name, ov.Tensor(data))
    infer_request.infer()
    return infer_request.get_tensor("mujoco_qpos").data.copy().flatten()


def main():
    args = parse_args()

    print(f"Model: {args.model}")
    print(f"CSV:   {args.csv}")
    print(f"Lines: {args.start} to {args.end - 1}")
    print()

    header, rows = load_all_rows(args.csv)
    total_data_lines = len(rows)

    # Compile models once
    core = ov.Core()
    model = core.read_model(args.model)

    print("Compiling CPU model (FP32)...")
    cpu_compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})

    print("Compiling NPU model (FP16, high precision)...")
    npu_compiled = core.compile_model(model, "NPU", {
        "NPU_COMPILATION_MODE_PARAMS": "compute-layers-with-higher-precision=ReduceSum,Multiply"
    })
    print()

    # Run batch comparison
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_line", "max_abs_error", "mean_abs_error", "rms_error",
                         "cpu_range_min", "cpu_range_max", "npu_range_min", "npu_range_max"])

        for line_num in range(args.start, min(args.end, total_data_lines + 2)):
            row_idx = line_num - 2  # rows list is 0-indexed, line 2 = rows[0]
            if row_idx < 0 or row_idx >= total_data_lines:
                break

            inputs = build_inputs_from_row(header, rows[row_idx])
            cpu_out = infer(cpu_compiled, inputs)
            npu_out = infer(npu_compiled, inputs)

            delta = cpu_out - npu_out
            max_abs = np.max(np.abs(delta))
            mean_abs = np.mean(np.abs(delta))
            rms = np.sqrt(np.mean(delta ** 2))

            writer.writerow([
                line_num, f"{max_abs:.8f}", f"{mean_abs:.8f}", f"{rms:.8f}",
                f"{cpu_out.min():.6f}", f"{cpu_out.max():.6f}",
                f"{npu_out.min():.6f}", f"{npu_out.max():.6f}"
            ])

            if (line_num - args.start + 1) % 20 == 0:
                print(f"  Processed {line_num - args.start + 1} lines...")

    print(f"\nDone! Results saved to: {args.output}")

    # Plot top 10 mean_abs_error inferences
    plot_top10(args.output)


def plot_top10(csv_path):
    """Plot top 10 inferences by mean_abs_error."""
    lines = []
    max_abs_errors = []
    mean_abs_errors = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(int(row["input_line"]))
            max_abs_errors.append(float(row["max_abs_error"]))
            mean_abs_errors.append(float(row["mean_abs_error"]))

    # Get top 10 by mean_abs_error
    indices = np.argsort(mean_abs_errors)[-10:][::-1]

    top_lines = [lines[i] for i in indices]
    top_max = [max_abs_errors[i] for i in indices]
    top_mean = [mean_abs_errors[i] for i in indices]

    x = np.arange(len(top_lines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, top_max, width, label="max_abs_error", color="tab:red")
    ax.bar(x + width/2, top_mean, width, label="mean_abs_error", color="tab:blue")

    ax.set_xlabel("Inference Number (input line)")
    ax.set_ylabel("Error")
    ax.set_title("Top 10 Inferences by Mean Absolute Error (CPU FP32 vs NPU FP16 High)")
    ax.set_xticks(x)
    ax.set_xticklabels(top_lines)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = csv_path.replace(".csv", "_top10.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
