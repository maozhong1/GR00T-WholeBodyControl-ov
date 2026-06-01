#!/usr/bin/env python3
"""
Binary search (bisect) to find which layer(s) in planner_sonic.onnx cause
NPU vs CPU accuracy divergence.

Strategy:
  1. List all layers in the model
  2. Binary search — add midpoint layer as output, compare CPU vs NPU
  3. Narrow down to the specific layer(s) causing divergence

Usage:
    # List all layers
    python bisect_planner_layers.py --list

    # Run binary search with default threshold
    python bisect_planner_layers.py --line 88

    # Custom threshold (relative mean error > 5% considered divergent)
    python bisect_planner_layers.py --line 88 --threshold 0.05

    # Compare a specific layer
    python bisect_planner_layers.py --line 88 --layer "layer_name"

    # Scan all layers sequentially (full report)
    python bisect_planner_layers.py --line 88 --scan-all
"""

import argparse
import csv
import os
import sys

import numpy as np
import openvino as ov


def parse_args():
    parser = argparse.ArgumentParser(description="Bisect planner model layers for NPU accuracy debug")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "planner", "target_vel", "V2", "planner_sonic.onnx"),
                        help="Path to planner_sonic.onnx")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".", "planner_input_dump.csv"),
                        help="Path to planner_input_dump.csv")
    parser.add_argument("--line", type=int, default=88, help="CSV data line to use as input")
    parser.add_argument("--list", action="store_true", help="List all layers and exit")
    parser.add_argument("--layer", type=str, default=None, help="Compare a specific layer by name")
    parser.add_argument("--scan-all", action="store_true", help="Scan all layers sequentially (full report)")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="Relative mean error threshold to flag divergence (default: 0.03 = 3%%)")
    parser.add_argument("--abs-threshold", type=float, default=0.09,
                        help="Absolute mean error threshold (default: 0.09)")
    parser.add_argument("-q", "--quality", type=str, default="high", choices=["low", "high"],
                        help="NPU precision quality: 'high' (default, higher precision for ReduceSum,Multiply) or 'low' (FP16)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV for scan-all results (default: layer_accuracy_report.csv)")
    return parser.parse_args()


def get_model_layers(model):
    """Get all operations that produce tensor outputs with valid tensor names.
    Skip Parameters, Constants, Results, and ops without tensor names (not addressable)."""
    skip_types = {"Parameter", "Constant", "Result"}
    layers = []
    for op in model.get_ordered_ops():
        type_name = op.get_type_name()
        if type_name in skip_types:
            continue
        if op.get_output_size() == 0:
            continue
        # Only include ops whose output port 0 has a tensor name (required for add_outputs)
        port = op.output(0)
        tensor_names = port.get_names()
        if not tensor_names:
            continue
        tensor_name = list(tensor_names)[0]
        layers.append({
            "name": op.get_friendly_name(),
            "type": type_name,
            "tensor_name": tensor_name,
            "output_shape": [op.get_output_shape(i) for i in range(op.get_output_size())],
        })
    return layers


def build_inputs_from_csv(csv_path, line_num):
    """Parse a specific line from planner_input_dump.csv into model inputs."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader, start=2):
            if i == line_num:
                break
        else:
            raise ValueError(f"Line {line_num} not found in CSV")

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


def compute_error_metrics(cpu_data, npu_data):
    """Compute error metrics between CPU and NPU outputs."""
    cpu_flat = cpu_data.flatten().astype(np.float64)
    npu_flat = npu_data.flatten().astype(np.float64)

    abs_diff = np.abs(cpu_flat - npu_flat)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    rms_error = np.sqrt(np.mean((cpu_flat - npu_flat) ** 2))

    # Relative mean error: normalize by the range of CPU output
    cpu_range = np.max(np.abs(cpu_flat))
    if cpu_range > 1e-8:
        relative_mean_error = mean_abs_error / cpu_range
    else:
        relative_mean_error = mean_abs_error  # fallback for near-zero outputs

    # Cosine similarity
    norm_cpu = np.linalg.norm(cpu_flat)
    norm_npu = np.linalg.norm(npu_flat)
    if norm_cpu > 1e-8 and norm_npu > 1e-8:
        cosine_sim = np.dot(cpu_flat, npu_flat) / (norm_cpu * norm_npu)
    else:
        cosine_sim = 1.0

    # Signal-to-noise ratio (SNR in dB)
    signal_power = np.mean(cpu_flat ** 2)
    noise_power = np.mean((cpu_flat - npu_flat) ** 2)
    if noise_power > 1e-12:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float("inf")

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rms_error": rms_error,
        "relative_mean_error": relative_mean_error,
        "cosine_similarity": cosine_sim,
        "snr_db": snr_db,
        "cpu_range": cpu_range,
    }


def is_divergent(metrics, threshold, abs_threshold):
    """
    Check if a layer output shows significant divergence.

    Criteria (ANY one failing means divergent):
      - mean_abs_error >= abs_threshold (default 0.09)
      - relative_mean_error >= threshold (default 3%)
      - cosine_similarity < 0.99
      - snr_db < 26 dB
    """
    if metrics["mean_abs_error"] >= abs_threshold:
        return True
    if metrics["relative_mean_error"] >= threshold:
        return True
    if metrics["cosine_similarity"] < 0.99:
        return True
    if metrics["snr_db"] < 26.0:
        return True
    return False


class CompiledModelPair:
    """CPU model compiled once with all outputs; NPU compiled per-layer on demand.

    CPU can handle all 5000+ outputs at once. NPU cannot, so we compile it
    per-layer. For binary search (~13 iterations) this is fast enough.
    CPU results are cached after one inference since the model doesn't change.
    """

    def __init__(self, model_path, layers, npu_quality="high"):
        self.model_path = model_path
        self.layers = layers
        self.core = ov.Core()

        self.npu_config = {}
        if npu_quality == "high":
            self.npu_config["NPU_COMPILATION_MODE_PARAMS"] = "compute-layers-with-higher-precision=ReduceSum,Multiply"

        # Compile CPU model once with all layer outputs
        all_tensor_names = [layer["tensor_name"] for layer in layers]
        print(f"Compiling CPU model (FP32) with {len(all_tensor_names)} layer outputs...")
        model_cpu = self.core.read_model(model_path)
        model_cpu.add_outputs(all_tensor_names)
        self.cpu_compiled = self.core.compile_model(model_cpu, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})

        # Build CPU output name lookup
        self.cpu_output_map = {}
        for output in self.cpu_compiled.outputs:
            for name in output.get_names():
                self.cpu_output_map[name] = output
            self.cpu_output_map[output.get_any_name()] = output

        # Cache CPU inference results (run once, reuse for all layer comparisons)
        self._cpu_results = None
        print("CPU compilation done.")
        print(f"NPU will be compiled per-layer on demand.\n")

    def _ensure_cpu_inferred(self, inputs):
        """Run CPU inference once and cache all layer outputs."""
        if self._cpu_results is not None:
            return
        print("  Running CPU inference (one-time)...")
        cpu_request = self.cpu_compiled.create_infer_request()
        for name, data in inputs.items():
            cpu_request.set_tensor(name, ov.Tensor(data))
        cpu_request.infer()

        self._cpu_results = {}
        for tensor_name, output_port in self.cpu_output_map.items():
            self._cpu_results[tensor_name] = cpu_request.get_tensor(output_port).data.copy()
        print("  CPU results cached.")

    def infer_and_compare(self, inputs, tensor_name):
        """Compare CPU (cached) vs NPU (compiled on-demand) for one layer."""
        # Get CPU result from cache
        self._ensure_cpu_inferred(inputs)
        if tensor_name not in self._cpu_results:
            raise ValueError(f"Tensor '{tensor_name}' not found in CPU outputs")
        cpu_data = self._cpu_results[tensor_name]

        # Compile NPU with just this one layer output
        model_npu = self.core.read_model(self.model_path)
        model_npu.add_outputs(tensor_name)
        npu_compiled = self.core.compile_model(model_npu, "NPU", self.npu_config)

        npu_request = npu_compiled.create_infer_request()
        for name, data in inputs.items():
            npu_request.set_tensor(name, ov.Tensor(data))
        npu_request.infer()

        # Find NPU output
        npu_data = None
        for output in npu_compiled.outputs:
            if tensor_name in output.get_names() or tensor_name == output.get_any_name():
                npu_data = npu_request.get_tensor(output).data.copy()
                break

        if npu_data is None:
            raise ValueError(f"Tensor '{tensor_name}' not found in NPU outputs")

        return compute_error_metrics(cpu_data, npu_data)


def compare_layer(model_pair, inputs, tensor_name):
    """Compare a single layer's output between CPU and NPU using pre-compiled models."""
    return model_pair.infer_and_compare(inputs, tensor_name)


def binary_search(model_pair, inputs, layers, threshold, abs_threshold):
    """Binary search to find the first layer where divergence appears."""
    print(f"\n{'='*70}")
    print(f"BINARY SEARCH - divergence criteria (ANY triggers):")
    print(f"  mean_abs_error >= {abs_threshold}")
    print(f"  relative_mean_error >= {threshold*100:.1f}%")
    print(f"  cosine_similarity < 0.99")
    print(f"  snr_db < 26 dB")
    print(f"{'='*70}\n")

    lo, hi = 0, len(layers) - 1
    divergent_layers = []

    # First check if the final output is even divergent
    print(f"Checking final layer [{hi}] {layers[hi]['name']} ({layers[hi]['type']})...")
    metrics = compare_layer(model_pair, inputs, layers[hi]["tensor_name"])
    print_metrics(metrics, layers[hi])

    if not is_divergent(metrics, threshold, abs_threshold):
        print("\n✓ Final output is within threshold — no divergence detected!")
        return []

    # Check first layer
    print(f"\nChecking first layer [{lo}] {layers[lo]['name']} ({layers[lo]['type']})...")
    metrics = compare_layer(model_pair, inputs, layers[lo]["tensor_name"])
    print_metrics(metrics, layers[lo])

    if is_divergent(metrics, threshold, abs_threshold):
        print(f"\n✗ First layer already divergent!")
        divergent_layers.append((lo, layers[lo], metrics))
        return divergent_layers

    # Binary search
    print(f"\nSearching between layer {lo} and {hi}...")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        layer = layers[mid]
        print(f"\n  Checking [{mid}] {layer['name']} ({layer['type']})...")

        try:
            metrics = compare_layer(model_pair, inputs, layer["tensor_name"])
            print_metrics(metrics, layer, indent=4)

            if is_divergent(metrics, threshold, abs_threshold):
                hi = mid
                print(f"    → DIVERGENT, searching [{lo}..{mid}]")
            else:
                lo = mid
                print(f"    → OK, searching [{mid}..{hi}]")
        except Exception as e:
            print(f"    → SKIP (error: {e})")
            if mid - lo > hi - mid:
                hi = mid
            else:
                lo = mid

    # Found the transition point
    print(f"\n{'='*70}")
    print(f"RESULT: Divergence starts at layer [{hi}]")
    print(f"  Name: {layers[hi]['name']}")
    print(f"  Type: {layers[hi]['type']}")
    print(f"  Last OK layer [{lo}]: {layers[lo]['name']} ({layers[lo]['type']})")
    print(f"{'='*70}")

    # Also check a few layers around the transition
    print(f"\nChecking neighbors around transition point...")
    for i in range(max(0, hi-2), min(len(layers), hi+3)):
        layer = layers[i]
        try:
            metrics = compare_layer(model_pair, inputs, layer["tensor_name"])
            marker = "✗" if is_divergent(metrics, threshold, abs_threshold) else "✓"
            print(f"  [{i}] {marker} {layer['name']} ({layer['type']}): "
                  f"rel_err={metrics['relative_mean_error']:.6f}, "
                  f"cos_sim={metrics['cosine_similarity']:.8f}")
            if is_divergent(metrics, threshold, abs_threshold):
                divergent_layers.append((i, layer, metrics))
        except Exception as e:
            print(f"  [{i}] ? {layer['name']} — error: {e}")

    return divergent_layers


def scan_all_layers(model_pair, inputs, layers, threshold, abs_threshold, output_csv):
    """Scan all layers and report errors."""
    print(f"\n{'='*70}")
    print(f"FULL LAYER SCAN - {len(layers)} layers")
    print(f"{'='*70}\n")

    results = []
    for i, layer in enumerate(layers):
        try:
            metrics = compare_layer(model_pair, inputs, layer["tensor_name"])
            divergent = is_divergent(metrics, threshold, abs_threshold)
            marker = "✗" if divergent else "✓"
            print(f"  [{i:4d}] {marker} {layer['type']:20s} | rel_err={metrics['relative_mean_error']:.6f} | "
                  f"cos={metrics['cosine_similarity']:.6f} | {layer['name']}")
            results.append({
                "index": i,
                "name": layer["name"],
                "type": layer["type"],
                "divergent": divergent,
                **metrics,
            })
        except Exception as e:
            print(f"  [{i:4d}] ? {layer['type']:20s} | error: {e} | {layer['name']}")
            results.append({
                "index": i,
                "name": layer["name"],
                "type": layer["type"],
                "divergent": None,
                "error": str(e),
            })

        if (i + 1) % 50 == 0:
            print(f"  --- Progress: {i+1}/{len(layers)} ---")

    # Save results to CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "layer_name", "layer_type", "divergent",
                             "max_abs_error", "mean_abs_error", "rms_error",
                             "relative_mean_error", "cosine_similarity", "snr_db", "cpu_range"])
            for r in results:
                if "error" in r:
                    writer.writerow([r["index"], r["name"], r["type"], "ERROR",
                                     "", "", "", "", "", "", ""])
                else:
                    writer.writerow([r["index"], r["name"], r["type"], r["divergent"],
                                     f"{r['max_abs_error']:.8f}",
                                     f"{r['mean_abs_error']:.8f}",
                                     f"{r['rms_error']:.8f}",
                                     f"{r['relative_mean_error']:.8f}",
                                     f"{r['cosine_similarity']:.10f}",
                                     f"{r['snr_db']:.2f}",
                                     f"{r['cpu_range']:.6f}"])
        print(f"\nResults saved to: {output_csv}")

    # Summary
    divergent_count = sum(1 for r in results if r.get("divergent") is True)
    print(f"\nSummary: {divergent_count}/{len(results)} layers divergent (threshold={threshold*100:.1f}%)")

    if divergent_count > 0:
        print("\nTop 10 most divergent layers:")
        sorted_results = sorted(
            [r for r in results if r.get("divergent") is True],
            key=lambda r: r.get("relative_mean_error", 0),
            reverse=True
        )
        for r in sorted_results[:10]:
            print(f"  [{r['index']:4d}] {r['type']:20s} | rel_err={r['relative_mean_error']:.6f} | "
                  f"cos={r['cosine_similarity']:.8f} | {r['name']}")

    return results


def print_metrics(metrics, layer, indent=2):
    """Print error metrics for a layer."""
    pad = " " * indent
    print(f"{pad}max_abs_error:      {metrics['max_abs_error']:.8f}")
    print(f"{pad}mean_abs_error:     {metrics['mean_abs_error']:.8f}")
    print(f"{pad}relative_mean_err:  {metrics['relative_mean_error']:.6f} ({metrics['relative_mean_error']*100:.2f}%)")
    print(f"{pad}cosine_similarity:  {metrics['cosine_similarity']:.10f}")
    print(f"{pad}snr_db:             {metrics['snr_db']:.2f}")
    print(f"{pad}cpu_range:          {metrics['cpu_range']:.6f}")


def main():
    args = parse_args()

    core = ov.Core()
    model = core.read_model(args.model)
    layers = get_model_layers(model)

    # --list: just print layers and exit
    if args.list:
        print(f"Model: {args.model}")
        print(f"Total addressable layers: {len(layers)}\n")
        print(f"{'Idx':>5} | {'Type':<25} | {'Output Shape':<30} | {'Name':<40} | Tensor Name")
        print("-" * 140)
        for i, layer in enumerate(layers):
            shapes_str = str(layer["output_shape"])
            print(f"{i:5d} | {layer['type']:<25} | {shapes_str:<30} | {layer['name']:<40} | {layer['tensor_name']}")
        return

    # Load input data
    print(f"Model:     {args.model}")
    print(f"CSV:       {args.csv}")
    print(f"Line:      {args.line}")
    print(f"NPU mode:  {args.quality}")
    print(f"Threshold: {args.threshold*100:.1f}% relative mean error")
    print(f"Total layers: {len(layers)}")

    inputs = build_inputs_from_csv(args.csv, args.line)

    # Compile models once with all layer outputs
    model_pair = CompiledModelPair(args.model, layers, npu_quality=args.quality)

    if args.layer:
        # Compare a specific layer (accept either friendly name or tensor name)
        layer_info = next((l for l in layers if l["name"] == args.layer or l["tensor_name"] == args.layer), None)
        if layer_info is None:
            print(f"ERROR: Layer '{args.layer}' not found. Use --list to see available layers.")
            return
        print(f"\nComparing layer: {layer_info['name']} (tensor: {layer_info['tensor_name']})")
        metrics = compare_layer(model_pair, inputs, layer_info["tensor_name"])
        print_metrics(metrics, layer_info)
        divergent = is_divergent(metrics, args.threshold, args.abs_threshold)
        print(f"\n  Divergent: {'YES ✗' if divergent else 'NO ✓'}")

    elif args.scan_all:
        # Full scan
        output_csv = args.output or "layer_accuracy_report.csv"
        scan_all_layers(model_pair, inputs, layers, args.threshold, args.abs_threshold, output_csv)

    else:
        # Binary search (default)
        binary_search(model_pair, inputs, layers, args.threshold, args.abs_threshold)


if __name__ == "__main__":
    main()
