# combine_test_sample

Single-process two-thread benchmark that runs **YOLOv12** (perception) and the
**Sonic decoder** concurrently on the Intel NPU. Use it to measure how the two
workloads coexist on the NPU at different tile splits, frame rates, and
performance hints.

- `perception` thread: paced to a target FPS (camera-like)
- `sonic` thread: runs flat-out, dominated by inference latency

Both models are compiled with the OpenVINO `THROUGHPUT` performance hint.
Each model gets its own NPU tile count via the `NPU_TILES` device property.

## Layout

```
combine_test_sample/
├── CMakeLists.txt   # build config
├── main.cpp         # two-thread benchmark
└── README.md
```

Default model paths (relative to repo root):

| Role          | Path                                                          |
|---------------|---------------------------------------------------------------|
| YOLOv12       | `./perception_sample/model/YoloV12/yolo12s.onnx`              |
| Sonic decoder | `./gear_sonic_deploy/policy/release/model_decoder.onnx`       |

## Prerequisites

- OpenVINO 2026 (`/opt/intel/openvino_2026`)
- OpenCV (used only to build a dummy YOLO input blob)
- CMake ≥ 3.16, C++17 toolchain
- Intel NPU driver if running on NPU

## Build

```bash
source /opt/intel/openvino_2026/setupvars.sh
cd combine_test_sample
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`CMakeLists.txt` autodetects `/opt/intel/openvino_2026` and falls back to
`/opt/intel/openvino_2016` if present. Override with `-DOpenVINO_DIR=...`.

## Run

Run from the **repo root** so the default relative model paths resolve:

```bash
# Defaults: NPU, yolo=3 tiles @ 30 FPS, decoder=1 tile, throughput hint, 60 s. Max tiles is 3.
./combine_test_sample/build/combine_test_sample

# Custom: yolo at 60 FPS using 3 tiles, decoder using 2 tiles, 90 s run
./combine_test_sample/build/combine_test_sample \
    --yolo-tiles 3 --yolo-fps 60 \
    --decoder-tiles 2 \
    --duration 90

# Standalone decoder benchmark (no perception load)
./combine_test_sample/build/combine_test_sample --decoder-only --decoder-tiles 1

# Standalone perception benchmark (no decoder load)
./combine_test_sample/build/combine_test_sample --yolo-only --yolo-tiles 3 --yolo-fps 60
```

Send `SIGINT` (Ctrl-C) at any time — both threads stop cleanly and the
final summary is printed.

### Options

| Option | Default | Description |
|---|---|---|
| `--yolo-model <path>`    | `./perception_sample/model/YoloV12/yolo12s.onnx`         | YOLO model (ONNX or OpenVINO IR `.xml`) |
| `--decoder-model <path>` | `./gear_sonic_deploy/policy/release/model_decoder.onnx`  | Sonic decoder model |
| `--device <name>`        | `NPU`                                                    | OpenVINO device (`NPU`, `CPU`, `GPU`, `AUTO:NPU,CPU`, …) |
| `--yolo-tiles <n>`       | `3`                                                      | NPU tiles for YOLO (applied only when device contains `NPU`) |
| `--decoder-tiles <n>`    | `1`                                                      | NPU tiles for decoder (applied only when device contains `NPU`) |
| `--yolo-fps <n>`         | `30`                                                     | Target frame rate for YOLO |
| `--duration <sec>`       | `60`                                                     | Run length in seconds (`0` = until Ctrl-C) |
| `--warmup <n>`           | `100`                                                    | First N iterations of each thread are excluded from FPS / min / max / avg |
| `--decoder-only`         | off                                                      | Run only the Sonic decoder thread |
| `--yolo-only`            | off                                                      | Run only the YOLO perception thread |
| `--help`                 | —                                                        | Print usage |

## Output

Each thread emits a one-line stats summary every second:

```
[perception] FPS=30.0 | infer=11.84 ms | frames=30
[sonic]      FPS=82.3 | avg=12.15 ms | min=11.41 ms | max=13.07 ms | frames=82
```

When the run ends (timeout or Ctrl-C), per-model totals are printed:

```
========== Final Results (60.00 s) ==========
[perception/yolo12s]
  frames        : 1800
  fps           : 30.00
  avg latency   : 11.92 ms
  min latency   : 11.30 ms
  max latency   : 18.45 ms
[sonic/model_decoder]
  frames        : 4892
  fps           : 81.53
  avg latency   : 12.21 ms
  min latency   : 11.40 ms
  max latency   : 19.83 ms
```

## Notes

- **Inputs are synthetic.** YOLO is fed a constant gray image; the decoder
  is fed random tensors sized from the model's static dims (dynamic dims
  are resolved to `1`). This is intentional: the goal is steady-state
  inference timing, not accuracy.
- **Tile config is applied only when the device string contains `NPU`.**
  On `--device CPU` or `--device GPU`, `--yolo-tiles` / `--decoder-tiles`
  are silently ignored.
- **Warmup iterations are still executed** (so caches and the NPU are
  warm), they just don't count toward the reported stats.
- **Throughput hint** lets OpenVINO pick its preferred number of inference
  requests internally. The driver thread still issues one synchronous
  `infer()` at a time, so the per-iteration timing reflects steady-state
  throughput-mode latency under contention.

## Comparing against standalone

Run each model in isolation first to establish a baseline, then run both
together to see how shared NPU usage affects each:

```bash
# baseline
./combine_test_sample/build/combine_test_sample --decoder-only --decoder-tiles 2
./combine_test_sample/build/combine_test_sample --yolo-only --yolo-tiles 3 --yolo-fps 60

# combined
./combine_test_sample/build/combine_test_sample --decoder-tiles 3 --yolo-tiles 3 --yolo-fps 60
```
