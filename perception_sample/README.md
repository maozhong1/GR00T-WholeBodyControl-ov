# Perception Sample

OpenVINO-based perception benchmark app using YOLOv12 for object detection on Intel NPU/CPU/GPU. Designed to run in parallel with the Sonic encoder/decoder pipeline to validate concurrent NPU workloads.

## Prerequisites

- OpenVINO 2026 installed (default: `/opt/intel/openvino_2016/runtime/cmake`)
- OpenCV (bundled with OpenVINO or installed separately)
- CMake >= 3.16
- C++17 compiler

Source the OpenVINO environment before building:

```bash
source /opt/intel/openvino_2016/setupvars.sh
```

## Build

```bash
cd perception_sample
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

```bash
# Default: YOLOv12s on NPU, 3 tiles, 30 FPS
./build/perception_sample --model model/YoloV12/yolo12s.xml

# YOLOv12n (lighter model) on CPU
./build/perception_sample --model model/YoloV12/yolo12n.xml --device CPU

# Custom tile count and FPS target
./build/perception_sample --model model/YoloV12/yolo12s.xml --device NPU --tiles 6 --fps 60
```

### Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | `/home/maozhong/work/models/YoloV12/yolo12s.xml` | Path to OpenVINO IR model (.xml) |
| `--device <name>` | `NPU` | Inference device (`NPU`, `CPU`, `GPU`) |
| `--tiles <n>` | `3` | Number of NPU tiles |
| `--fps <n>` | `30` | Target inference rate (Hz) |
| `--help` | — | Print usage |

## Models

Pre-converted models are in `model/YoloV12/`:

| Model | Format | Notes |
|-------|--------|-------|
| `yolo12s.xml` / `.bin` | OpenVINO IR | Default, YOLOv12-small |
| `yolo12n.xml` / `.bin` | OpenVINO IR | YOLOv12-nano (faster, less accurate) |
| `yolo12s.onnx` | ONNX | Source model for IR conversion |
| `yolo12n.onnx` | ONNX | Source model for IR conversion |

## Output

The app prints real-time stats every second:

```
FPS: 30.1 | Inference time: 12.3 ms | Frames: 30
```

Currently uses a dummy input frame (gray image). Replace the frame capture in `main.cpp` with actual camera input for real workloads.
