#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

struct Config {
    std::string model_path = "/home/maozhong/work/models/YoloV12/yolo12s.xml";
    std::string device = "NPU";
    int num_tiles = 3;
    int target_fps = 30;
};

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            cfg.device = argv[++i];
        } else if (arg == "--tiles" && i + 1 < argc) {
            cfg.num_tiles = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            cfg.target_fps = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: perception_sample [options]\n"
                      << "  --model <path>   Model path (default: " << cfg.model_path << ")\n"
                      << "  --device <name>  Device name (default: NPU)\n"
                      << "  --tiles <n>      Number of tiles (default: 3)\n"
                      << "  --fps <n>        Target FPS (default: 30)\n";
            std::exit(0);
        }
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "Configuration:\n"
              << "  Model:  " << cfg.model_path << "\n"
              << "  Device: " << cfg.device << "\n"
              << "  Tiles:  " << cfg.num_tiles << "\n"
              << "  FPS:    " << cfg.target_fps << "\n\n";

    const auto frame_duration = std::chrono::microseconds(1'000'000 / cfg.target_fps);

    // Initialize OpenVINO
    ov::Core core;

    // Read model
    std::cout << "Loading model: " << cfg.model_path << "\n";
    auto model = core.read_model(cfg.model_path);

    // Configure NPU with latency mode
    ov::AnyMap device_config;
    device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    device_config[ov::hint::num_requests.name()] = 1;

    // Set number of tiles for NPU
    device_config["NPU_TILES"] = std::to_string(cfg.num_tiles);

    // Compile model
    std::cout << "Compiling model on " << cfg.device << " (latency mode, "
              << cfg.num_tiles << " tiles)...\n";
    auto compiled_model = core.compile_model(model, cfg.device, device_config);

    // Create infer request
    auto infer_request = compiled_model.create_infer_request();

    // Get input tensor info
    auto input_port = compiled_model.input(0);
    auto input_shape = input_port.get_shape();
    std::cout << "Input shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i + 1 < input_shape.size() ? ", " : "");
    }
    std::cout << "]\n\n";

    // Determine input dimensions (assume NCHW layout)
    int input_h = static_cast<int>(input_shape[2]);
    int input_w = static_cast<int>(input_shape[3]);

    // Inference loop
    std::cout << "Starting inference loop at " << cfg.target_fps << " FPS...\n";
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();

        // Create a dummy input frame (replace with actual camera capture)
        cv::Mat frame(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));

        // Preprocess: convert to blob format
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_w, input_h),
                               cv::Scalar(), true, false, CV_32F);

        // Set input tensor
        auto input_tensor = infer_request.get_input_tensor();
        std::memcpy(input_tensor.data<float>(), blob.ptr<float>(),
                    input_tensor.get_byte_size());

        // Run inference
        infer_request.infer();

        // Get output
        auto output_tensor = infer_request.get_output_tensor();
        frame_count++;

        // Print stats every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        if (elapsed.count() >= 1000) {
            double actual_fps = frame_count * 1000.0 / elapsed.count();
            auto infer_time = std::chrono::duration_cast<std::chrono::microseconds>(now - frame_start);
            std::cout << "FPS: " << actual_fps
                      << " | Inference time: " << infer_time.count() / 1000.0 << " ms"
                      << " | Frames: " << frame_count << "\n";
            frame_count = 0;
            start_time = now;
        }

        // Pace to target FPS
        auto frame_end = std::chrono::steady_clock::now();
        auto processing_time = frame_end - frame_start;
        if (processing_time < frame_duration) {
            std::this_thread::sleep_for(frame_duration - processing_time);
        }
    }

    return 0;
}
