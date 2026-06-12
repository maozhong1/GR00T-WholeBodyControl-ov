// combine_test_sample
//
// Runs YOLOv12 (perception) and Sonic decoder (sonic) on Intel NPU in two
// dedicated threads inside a single process so we can measure how well the
// two models share NPU resources.
//
//   - perception thread: runs yolo12s.onnx paced at a target FPS
//   - sonic thread:      runs model_decoder.onnx flat-out with LATENCY hint
//
// Each model is configured with its own NPU tile count.  At exit we print
// FPS / min / max / avg latency for the decoder, plus FPS / avg inference
// time for YOLO.

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
    std::string yolo_model =
        "./perception_sample/model/YoloV12/yolo12s.onnx";
    std::string decoder_model =
        "./gear_sonic_deploy/policy/release/model_decoder.onnx";
    std::string device = "NPU";
    int yolo_tiles = 3;
    int decoder_tiles = 1;
    int yolo_fps = 30;
    int duration_sec = 60;  // 0 = run until Ctrl-C
    int warmup_iters = 100;  // skipped from min/max/avg stats
    bool decoder_only = false;
    bool yolo_only = false;
    std::string decoder_priority = "NORMAL";  // HIGH | NORMAL | LOW
};

// Map priority string to ov::hint::Priority. NORMAL → MEDIUM (the OV default).
ov::hint::Priority parse_priority(const std::string& s) {
    if (s == "HIGH") return ov::hint::Priority::HIGH;
    if (s == "LOW")  return ov::hint::Priority::LOW;
    return ov::hint::Priority::MEDIUM;
}

void print_usage(const Config& cfg) {
    std::cout
        << "Usage: combine_test_sample [options]\n"
        << "  --yolo-model <path>      YOLOv12 model (default: " << cfg.yolo_model << ")\n"
        << "  --decoder-model <path>   Sonic decoder model (default: " << cfg.decoder_model << ")\n"
        << "  --device <name>          OpenVINO device (default: " << cfg.device << ")\n"
        << "  --yolo-tiles <n>         NPU tiles for YOLO (default: " << cfg.yolo_tiles << ")\n"
        << "  --decoder-tiles <n>      NPU tiles for decoder (default: " << cfg.decoder_tiles << ")\n"
        << "  --yolo-fps <n>           Target FPS for YOLO (default: " << cfg.yolo_fps << ")\n"
        << "  --duration <sec>         Run duration in seconds, 0 = forever (default: "
        << cfg.duration_sec << ")\n"
        << "  --warmup <n>             Warmup iterations excluded from stats (default: "
        << cfg.warmup_iters << ")\n"
        << "  --decoder-only           Run only the Sonic decoder thread\n"
        << "  --yolo-only              Run only the YOLO perception thread\n"
        << "  --decoder-priority <P>   Decoder model priority: HIGH|NORMAL|LOW (default: "
        << cfg.decoder_priority << ")\n"
        << "  --help                   Show this help\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--yolo-model")           cfg.yolo_model = next("--yolo-model");
        else if (a == "--decoder-model")   cfg.decoder_model = next("--decoder-model");
        else if (a == "--device")          cfg.device = next("--device");
        else if (a == "--yolo-tiles")      cfg.yolo_tiles = std::stoi(next("--yolo-tiles"));
        else if (a == "--decoder-tiles")   cfg.decoder_tiles = std::stoi(next("--decoder-tiles"));
        else if (a == "--yolo-fps")        cfg.yolo_fps = std::stoi(next("--yolo-fps"));
        else if (a == "--duration")        cfg.duration_sec = std::stoi(next("--duration"));
        else if (a == "--warmup")          cfg.warmup_iters = std::stoi(next("--warmup"));
        else if (a == "--decoder-only")    cfg.decoder_only = true;
        else if (a == "--yolo-only")       cfg.yolo_only = true;
        else if (a == "--decoder-priority") {
            cfg.decoder_priority = next("--decoder-priority");
            if (cfg.decoder_priority != "HIGH" && cfg.decoder_priority != "NORMAL" &&
                cfg.decoder_priority != "LOW") {
                std::cerr << "Invalid --decoder-priority: " << cfg.decoder_priority
                          << " (expected HIGH, NORMAL, or LOW)\n";
                std::exit(1);
            }
        }
        else if (a == "--help")            { print_usage(cfg); std::exit(0); }
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            print_usage(cfg);
            std::exit(1);
        }
    }
    return cfg;
}

std::atomic<bool> g_stop{false};
void sigint_handler(int) { g_stop = true; }

// Resolve a possibly-symbolic dimension to a concrete one.  Sonic decoder ONNX
// exports often have batch (or some sequence dim) as dynamic; treat dynamic
// dims as 1 so we can allocate a real tensor for benchmarking.
ov::Shape resolve_shape(const ov::PartialShape& pshape) {
    ov::Shape shape;
    shape.reserve(pshape.size());
    for (const auto& d : pshape) {
        shape.push_back(d.is_static() ? static_cast<size_t>(d.get_length()) : 1u);
    }
    return shape;
}

void fill_random(ov::Tensor& tensor, std::mt19937& rng) {
    const auto et = tensor.get_element_type();
    const size_t n = tensor.get_size();
    if (et == ov::element::f32) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto* p = tensor.data<float>();
        for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
    } else if (et == ov::element::f16) {
        // Write zero-pattern (half 0.0); fine for latency benchmarking.
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    } else if (et == ov::element::i64) {
        std::uniform_int_distribution<int64_t> dist(0, 1);
        auto* p = tensor.data<int64_t>();
        for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
    } else if (et == ov::element::i32) {
        std::uniform_int_distribution<int32_t> dist(0, 1);
        auto* p = tensor.data<int32_t>();
        for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
    } else if (et == ov::element::boolean || et == ov::element::u8 || et == ov::element::i8) {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    } else {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }
}

std::string shape_to_string(const ov::Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size()) out += ",";
    }
    out += "]";
    return out;
}

// ---------------------------------------------------------------------------
// Perception thread (YOLO)
// ---------------------------------------------------------------------------
struct YoloStats {
    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> total_infer_us{0};
    std::atomic<uint64_t> max_us{0};
    std::atomic<uint64_t> min_us{UINT64_MAX};
};

void yolo_thread_fn(ov::Core& core, const Config& cfg, YoloStats& stats) {
    try {
        std::cout << "[perception] loading " << cfg.yolo_model << "\n";
        auto model = core.read_model(cfg.yolo_model);

        ov::AnyMap dev_cfg;
        dev_cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        if (cfg.device.find("NPU") != std::string::npos) {
            dev_cfg["NPU_TILES"] = std::to_string(cfg.yolo_tiles);
        }
        std::cout << "[perception] compiling on " << cfg.device
                  << " (throughput, " << cfg.yolo_tiles << " tiles)\n";
        auto compiled = core.compile_model(model, cfg.device, dev_cfg);
        auto req = compiled.create_infer_request();

        auto in_port = compiled.input(0);
        auto in_shape = resolve_shape(in_port.get_partial_shape());
        std::cout << "[perception] input shape: " << shape_to_string(in_shape) << "\n";

        const int input_h = static_cast<int>(in_shape[2]);
        const int input_w = static_cast<int>(in_shape[3]);
        cv::Mat frame(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(input_w, input_h),
                               cv::Scalar(), true, false, CV_32F);
        auto in_tensor = req.get_input_tensor();
        std::memcpy(in_tensor.data<float>(), blob.ptr<float>(), in_tensor.get_byte_size());

        const auto frame_duration =
            std::chrono::microseconds(1'000'000 / std::max(1, cfg.yolo_fps));

        std::cout << "[perception] target " << cfg.yolo_fps << " FPS"
                  << ", warmup=" << cfg.warmup_iters << " iters\n";
        auto window_start = std::chrono::steady_clock::now();
        uint64_t window_frames = 0;
        uint64_t window_infer_us = 0;
        uint64_t iter_idx = 0;
        bool warmup_done_logged = false;

        while (!g_stop.load()) {
            auto loop_start = std::chrono::steady_clock::now();

            auto t0 = std::chrono::steady_clock::now();
            req.infer();
            auto t1 = std::chrono::steady_clock::now();

            auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            ++iter_idx;
            const bool counts = iter_idx > static_cast<uint64_t>(cfg.warmup_iters);
            if (counts && !warmup_done_logged) {
                std::cout << "[perception] warmup complete, recording stats\n";
                warmup_done_logged = true;
            }
            if (counts) {
                stats.frames.fetch_add(1, std::memory_order_relaxed);
                stats.total_infer_us.fetch_add(infer_us, std::memory_order_relaxed);
                uint64_t cur_max = stats.max_us.load(std::memory_order_relaxed);
                while (static_cast<uint64_t>(infer_us) > cur_max &&
                       !stats.max_us.compare_exchange_weak(cur_max, infer_us)) {}
                uint64_t cur_min = stats.min_us.load(std::memory_order_relaxed);
                while (static_cast<uint64_t>(infer_us) < cur_min &&
                       !stats.min_us.compare_exchange_weak(cur_min, infer_us)) {}

                ++window_frames;
                window_infer_us += infer_us;
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now - window_start).count();
            if (elapsed_ms >= 5000) {
                double fps = window_frames * 1000.0 / static_cast<double>(elapsed_ms);
                double avg_ms = window_frames ? (window_infer_us / 1000.0) / window_frames : 0.0;
                std::cout << "[perception] FPS=" << std::fixed << std::setprecision(1) << fps
                          << " | infer=" << std::setprecision(2) << avg_ms << " ms"
                          << " | frames=" << window_frames << "\n";
                window_start = now;
                window_frames = 0;
                window_infer_us = 0;
            }

            auto processed = std::chrono::steady_clock::now() - loop_start;
            if (processed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - processed);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[perception] FATAL: " << e.what() << "\n";
        g_stop = true;
    }
}

// ---------------------------------------------------------------------------
// Sonic thread (decoder)
// ---------------------------------------------------------------------------
struct DecoderStats {
    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> total_us{0};
    std::atomic<uint64_t> max_us{0};
    std::atomic<uint64_t> min_us{UINT64_MAX};
};

void decoder_thread_fn(ov::Core& core, const Config& cfg, DecoderStats& stats) {
    try {
        std::cout << "[sonic] loading " << cfg.decoder_model << "\n";
        auto model = core.read_model(cfg.decoder_model);

        ov::AnyMap dev_cfg;
        dev_cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        if (cfg.device.find("NPU") != std::string::npos) {
            dev_cfg["NPU_TILES"] = std::to_string(cfg.decoder_tiles);
            dev_cfg[ov::hint::model_priority.name()] = parse_priority(cfg.decoder_priority);
        }
        std::cout << "[sonic] compiling on " << cfg.device
                  << " (throughput hint, " << cfg.decoder_tiles << " tiles, priority "
                  << cfg.decoder_priority << ")\n";
        auto compiled = core.compile_model(model, cfg.device, dev_cfg);
        auto req = compiled.create_infer_request();

        std::mt19937 rng(0xC0FFEE);
        for (size_t i = 0; i < compiled.inputs().size(); ++i) {
            auto port = compiled.input(i);
            auto pshape = port.get_partial_shape();
            auto shape = resolve_shape(pshape);
            ov::Tensor t(port.get_element_type(), shape);
            fill_random(t, rng);
            req.set_input_tensor(i, t);
            std::cout << "[sonic] input[" << i << "] " << port.get_any_name()
                      << " " << port.get_element_type().get_type_name()
                      << " " << shape_to_string(shape) << "\n";
        }

        std::cout << "[sonic] starting latency benchmark, warmup="
                  << cfg.warmup_iters << " iters\n";
        auto window_start = std::chrono::steady_clock::now();
        uint64_t window_frames = 0;
        uint64_t window_us = 0;
        uint64_t window_max = 0;
        uint64_t window_min = UINT64_MAX;
        uint64_t iter_idx = 0;
        bool warmup_done_logged = false;

        while (!g_stop.load()) {
            auto t0 = std::chrono::steady_clock::now();
            req.infer();
            auto t1 = std::chrono::steady_clock::now();

            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            ++iter_idx;
            const bool counts = iter_idx > static_cast<uint64_t>(cfg.warmup_iters);
            if (counts && !warmup_done_logged) {
                std::cout << "[sonic] warmup complete, recording stats\n";
                warmup_done_logged = true;
            }
            if (!counts) continue;
            stats.frames.fetch_add(1, std::memory_order_relaxed);
            stats.total_us.fetch_add(us, std::memory_order_relaxed);
            uint64_t cur_max = stats.max_us.load(std::memory_order_relaxed);
            while (static_cast<uint64_t>(us) > cur_max &&
                   !stats.max_us.compare_exchange_weak(cur_max, us)) {}
            uint64_t cur_min = stats.min_us.load(std::memory_order_relaxed);
            while (static_cast<uint64_t>(us) < cur_min &&
                   !stats.min_us.compare_exchange_weak(cur_min, us)) {}

            ++window_frames;
            window_us += us;
            if (static_cast<uint64_t>(us) > window_max) window_max = us;
            if (static_cast<uint64_t>(us) < window_min) window_min = us;

            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now - window_start).count();
            if (elapsed_ms >= 5000) {
                double fps = window_frames * 1000.0 / static_cast<double>(elapsed_ms);
                double avg_ms = window_frames ? (window_us / 1000.0) / window_frames : 0.0;
                std::cout << "[sonic] FPS=" << std::fixed << std::setprecision(1) << fps
                          << " | avg=" << std::setprecision(2) << avg_ms << " ms"
                          << " | min=" << window_min / 1000.0 << " ms"
                          << " | max=" << window_max / 1000.0 << " ms"
                          << " | frames=" << window_frames << "\n";
                window_start = now;
                window_frames = 0;
                window_us = 0;
                window_max = 0;
                window_min = UINT64_MAX;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[sonic] FATAL: " << e.what() << "\n";
        g_stop = true;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.decoder_only && cfg.yolo_only) {
        std::cerr << "Error: --decoder-only and --yolo-only are mutually exclusive\n";
        return 1;
    }
    const bool run_yolo = !cfg.decoder_only;
    const bool run_decoder = !cfg.yolo_only;
    const char* mode_str = (run_yolo && run_decoder) ? "both"
                         : (run_decoder ? "decoder-only" : "yolo-only");

    std::cout << "Configuration:\n"
              << "  mode:           " << mode_str << "\n"
              << "  yolo-model:     " << cfg.yolo_model << "\n"
              << "  decoder-model:  " << cfg.decoder_model << "\n"
              << "  device:         " << cfg.device << "\n"
              << "  yolo-tiles:     " << cfg.yolo_tiles << "\n"
              << "  decoder-tiles:  " << cfg.decoder_tiles << "\n"
              << "  decoder-prio:   " << cfg.decoder_priority << "\n"
              << "  yolo-fps:       " << cfg.yolo_fps << "\n"
              << "  warmup:         " << cfg.warmup_iters << " iters\n"
              << "  duration:       " << cfg.duration_sec << " s ("
              << (cfg.duration_sec ? "fixed" : "until Ctrl-C") << ")\n\n";

    std::signal(SIGINT, sigint_handler);
    std::signal(SIGTERM, sigint_handler);

    ov::Core core;

    YoloStats yolo_stats;
    DecoderStats dec_stats;

    auto t_start = std::chrono::steady_clock::now();
    std::thread t_yolo;
    std::thread t_dec;
    if (run_yolo) {
        t_yolo = std::thread(yolo_thread_fn, std::ref(core), std::cref(cfg), std::ref(yolo_stats));
    }
    if (run_decoder) {
        t_dec = std::thread(decoder_thread_fn, std::ref(core), std::cref(cfg), std::ref(dec_stats));
    }

    if (cfg.duration_sec > 0) {
        for (int s = 0; s < cfg.duration_sec && !g_stop.load(); ++s) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        g_stop = true;
    }

    if (t_yolo.joinable()) t_yolo.join();
    if (t_dec.joinable()) t_dec.join();
    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========== Final Results (" << std::fixed << std::setprecision(2)
              << total_sec << " s) ==========\n";

    if (run_yolo) {
        uint64_t f = yolo_stats.frames.load();
        uint64_t total_us = yolo_stats.total_infer_us.load();
        uint64_t mn = yolo_stats.min_us.load();
        uint64_t mx = yolo_stats.max_us.load();
        double fps = total_sec > 0 ? f / total_sec : 0.0;
        double avg_ms = f ? (total_us / 1000.0) / f : 0.0;
        std::cout << "[perception/yolo12s]\n"
                  << "  frames        : " << f << "\n"
                  << "  fps           : " << std::setprecision(2) << fps << "\n"
                  << "  avg latency   : " << avg_ms << " ms\n"
                  << "  min latency   : " << (f ? mn / 1000.0 : 0.0) << " ms\n"
                  << "  max latency   : " << (f ? mx / 1000.0 : 0.0) << " ms\n";
    }
    if (run_decoder) {
        uint64_t f = dec_stats.frames.load();
        uint64_t total_us = dec_stats.total_us.load();
        uint64_t mn = dec_stats.min_us.load();
        uint64_t mx = dec_stats.max_us.load();
        double fps = total_sec > 0 ? f / total_sec : 0.0;
        double avg_ms = f ? (total_us / 1000.0) / f : 0.0;
        std::cout << "[sonic/model_decoder]\n"
                  << "  frames        : " << f << "\n"
                  << "  fps           : " << std::setprecision(2) << fps << "\n"
                  << "  avg latency   : " << avg_ms << " ms\n"
                  << "  min latency   : " << (f ? mn / 1000.0 : 0.0) << " ms\n"
                  << "  max latency   : " << (f ? mx / 1000.0 : 0.0) << " ms\n";
    }
    return 0;
}
