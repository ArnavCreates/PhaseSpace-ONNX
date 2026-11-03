#include "phasespace_driver.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>

struct BenchmarkArgs {
    std::string model_path{"models/phasespace_mlp.onnx"};
    std::string lut_path{"models/calibration_lut.bin"};
    std::string input_h5{"data/test_kinematics.h5"};
    std::string output_h5{"data/cpp_weights_out.h5"};
    int threads{12};
    int iterations{50};
};

BenchmarkArgs parse_cli_args(int argc, char* argv[]) {
    BenchmarkArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--lut" && i + 1 < argc) {
            args.lut_path = argv[++i];
        } else if (arg == "--input-h5" && i + 1 < argc) {
            args.input_h5 = argv[++i];
        } else if (arg == "--output-h5" && i + 1 < argc) {
            args.output_h5 = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            args.iterations = std::stoi(argv[++i]);
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    try {
        const BenchmarkArgs args = parse_cli_args(argc, argv);

        std::cout << "=== PhaseSpace-ONNX C++20 Benchmark Driver ===" << std::endl;
        std::cout << "Model Path     : " << args.model_path << std::endl;
        std::cout << "LUT Path       : " << args.lut_path << std::endl;
        std::cout << "Input HDF5     : " << args.input_h5 << std::endl;
        std::cout << "Output HDF5    : " << args.output_h5 << std::endl;
        std::cout << "Threads        : " << args.threads << std::endl;
        std::cout << "Iterations     : " << args.iterations << std::endl;

        std::size_t num_events = 0;
        std::size_t num_features = 0;
        std::cout << "Loading kinematics from " << args.input_h5 << "..." << std::endl;
        std::vector<float> features = phasespace::OnnxReweightDriver::read_features_hdf5(
            args.input_h5,
            num_events,
            num_features,
            "kinematics/features"
        );
        std::cout << "Loaded " << num_events << " events with " << num_features << " features." << std::endl;

        phasespace::OnnxReweightDriver driver(args.model_path, args.lut_path);
        driver.prepare_batch(features.data(), num_events, num_features, args.threads);
        std::vector<float> final_weights(num_events);

        // Warmup runs
        std::cout << "Executing 5 warmup iterations..." << std::endl;
        for (int w = 0; w < 5; ++w) {
            driver.score_batch(features.data(), final_weights.data());
        }

        // Timed benchmark iterations
        std::cout << "Executing " << args.iterations << " timed benchmark iterations..." << std::endl;
        std::vector<double> batch_times_ms;
        batch_times_ms.reserve(args.iterations);

        for (int it = 0; it < args.iterations; ++it) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            driver.score_batch(features.data(), final_weights.data());
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            batch_times_ms.push_back(ms);
        }

        // Write resulting weights for numerical parity assertion
        std::cout << "Writing output weights to " << args.output_h5 << "..." << std::endl;
        phasespace::OnnxReweightDriver::write_weights_hdf5(
            args.output_h5,
            final_weights,
            "weights"
        );

        // Compute metrics
        double sum_ms = std::accumulate(batch_times_ms.begin(), batch_times_ms.end(), 0.0);
        double mean_batch_ms = sum_ms / static_cast<double>(batch_times_ms.size());
        double squared_deviation_sum = 0.0;
        for (double batch_ms : batch_times_ms) {
            const double deviation = batch_ms - mean_batch_ms;
            squared_deviation_sum += deviation * deviation;
        }
        const double std_batch_ms = batch_times_ms.size() > 1
            ? std::sqrt(squared_deviation_sum / static_cast<double>(batch_times_ms.size() - 1))
            : 0.0;
        double per_event_us = (mean_batch_ms * 1000.0) / static_cast<double>(num_events);
        double throughput_eps = static_cast<double>(num_events) / (mean_batch_ms / 1000.0);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=== BENCHMARK_RESULTS ===" << std::endl;
        std::cout << "CPP_TOTAL_EVENTS=" << num_events << std::endl;
        std::cout << "CPP_MEAN_BATCH_MS=" << mean_batch_ms << std::endl;
        std::cout << "CPP_STD_BATCH_MS=" << std_batch_ms << std::endl;
        std::cout << "CPP_PER_EVENT_US=" << per_event_us << std::endl;
        std::cout << "CPP_THROUGHPUT_EVENTS_PER_SEC=" << throughput_eps << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "CRITICAL ERROR in benchmark driver: " << ex.what() << std::endl;
        return 1;
    }
}
