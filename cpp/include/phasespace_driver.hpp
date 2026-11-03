#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <cstddef>
#include <cstdint>

#ifndef _stdcall
#define _stdcall
#endif

#include <onnxruntime_cxx_api.h>

namespace phasespace {

struct CalibrationLUT {
    uint32_t num_points{0};
    float eps_clip{1e-5f};
    float weight_clip_max{20.0f};
    std::vector<float> grid_y;

    [[nodiscard]] float interpolate(float raw_score) const noexcept;
};

class OnnxReweightDriver {
public:
    explicit OnnxReweightDriver(
        const std::string& onnx_model_path,
        const std::string& lut_bin_path
    );

    ~OnnxReweightDriver() = default;

    // Disallow copy, allow move
    OnnxReweightDriver(const OnnxReweightDriver&) = delete;
    OnnxReweightDriver& operator=(const OnnxReweightDriver&) = delete;
    OnnxReweightDriver(OnnxReweightDriver&&) noexcept = default;
    OnnxReweightDriver& operator=(OnnxReweightDriver&&) noexcept = default;

    [[nodiscard]] std::vector<float> score_and_reweight_batch(
        const float* features,
        std::size_t num_events,
        std::size_t num_features,
        int num_threads = 12
    );

    void prepare_batch(
        const float* features,
        std::size_t num_events,
        std::size_t num_features,
        int num_threads
    );

    void score_batch(const float* features, float* weights);

    static void write_weights_hdf5(
        const std::string& h5_out_path,
        const std::vector<float>& weights,
        const std::string& dataset_name = "weights"
    );

    static std::vector<float> read_features_hdf5(
        const std::string& h5_in_path,
        std::size_t& num_events,
        std::size_t& num_features,
        const std::string& dataset_name = "kinematics/features"
    );

    [[nodiscard]] const CalibrationLUT& get_lut() const noexcept { return lut_; }

private:
    struct ThreadScratch {
        std::size_t start_index{0};
        std::size_t event_count{0};
        Ort::Value input_tensor{nullptr};
        Ort::Value output_tensor{nullptr};
        Ort::RunOptions run_options;
    };

    void load_lut(const std::string& lut_bin_path);

    CalibrationLUT lut_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::string input_name_{"kinematics"};
    std::string output_name_{"raw_score"};
    const float* prepared_features_{nullptr};
    std::size_t prepared_events_{0};
    std::size_t prepared_features_count_{0};
    int prepared_threads_{0};
    std::vector<float> raw_scores_;
    std::vector<ThreadScratch> thread_scratch_;
};

} // namespace phasespace
