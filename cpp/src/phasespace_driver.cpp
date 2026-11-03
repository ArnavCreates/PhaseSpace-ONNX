#include "phasespace_driver.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <hdf5.h>

namespace phasespace {

float CalibrationLUT::interpolate(float raw_score) const noexcept {
    if (num_points < 2 || grid_y.empty()) {
        return 1.0f;
    }
    const float clamped_score = std::clamp(raw_score, 0.0f, 1.0f);
    std::size_t low = 0;
    std::size_t high = static_cast<std::size_t>(num_points - 1);
    while (high - low > 1) {
        const std::size_t middle = low + (high - low) / 2;
        const float grid_value = static_cast<float>(middle) / static_cast<float>(num_points - 1);
        if (grid_value < clamped_score) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const std::size_t i0 = std::min(low, static_cast<std::size_t>(num_points - 2));
    const std::size_t i1 = i0 + 1;
    const float pos = clamped_score * static_cast<float>(num_points - 1);
    const float frac = pos - static_cast<float>(i0);

    const float s_cal = (1.0f - frac) * grid_y[i0] + frac * grid_y[i1];
    const float s_clipped = std::clamp(s_cal, eps_clip, 1.0f - eps_clip);

    const float r = s_clipped / (1.0f - s_clipped);
    return std::clamp(r, 0.0f, weight_clip_max);
}

OnnxReweightDriver::OnnxReweightDriver(
    const std::string& onnx_model_path,
    const std::string& lut_bin_path
) : env_(ORT_LOGGING_LEVEL_WARNING, "PhaseSpaceDriver"),
    session_options_(),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    load_lut(lut_bin_path);

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // On Windows (_WIN32), Ort::Session requires const wchar_t* (std::filesystem::path::c_str())
    const std::filesystem::path model_fs_path(onnx_model_path);
    if (!std::filesystem::exists(model_fs_path)) {
        throw std::runtime_error("ONNX model file does not exist: " + onnx_model_path);
    }

    session_ = std::make_unique<Ort::Session>(
        env_,
        model_fs_path.c_str(),
        session_options_
    );

    // Discover input and output node names dynamically if needed
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_->GetInputCount() > 0) {
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name_ptr.get();
    }
    if (session_->GetOutputCount() > 0) {
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name_ptr.get();
    }
}

void OnnxReweightDriver::load_lut(const std::string& lut_bin_path) {
    std::ifstream file(lut_bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibration LUT binary file: " + lut_bin_path);
    }

    // Binary format: uint32 num_points, float eps_clip, float weight_clip_max, followed by floats
    file.read(reinterpret_cast<char*>(&lut_.num_points), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&lut_.eps_clip), sizeof(float));
    file.read(reinterpret_cast<char*>(&lut_.weight_clip_max), sizeof(float));

    if (lut_.num_points == 0 || lut_.num_points > 1000000) {
        throw std::runtime_error("Invalid LUT binary header in " + lut_bin_path);
    }

    lut_.grid_y.resize(lut_.num_points);
    file.read(reinterpret_cast<char*>(lut_.grid_y.data()), static_cast<std::streamsize>(lut_.num_points * sizeof(float)));

    if (!file) {
        throw std::runtime_error("Incomplete read of calibration LUT data from " + lut_bin_path);
    }
}

void OnnxReweightDriver::prepare_batch(
    const float* features,
    std::size_t num_events,
    std::size_t num_features,
    int num_threads
) {
    if (num_events > 0 && features == nullptr) {
        throw std::invalid_argument("Features pointer cannot be null.");
    }
    if (num_events == 0 || num_features == 0) {
        prepared_features_ = features;
        prepared_events_ = num_events;
        prepared_features_count_ = num_features;
        prepared_threads_ = 0;
        raw_scores_.clear();
        thread_scratch_.clear();
        return;
    }

    const int effective_threads = std::max(1, std::min(num_threads, static_cast<int>(num_events)));
    prepared_features_ = features;
    prepared_events_ = num_events;
    prepared_features_count_ = num_features;
    prepared_threads_ = effective_threads;
    raw_scores_.assign(num_events, 0.0f);
    thread_scratch_.clear();
    thread_scratch_.reserve(static_cast<std::size_t>(effective_threads));
    omp_set_dynamic(0);

    const std::size_t base_events = num_events / static_cast<std::size_t>(effective_threads);
    const std::size_t remainder = num_events % static_cast<std::size_t>(effective_threads);
    for (int thread_index = 0; thread_index < effective_threads; ++thread_index) {
        const std::size_t thread_number = static_cast<std::size_t>(thread_index);
        const std::size_t start_index = thread_number * base_events + std::min(thread_number, remainder);
        const std::size_t event_count = base_events + (thread_number < remainder ? 1U : 0U);
        const int64_t input_shape[2] = {
            static_cast<int64_t>(event_count),
            static_cast<int64_t>(num_features),
        };
        const int64_t output_shape[2] = {static_cast<int64_t>(event_count), 1};

        ThreadScratch scratch;
        scratch.start_index = start_index;
        scratch.event_count = event_count;
        float* score_output = raw_scores_.data() + start_index;
        scratch.input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(features + start_index * num_features),
            event_count * num_features,
            input_shape,
            2
        );
        scratch.output_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            score_output,
            event_count,
            output_shape,
            2
        );
        thread_scratch_.push_back(std::move(scratch));
    }
}

void OnnxReweightDriver::score_batch(const float* features, float* weights) {
    if (prepared_events_ == 0) {
        return;
    }
    if (features == nullptr || weights == nullptr) {
        throw std::invalid_argument("Feature and weight buffers must not be null.");
    }
    if (features != prepared_features_ || prepared_threads_ <= 0 || thread_scratch_.empty()) {
        throw std::invalid_argument("Call prepare_batch with these features before score_batch.");
    }

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    #pragma omp parallel num_threads(prepared_threads_)
    {
        const int thread_index = omp_get_thread_num();
        ThreadScratch& scratch = thread_scratch_[static_cast<std::size_t>(thread_index)];
        session_->Run(
            scratch.run_options,
            input_names,
            &scratch.input_tensor,
            1,
            output_names,
            &scratch.output_tensor,
            1
        );
        const float* scores = raw_scores_.data() + scratch.start_index;
        for (std::size_t local_index = 0; local_index < scratch.event_count; ++local_index) {
            weights[scratch.start_index + local_index] = lut_.interpolate(scores[local_index]);
        }
    }
}

std::vector<float> OnnxReweightDriver::score_and_reweight_batch(
    const float* features,
    std::size_t num_events,
    std::size_t num_features,
    int num_threads
) {
    prepare_batch(features, num_events, num_features, num_threads);
    std::vector<float> weights(num_events);
    score_batch(features, weights.data());
    return weights;
}

void OnnxReweightDriver::write_weights_hdf5(
    const std::string& h5_out_path,
    const std::vector<float>& weights,
    const std::string& dataset_name
) {
    hid_t file_id = H5Fcreate(h5_out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("H5Fcreate failed for: " + h5_out_path);
    }

    const hsize_t dims[1] = {weights.size()};
    hid_t space_id = H5Screate_simple(1, dims, nullptr);
    if (space_id < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("H5Screate_simple failed.");
    }

    hid_t dset_id = H5Dcreate2(
        file_id,
        dataset_name.c_str(),
        H5T_NATIVE_FLOAT,
        space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );
    if (dset_id < 0) {
        H5Sclose(space_id);
        H5Fclose(file_id);
        throw std::runtime_error("H5Dcreate2 failed for dataset: " + dataset_name);
    }

    herr_t status = H5Dwrite(
        dset_id,
        H5T_NATIVE_FLOAT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        weights.data()
    );

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);

    if (status < 0) {
        throw std::runtime_error("H5Dwrite failed while saving weights.");
    }
}

std::vector<float> OnnxReweightDriver::read_features_hdf5(
    const std::string& h5_in_path,
    std::size_t& num_events,
    std::size_t& num_features,
    const std::string& dataset_name
) {
    hid_t file_id = H5Fopen(h5_in_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("H5Fopen failed for: " + h5_in_path);
    }

    hid_t dset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dset_id < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("H5Dopen2 failed for dataset: " + dataset_name);
    }

    hid_t space_id = H5Dget_space(dset_id);
    const int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 2) {
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        throw std::runtime_error("Expected 2D dataset in HDF5 file.");
    }

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space_id, dims, nullptr);
    num_events = static_cast<std::size_t>(dims[0]);
    num_features = static_cast<std::size_t>(dims[1]);

    std::vector<float> features(num_events * num_features);
    herr_t status = H5Dread(
        dset_id,
        H5T_NATIVE_FLOAT,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        features.data()
    );

    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    if (status < 0) {
        throw std::runtime_error("H5Dread failed for features.");
    }

    return features;
}

} // namespace phasespace
