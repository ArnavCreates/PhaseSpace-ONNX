# PhaseSpace-ONNX

PhaseSpace-ONNX is a reproducible Monte Carlo likelihood-ratio reweighting pipeline for high-dimensional scientific simulation workflows. Python trains and calibrates a classifier; a C++20 driver uses ONNX Runtime and OpenMP to score the same model.

The generated benchmark contains 100,000 synthetic Monte Carlo events across 10 continuous phase-space observables. It is a self-contained sample for software and statistical-method demonstrations on columnar scientific data pipelines.

## Data and training

The generator writes ROOT and HDF5 samples. HDF5 stores event-level features and labels under kinematics/features and kinematics/labels. Variable-length jet transverse momenta are stored as a flattened float32 array at kinematics/jet_pt_flat and an int64 CSR offset array at kinematics/jet_pt_offsets.

Both ROOT and HDF5 readers process bounded chunks. They sort each event's jets by descending pT, then extract the leading and subleading jet pT. HDF5 chunks rebuild Awkward v2 list-offset arrays from NumPy buffer views.

The pipeline uses a deterministic 60,000 / 20,000 / 20,000 train, calibration, and test split. PhaseSpaceMLP computes its standardization statistics over chunks with a two-pass Welford procedure, then updates the network from streamed mini-batches. Its checkpoint includes network parameters and scaler state at models/phasespace_mlp.pt. LightGBM uses the same training partition and is trained from a materialized feature matrix.

The score-to-probability calibration combines histogram estimates with a monotone isotonic curve. The exported uniform-grid lookup table is shared by Python and C++ inference.

Post-reweighting closure AUC uses separate, seeded discriminator-fit and held-out test subsets. Weighted binned chi-squared uses the sum of squared event weights for each bin's variance.

## Measured results

These tables reproduce the generated JSON values. Memory and timing measurements are machine- and environment-dependent; the JSON files retain the full precision and the run configuration.

### Ingestion

Source: assets/ingestion_memory_comparison.json. RSS reduction is relative to CSV flattening.

| Ingestion path | Events | Time (ms) | Traced peak (MB) | RSS delta (MB) | Throughput (events/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSV flattening | 100000 | 2168.6293000821024 | 64.45106887817383 | 143.03515625 | 46112.07641444948 |
| Monolithic ROOT | 100000 | 114.23270008526742 | 32.964728355407715 | 22.06640625 | 875406.0783414589 |
| Awkward ROOT streaming, 2 MB step | 100000 | 214.88670003600419 | 14.94155502319336 | 15.3671875 | 465361.5136871897 |
| Jagged HDF5 streaming, 10000-event chunks | 100000 | 141.62729983218014 | 2.664576530456543 | 11.5625 | 706078.5605493715 |

| Ingestion path | RSS reduction vs CSV (%) |
| --- | ---: |
| Monolithic ROOT | 84.57273943796596 |
| Awkward ROOT streaming | 89.25635633722042 |
| Jagged HDF5 streaming | 91.9163230193626 |

The benchmark records Python tracemalloc peak and process RSS delta separately. RSS delta is measured with the platform-specific process API and is not a substitute for a process high-water mark.

### Reweighting and closure

Source: assets/model_comparison.json. Closure AUC is computed on the held-out discriminator-test subset.

| Metric | PhaseSpaceMLP | LightGBM |
| --- | ---: | ---: |
| Initial separation ROC AUC | 0.5918668300000001 | 0.641316965 |
| Held-out post-reweighting closure AUC | 0.5154149520863798 | 0.5161410587749569 |
| Nominal events | 10000 | 10000 |
| Kish effective sample size | 9125.616901548092 | 8078.210325011258 |
| Kish effective sample size fraction | 0.9125616901548091 | 0.8078210325011258 |
| Mean KS, unweighted | 0.04924000000000002 | 0.04924000000000002 |
| Mean KS, reweighted | 0.02204953351974487 | 0.014760949263572682 |
| KS reduction (%) | 55.22028123528664 | 70.02244260038043 |
| Mean chi-squared / ndof, unweighted | 5.074808362233334 | 5.074808362233334 |
| Mean chi-squared / ndof, reweighted | 2.527286674446954 | 1.2672225257857839 |

Per-observable KS and chi-squared values are available in the JSON artifact.

### Numerical parity

Source: assets/benchmark_results.json. All paths score the same 20,000 held-out events using the trained checkpoint and calibration lookup table.

| Comparison | Max absolute error | Mean absolute error | Max relative error | Mean relative error |
| --- | ---: | ---: | ---: | ---: |
| PyTorch vs ONNX Runtime | 3.814697265625e-06 | 7.62015588406939e-08 | 1.8922982008007239e-06 | 7.063745499635843e-08 |
| PyTorch vs C++ | 3.337860107421875e-06 | 1.1328756954753771e-07 | 1.991072622331558e-06 | 1.0269377526128665e-07 |
| ONNX Runtime vs C++ | 3.337860107421875e-06 | 8.515715421708592e-08 | 1.8105778281096718e-06 | 7.577865801522421e-08 |

The maximum absolute error is below the recorded threshold of 1e-05.

### Latency and throughput

Source: assets/benchmark_results.json. Each configuration uses 5 warmup passes and 50 timed passes over 20,000 events. Standard deviations are across timed pass durations. Python inference uses one Torch and ONNX Runtime thread; the C++ driver used 12 OpenMP threads.

| Execution path | Batch size | Threads | Mean pass (ms) | Std pass (ms) | Mean per event (µs) | Std per event (µs) | Throughput (events/s) | C++ speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python online event loop | 1 | 1 | 3365.9324180008844 | 101.26163547812845 | 168.29662090004422 | 5.063081773906423 | 5941.889947950448 | 528.4038332811435 |
| Python streaming micro-batch | 256 | 1 | 35.974830011837184 | 8.170601628912442 | 1.7987415005918592 | 0.4085300814456221 | 555944.2530630217 | 5.647540033255445 |
| Python vectorized PyTorch | 20000 | 1 | 21.548129986040294 | 1.4420726984237555 | 1.0774064993020147 | 0.07210363492118778 | 928154.7871187322 | 3.3827519601319143 |
| Python vectorized ONNX Runtime | 20000 | 1 | 24.316608007065952 | 0.9383649553586755 | 1.2158304003532976 | 0.046918247767933775 | 822483.1355667852 | 3.817363894358862 |
| C++ ONNX Runtime + OpenMP | 20000 | 12 | 6.3692 | 1.0942 | 0.3185 | 0.05471 | 3140113.76 | 1 |

The C++ driver measured 5.647540033255445x lower per-event latency than the Python streaming micro-batch path. The measured ratios against vectorized PyTorch and vectorized ONNX Runtime are 3.3827519601319143x and 3.817363894358862x, respectively; the JSON retains the full timing distributions and run configuration.

## Build and run

### Requirements

- Python 3.10 or later and the dependencies in pyproject.toml.
- CMake, a C++20 compiler, OpenMP, and the HDF5 C development library.
- The ONNX Runtime C++ SDK matching the host platform. The project was verified with the 1.20.1 SDK. Download the matching archive from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1) and extract it.
- For the C++ benchmark, set ONNXRUNTIME_ROOT to the extracted SDK directory, or place the SDK at the configured fallback location ../../.deps/onnxruntime relative to this project.

### Ubuntu

Install the compiler, CMake, HDF5 development files, and Python venv support, then use the shared Linux commands below:

~~~bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libhdf5-dev python3-venv
~~~

### Linux (GCC 12+)

On shared workstations or computing clusters, activate an environment providing a modern C++20 compiler, CMake, OpenMP, and HDF5 development libraries (via environment modules or package manager) before building.

### Linux commands

From the project directory, extract the Linux ONNX Runtime C++ SDK and set ONNXRUNTIME_ROOT to its extracted directory. The example below assumes it is under $HOME/opt:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-linux-x64-1.20.1"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build build
pytest
python scripts/run_pipeline.py
python scripts/benchmark_ingestion.py
python scripts/benchmark_parity.py
~~~

Run the parity benchmark after the build and pipeline have produced the model and test data.
### Windows PowerShell with MSYS2 UCRT64

Install the MSYS2 UCRT64 compiler toolchain, CMake, Ninja, and HDF5 package. Set `MSYS2_ROOT` to the MSYS2 installation directory and `ONNXRUNTIME_ROOT` to the extracted Windows x64 ONNX Runtime C++ SDK. From the project directory:

~~~powershell
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;" + $env:Path
$env:ONNXRUNTIME_ROOT = "path\to\onnxruntime-win-x64"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT"
cmake --build build --config Release
.\.venv\Scripts\pytest.exe
.\.venv\Scripts\python.exe scripts/run_pipeline.py
.\.venv\Scripts\python.exe scripts/benchmark_ingestion.py
.\.venv\Scripts\python.exe scripts/benchmark_parity.py
~~~

For PowerShell, the CMake executable and compiler must resolve to the MSYS2 UCRT64 installation. CMake copies onnxruntime.dll beside the executable when it finds the DLL in the SDK.

A fresh checkout can run pytest before generating project data; tests create their fixtures in temporary directories.

## Generated files

- models/phasespace_mlp.pt: trained PyTorch checkpoint, including the fitted scaler.
- models/phasespace_mlp.onnx: exported inference graph.
- models/calibration_lut.bin and models/calibration_lut.json: calibration lookup table.
- data/test_kinematics.h5 and data/cpp_weights_out.h5: held-out inputs and C++ weights.
- assets/ingestion_memory_comparison.json, assets/model_comparison.json, and assets/benchmark_results.json: full precision measurements.
- assets/kinematic_reweighting_closure.png and assets/cpp_vs_python_latency.png: generated plots.

## References & Acknowledgments

The classifier ratio estimator follows K. Cranmer, J. Pavez, and G. Louppe, [“Approximating Likelihood Ratios with Calibrated Discriminative Classifiers”](https://arxiv.org/abs/1506.02169) (2015). The columnar readers use [Uproot](https://github.com/scikit-hep/uproot5), [Awkward Array](https://github.com/scikit-hep/awkward), and HDF5 for the sample formats.
