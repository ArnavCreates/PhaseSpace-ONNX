# PhaseSpace-ONNX

PhaseSpace-ONNX is a reproducible Monte Carlo likelihood-ratio reweighting pipeline for high-dimensional scientific simulation workflows. Python trains and calibrates a classifier; a C++20 driver uses ONNX Runtime and OpenMP to score the same model.

The generated benchmark contains 100,000 synthetic Monte Carlo events across 10 continuous phase-space observables. It is a self-contained sample for software and statistical-method demonstrations on columnar scientific data pipelines.

The likelihood-ratio method follows Cranmer, Pavez, and Louppe, “Approximating Likelihood Ratios with Calibrated Classifiers” (2015), [arXiv:1506.02169](https://arxiv.org/abs/1506.02169).

## Data and training

The generator writes ROOT and HDF5 samples. HDF5 stores event-level features and labels under kinematics/features and kinematics/labels. Variable-length jet transverse momenta are stored as a flattened float32 array at kinematics/jet_pt_flat and an int64 CSR offset array at kinematics/jet_pt_offsets.

Both ROOT and HDF5 readers process bounded chunks. They sort each event's jets by descending pT, then extract the leading and subleading jet pT. HDF5 chunks rebuild Awkward v2 list-offset arrays from NumPy buffer views.

The pipeline uses a deterministic 60,000 / 20,000 / 20,000 train, calibration, and test split. PhaseSpaceMLP computes its standardization statistics over chunks with a two-pass Welford procedure, then updates the network from streamed mini-batches. Its checkpoint includes network parameters and scaler state at models/phasespace_mlp.pt. LightGBM uses the same training partition and is trained from a materialized feature matrix.

The score-to-probability calibration combines histogram estimates with isotonic regression. Its response is monotonically non-decreasing, and the exported uniform-grid lookup table is used by Python and C++ inference.

Post-reweighting closure AUC uses separate, seeded discriminator-fit and held-out test subsets. Weighted binned chi-squared uses the sum of squared event weights for each bin's variance.

## Measured results

These tables reproduce the generated JSON values. Memory and timing measurements are machine- and environment-dependent; the JSON files retain the full precision and the run configuration.

### Ingestion

Source: assets/ingestion_memory_comparison.json. RSS reduction is relative to CSV flattening.

| Ingestion path | Events | Time (ms) | Traced peak (MB) | RSS delta (MB) | Throughput (events/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSV flattening | 100000 | 1951.2388000730425 | 64.48300647735596 | 143.203125 | 51249.49339683929 |
| Monolithic ROOT | 100000 | 106.52590007521212 | 32.99075508117676 | 28.17578125 | 938738.841252648 |
| Awkward ROOT streaming, 2 MB step | 100000 | 191.32720003835857 | 14.914112091064453 | 15.171875 | 522664.8379318326 |
| Jagged HDF5 streaming, 10000-event chunks | 100000 | 129.16509993374348 | 2.689366340637207 | 11.8046875 | 774202.9391166499 |

| Ingestion path | RSS reduction vs CSV (%) |
| --- | ---: |
| Monolithic ROOT | 80.32460447354065 |
| Awkward ROOT streaming | 89.40534642662303 |
| Jagged HDF5 streaming | 91.75668303327878 |

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
| Python online event loop | 1 | 1 | 2964.135315991007 | 320.45847425344186 | 148.20676579955034 | 16.022923712672092 | 6747.330289580032 | 446.94440832192504 |
| Python streaming micro-batch | 256 | 1 | 34.35627399943769 | 3.1519780944122733 | 1.7178136999718845 | 0.15759890472061366 | 582135.3037389136 | 5.180379071085297 |
| Python vectorized PyTorch | 20000 | 1 | 20.373477996326983 | 1.718473575674088 | 1.0186738998163491 | 0.08592367878370441 | 981668.4222303961 | 3.07199607906016 |
| Python vectorized ONNX Runtime | 20000 | 1 | 22.058176011778414 | 1.117629940164905 | 1.1029088005889207 | 0.055881497008245254 | 906693.2818615914 | 3.3260217146831144 |
| C++ ONNX Runtime + OpenMP | 20000 | 12 | 6.6312 | 0.9102 | 0.3316 | 0.04551 | 3016027.1684 | 1 |

The C++ driver measured 5.180379071085297x lower per-event latency than the Python streaming micro-batch path, meeting the 4.2x target. The measured ratios against vectorized PyTorch and vectorized ONNX Runtime are 3.07199607906016x and 3.3260217146831144x, respectively; those results are shown directly.

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

### Enterprise & Cluster Linux

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

Install the MSYS2 UCRT64 compiler toolchain, CMake, Ninja, and HDF5 package. Extract the Windows x64 ONNX Runtime C++ SDK and set its location. From the project directory:

~~~powershell
$env:Path = "C:\msys64\ucrt64\bin;" + $env:Path
$env:ONNXRUNTIME_ROOT = "C:\deps\onnxruntime-win-x64-1.20.1"
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