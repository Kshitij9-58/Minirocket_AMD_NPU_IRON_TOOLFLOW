# MiniROCKET Implementation on AMD XDNA NPU (IRON Toolflow)

### This repository contains the bare metal implementation of the MiniROCKET time series classifier on an AMD Ryzen AI Neural Processing Unit (NPU). ROCKET is a time series classifier published in 2020 that achieves state-of-the-art accuracy with a fraction of the computational expense by transforming input time series using random convolutional kernels. MiniROCKET introduces engineering improvements that make the algorithm nearly deterministic and exceptionally fast, running ~75× faster on large datasets. The AMD NPU is a dedicated AI engine within AMD's Ryzen processors. It accelerates machine learning tasks, particularly inference, by offloading computationally intensive workloads from the CPU.

### This project implements MiniROCKET training and inference natively on an AMD NPU using the AMD IRON API and MLIR-AIE toolchain. By mapping the convolutional transforms and linear inference directly onto the dedicated spatial architecture of the AIE Tile array, this work explores the practical limits of bare-metal edge AI. The implementation specifically addresses the optimization strategies required to overcome I/O bandwidth constraints and memory wall bottlenecks inherent in high-throughput time-series classification. By utilizing a optimized spatial configuration, the pipeline demonstrates how localized Data Movement Accelerators (DMAs) can be programmed to sustain parallel compute cycles without host CPU intervention.

## Phase 1: Standard Environment Setup for AMD Ryzen AI

The execution of programs on the Ryzen AI NPU requires a specific software configuration on Ubuntu 24.04.

1. Hardware and Kernel Initialization

SecureBoot must be disabled in BIOS to permit unsigned driver installation.

Upgrade the Linux kernel to 6.11 or higher.

### Updates the local package lists
sudo apt update

# Installs Ubuntu Hardware Enablement stack
sudo apt install --install-recommends linux-generic-hwe-24.04

# Reboot to apply the new kernel
sudo reboot
2. Driver and Runtime Installation

The XDNA driver and Xilinx Runtime (XRT) manage communication between the OS and the physical NPU.

## Add AMD XRT repository
sudo add-apt-repository ppa:amd-team/xrt

## Refresh packages
sudo apt update

## Install XRT and XDNA driver
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms

## Reboot to load driver
sudo reboot

Grant your user access to the NPU:

## Add user to render group
sudo usermod -aG render $USER

## Verify NPU
xrt-smi examine

3. Compiler and Toolchain Prerequisites

Install foundational C++ build tools.

sudo apt install build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
4. IRON Application Development Environment

Clone MLIR-AIE and create a virtual environment.

## Clone repository
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

## Create virtual environment
python3 -m venv ironenv

## Activate environment
source ironenv/bin/activate

## Upgrade pip
python3 -m pip install --upgrade pip

Install IRON and LLVM-AIE:

python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3

python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

5. Environment Finalization
## Load compiler environment
source utils/env_setup.sh

## Build drivers
bash ./utils/build_drivers.sh

## Install dev dependencies
python3 -m pip install -r python/requirements_dev.txt
pre-commit install
pre-commit install --hook-type pre-push

## Install ML and notebook dependencies
python3 -m pip install -r python/requirements_ml.txt
python3 -m pip install -r python/requirements_notebook.txt

## Register environment with Jupyter
python3 -m ipykernel install --user --name ironenv

## Install MiniROCKET libraries
python3 -m pip install scikit-learn matplotlib numpy
6. Clone the MiniROCKET Repository
# Go back to workspace
cd ..

## Clone MiniROCKET project
git clone https://github.com/Kshitij9-58/Minirocket_AMD_NPU_IRON_TOOLFLOW.git

Phase 2: Hardware Compilation and Developer Build Guide

If you need to recompile the .xclbin hardware bitstream or C++ runner, follow these steps.

1. Directory Setup
cd Minirocket_AMD_NPU_IRON_TOOLFLOW/minirocket_implementation/minirocket_config_file_packages
2. Toolchain Version Pinning

Newer nightly builds break Vitis integration, so specific versions are required.


## Install stable MLIR-AIE
python3 -m pip install --user mlir-aie==1.2.0 \
-f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.2.0

## Install compatible LLVM-AIE
python3 -m pip install --user llvm-aie==20.0.0.2026011501+5e2bd2df \
-f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

## Add local python bin to PATH
export PATH=$HOME/.local/bin:$PATH
3. Vitis Compiler License Setup
## Load Vitis environment
source /tools/Xilinx/Vitis/2023.2/settings64.sh 2>/dev/null

## Add AI Engine compiler tools
export PATH=$PATH:/tools/Xilinx/Vitis/2023.2/aietools/bin

## Export license variables
export XILINXD_LICENSE_FILE=/home/Xilinx/Vivado/2023.2/data/ip/core_licenses/Xilinx.lic
export LM_LICENSE_FILE=/opt/Xilinx.lic

4. Compile AIE Microkernel
## Remove old builds
rm -rf ukernel.o xchesscc_wrapper

## Locate compiler wrapper
REAL_WRAPPER=$(which xchesscc_wrapper)

## Compile microkernel
"$REAL_WRAPPER" aie2 -c ukernel.cc -o ukernel.o
5. Compile Hardware Bitstream
aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
  --xclbin-name=final.xclbin \
  --npu-insts-name=insts.txt \
  minirocket.mlir
6. Compile Host C++ Runner
## Move to implementation directory
cd ..

## Compile runner
g++ -O3 -std=c++17 minirocket_runner.cpp -o minirocket_runner \
    -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib \
    -lxrt_coreutil \
    -pthread


Phase 3: How to Run the MiniROCKET Pipeline

After completing Phase 1, run these commands every time you start a new terminal.

## Navigate to implementation directory
cd Minirocket_AMD_NPU_IRON_TOOLFLOW/minirocket_implementation

## Source XRT runtime
source /opt/xilinx/xrt/setup.sh

## Activate MLIR-AIE environment
source ../../mlir-aie/ironenv/bin/activate

## Load MLIR-AIE compiler variables
source ../../mlir-aie/utils/env_setup.sh

## Run the MiniROCKET pipeline
python3 minirocket_transform_inf.py
