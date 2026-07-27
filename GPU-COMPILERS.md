# GPU compiler policy

Runtime-readable tables; keep their pipe-delimited Markdown structure.

## Compiler artifacts

| Key            | Compiler toolchain                     | Linux worker addon      | Windows worker addon    | Allocation profile                    |
| -------------- | -------------------------------------- | ----------------------- | ----------------------- | ------------------------------------- |
| `oneapi`       | Intel oneAPI DPC++ 2026                | `oneapi/mom.node`       | `oneapi/mom.node`       | USM; C29 buffers                      |
| `dpcpp`        | open-source LLVM/DPC++, SPIR-V/CUDA    | `dpcpp/mom.node`        | `dpcpp/mom.node`        | USM; C29 buffers                      |
| `dpcpp-opencl` | standards-only LLVM/DPC++ SPIR-V       | `dpcpp-opencl/mom.node` | `dpcpp-opencl/mom.node` | buffers where available; USM fallback |
| `acpp-cuda`    | AdaptiveCpp generic/SSCP, CUDA runtime | `acpp-cuda/mom.node`    | `acpp-cuda/mom.node`    | USM; C29 buffers                      |
| `acpp-hip`     | AdaptiveCpp generic/SSCP, HIP runtime  | `acpp-hip/mom.node`     | `acpp-hip/mom.node`     | USM; C29 buffers                      |

## Selection policy

| OS      | GPU    | Default        | Compiler overrides                                                      | SYCL-native algorithms                                                                                           | Backend overrides                | PearlHash MxNxK/rank     |
| ------- | ------ | -------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------ |
| Linux   | Intel  | `oneapi`       | —                                                                       | `kawpow,firopow,evrprogpow,meowpow,pearlhash,zelhash,beamhash3`                                                  | `cn/gpu=sycl-opencl`             | —                        |
| Windows | Intel  | `oneapi`       | —                                                                       | `kawpow,firopow,evrprogpow,meowpow,pearlhash,zelhash,beamhash3`                                                  | `cn/gpu=sycl-opencl`             | —                        |
| Linux   | NVIDIA | `dpcpp`        | `autolykos2=acpp-cuda`, `fishhash=acpp-cuda`, `karlsenhashv2=acpp-cuda` | `cn/gpu,kawpow,firopow,evrprogpow,meowpow,etchash,autolykos2,fishhash,karlsenhashv2,pearlhash,zelhash,beamhash3` | `pearlhash=native`               | `65536x65536x4096/256`   |
| Windows | NVIDIA | `dpcpp`        | `autolykos2=acpp-cuda`, `fishhash=acpp-cuda`, `karlsenhashv2=acpp-cuda` | `cn/gpu,kawpow,firopow,evrprogpow,meowpow,etchash,autolykos2,fishhash,karlsenhashv2,pearlhash,zelhash,beamhash3` | `cn/gpu=native,pearlhash=native` | `65536x65536x4096/256`   |
| Linux   | AMD    | `acpp-hip`     | —                                                                       | `kawpow,firopow,evrprogpow,meowpow,autolykos2,zelhash,beamhash3`                                                 | `pearlhash=native`               | `32768x32768x2048/128`   |
| Windows | AMD    | `acpp-hip`     | —                                                                       | `kawpow,firopow,evrprogpow,meowpow,autolykos2,zelhash,beamhash3`                                                 | `pearlhash=native`               | `131072x131072x2048/128` |
| Linux   | OpenCL | `dpcpp-opencl` | —                                                                       | —                                                                                                                | `*=sycl-opencl`                  | —                        |
| Windows | OpenCL | `dpcpp-opencl` | —                                                                       | —                                                                                                                | `*=sycl-opencl`                  | —                        |

Legend: `—` means no override; backend values are defined in the README.
