# GPU configuration

## Compiler artifacts

| Key          | Linux worker addon    | Windows worker addon  |
| ------------ | --------------------- | --------------------- |
| oneapi       | oneapi/mom.node       | oneapi/mom.node       |
| dpcpp        | dpcpp/mom.node        | dpcpp/mom.node        |
| dpcpp-opencl | dpcpp-opencl/mom.node | dpcpp-opencl/mom.node |
| acpp-cuda    | acpp-cuda/mom.node    | acpp-cuda/mom.node    |
| acpp-hip     | acpp-hip/mom.node     | acpp-hip/mom.node     |

## Defaults

| OS      | GPU    | Compiler     | Backend     | PearlHash MxNxK/rank   |
| ------- | ------ | ------------ | ----------- | ---------------------- |
| Linux   | Intel  | oneapi       | sycl        | 131072x131072x2048/128 |
| Windows | Intel  | oneapi       | sycl        | 131072x131072x2048/128 |
| Linux   | NVIDIA | dpcpp        | sycl        | 131072x524288x8192/128 |
| Windows | NVIDIA | dpcpp        | sycl        | 131072x524288x8192/128 |
| Linux   | AMD    | acpp-hip     | sycl        | 131072x131072x2048/128 |
| Windows | AMD    | acpp-hip     | sycl        | 131072x131072x2048/128 |
| Linux   | OpenCL | dpcpp-opencl | sycl-opencl | —                      |
| Windows | OpenCL | dpcpp-opencl | sycl-opencl | —                      |

## Overrides

| Algorithm     | OS      | GPU    | Compiler  | Backend     |
| ------------- | ------- | ------ | --------- | ----------- |
| cn/gpu        | Linux   | Intel  | —         | sycl-opencl |
| kawpow        | Linux   | Intel  | —         | sycl-native |
| firopow       | Linux   | Intel  | —         | sycl-native |
| evrprogpow    | Linux   | Intel  | —         | sycl-native |
| meowpow       | Linux   | Intel  | —         | sycl-native |
| octopus       | Linux   | Intel  | —         | sycl-native |
| walahash      | Linux   | Intel  | —         | sycl-native |
| xelishashv3   | Linux   | Intel  | —         | sycl-native |
| zelhash       | Linux   | Intel  | —         | sycl-native |
| zhash         | Linux   | Intel  | dpcpp     | —           |
| pearlhash     | Linux   | Intel  | —         | sycl-native |
| beamhash3     | Linux   | Intel  | —         | sycl-native |
| cn/gpu        | Windows | Intel  | —         | sycl-opencl |
| kawpow        | Windows | Intel  | —         | sycl-native |
| firopow       | Windows | Intel  | —         | sycl-native |
| evrprogpow    | Windows | Intel  | —         | sycl-native |
| meowpow       | Windows | Intel  | —         | sycl-native |
| octopus       | Windows | Intel  | —         | sycl-native |
| walahash      | Windows | Intel  | —         | sycl-native |
| xelishashv3   | Windows | Intel  | —         | sycl-native |
| zelhash       | Windows | Intel  | —         | sycl-native |
| pearlhash     | Windows | Intel  | —         | sycl-native |
| beamhash3     | Windows | Intel  | —         | sycl-native |
| cn/gpu        | Linux   | NVIDIA | —         | sycl-native |
| kawpow        | Linux   | NVIDIA | —         | sycl-native |
| firopow       | Linux   | NVIDIA | —         | sycl-native |
| evrprogpow    | Linux   | NVIDIA | —         | sycl-native |
| meowpow       | Linux   | NVIDIA | —         | sycl-native |
| nexapow       | Linux   | NVIDIA | —         | sycl-native |
| c30           | Linux   | NVIDIA | acpp-cuda | —           |
| etchash       | Linux   | NVIDIA | —         | sycl-native |
| octopus       | Linux   | NVIDIA | —         | sycl-native |
| autolykos2    | Linux   | NVIDIA | acpp-cuda | sycl-native |
| fishhash      | Linux   | NVIDIA | acpp-cuda | sycl-native |
| karlsenhashv2 | Linux   | NVIDIA | acpp-cuda | sycl-native |
| walahash      | Linux   | NVIDIA | —         | sycl-native |
| xelishashv3   | Linux   | NVIDIA | —         | sycl-native |
| zelhash       | Linux   | NVIDIA | —         | sycl-native |
| equihash192_7 | Linux   | NVIDIA | acpp-cuda | —           |
| zhash         | Linux   | NVIDIA | acpp-cuda | —           |
| pearlhash     | Linux   | NVIDIA | —         | native      |
| beamhash3     | Linux   | NVIDIA | —         | sycl-native |
| verthash      | Linux   | NVIDIA | —         | sycl-native |
| cn/gpu        | Windows | NVIDIA | —         | native      |
| kawpow        | Windows | NVIDIA | —         | sycl-native |
| firopow       | Windows | NVIDIA | —         | sycl-native |
| evrprogpow    | Windows | NVIDIA | —         | sycl-native |
| meowpow       | Windows | NVIDIA | —         | sycl-native |
| nexapow       | Windows | NVIDIA | —         | sycl-native |
| c30           | Windows | NVIDIA | acpp-cuda | —           |
| etchash       | Windows | NVIDIA | —         | sycl-native |
| octopus       | Windows | NVIDIA | —         | sycl-native |
| autolykos2    | Windows | NVIDIA | acpp-cuda | sycl-native |
| fishhash      | Windows | NVIDIA | acpp-cuda | sycl-native |
| karlsenhashv2 | Windows | NVIDIA | acpp-cuda | sycl-native |
| walahash      | Windows | NVIDIA | —         | sycl-native |
| xelishashv3   | Windows | NVIDIA | —         | sycl-native |
| zelhash       | Windows | NVIDIA | —         | sycl-native |
| zhash         | Windows | NVIDIA | acpp-cuda | —           |
| pearlhash     | Windows | NVIDIA | —         | native      |
| beamhash3     | Windows | NVIDIA | —         | sycl-native |
| verthash      | Windows | NVIDIA | —         | sycl-native |
| kawpow        | Linux   | AMD    | —         | sycl-native |
| firopow       | Linux   | AMD    | —         | sycl-native |
| evrprogpow    | Linux   | AMD    | —         | sycl-native |
| meowpow       | Linux   | AMD    | —         | sycl-native |
| octopus       | Linux   | AMD    | —         | sycl-native |
| autolykos2    | Linux   | AMD    | —         | sycl-native |
| walahash      | Linux   | AMD    | —         | sycl-native |
| zelhash       | Linux   | AMD    | —         | sycl-native |
| pearlhash     | Linux   | AMD    | —         | native      |
| beamhash3     | Linux   | AMD    | —         | sycl-native |
| kawpow        | Windows | AMD    | —         | sycl-native |
| firopow       | Windows | AMD    | —         | sycl-native |
| evrprogpow    | Windows | AMD    | —         | sycl-native |
| meowpow       | Windows | AMD    | —         | sycl-native |
| octopus       | Windows | AMD    | —         | sycl-native |
| autolykos2    | Windows | AMD    | —         | sycl-native |
| walahash      | Windows | AMD    | —         | sycl-native |
| zelhash       | Windows | AMD    | —         | sycl-native |
| pearlhash     | Windows | AMD    | —         | native      |
| beamhash3     | Windows | AMD    | —         | sycl-native |

Legend: `—` means inherit the default in the matching Defaults row.
