param([ValidateSet('cuda','hip','intel')][string]$Backend)
$ErrorActionPreference = 'Stop'
switch ($Backend) {
  cuda {
    $env:ACPP_VISIBILITY_MASK = 'cuda'
    $env:PATH = "C:\Tools\acpp-cuda\bin;$env:PATH"
    & C:\Tools\acpp-cuda\bin\acpp-info.exe
    if ($LASTEXITCODE -ne 0) { throw 'AdaptiveCpp CUDA validation failed' }
  }
  hip {
    $env:ACPP_VISIBILITY_MASK = 'hip'
    $env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;C:\Tools\acpp-amd\bin;$env:PATH"
    & C:\Tools\acpp-amd\bin\acpp-info.exe
    if ($LASTEXITCODE -ne 0) { throw 'AdaptiveCpp HIP validation failed' }
  }
  intel {
    $env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'
    & 'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\sycl-ls.exe'
    if ($LASTEXITCODE -ne 0) { throw 'oneAPI validation failed' }
  }
}
& C:\Tools\dpcpp\bin\clang++.exe --version | Select-Object -First 2
Get-Content C:\Tools\mom-toolchains.txt
