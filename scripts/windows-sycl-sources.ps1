function Get-MomWindowsSyclSources {
  [ordered]@{
    lib          = 'sycl\lib.cpp'
    ethash       = 'sycl\etchash\ethash.cpp'
    etchash      = 'sycl\etchash\etchash.cpp'
    autolykos2   = 'sycl\autolykos2\autolykos2.cpp'
    pearlhash    = 'sycl\pearlhash\pearlhash.cpp'
    c29          = 'sycl\c29\c29.cpp'
    cn_gpu       = 'sycl\cn_gpu\cn_gpu.cpp'
    kawpow       = 'sycl\kawpow\kawpow.cpp'
    fishhash     = 'sycl\fishhash\fishhash.cpp'
    zelhash      = 'sycl\zelhash\zelhash.cpp'
    beamhash3    = 'sycl\beamhash3\beamhash3.cpp'
    blake2b      = 'sycl\c29\blake2b.cpp'
  }
}
