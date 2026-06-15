# https://hub.docker.com/r/intel/oneapi-runtime/tags
FROM intel/oneapi-runtime:2026.0.0-devel-ubuntu24.04

# Runtime image launches a prebuilt miner, so only Node is needed (no build toolchain).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs

COPY *.so* /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
