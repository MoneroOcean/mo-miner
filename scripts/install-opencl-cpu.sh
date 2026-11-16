#!/usr/bin/env bash
set -euo pipefail

# Shared by development hashing and standalone clean-host qualification. Install only
# the signed, version-pinned CPU ICD, not the compiler or GPU driver stack.
temporary=$(mktemp -d)
source_list="/etc/apt/sources.list.d/mom-opencl-cpu-$(basename "$temporary").list"
trap 'status=$?; rm -f -- "$temporary/key" "$temporary/keyring"; rmdir -- "$temporary"; sudo rm -f -- "$source_list"; exit "$status"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 --max-time 60 \
  https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -o "$temporary/key"
printf '%s  %s\n' db932ba032a71f732dc415eef5f4f185fc7ae768487d7a357bc2772746b6c48c "$temporary/key" | sha256sum --check
gpg --batch --yes --dearmor --output "$temporary/keyring" "$temporary/key"
chmod 0755 "$temporary"
chmod 0644 "$temporary/keyring"
# Match install-dev's repository identity: APT rejects two Signed-By paths for one source.
keyring=/usr/share/keyrings/oneapi-archive-keyring.gpg
if [[ -e "$keyring" || -L "$keyring" ]]; then
  [[ -f "$keyring" && ! -L "$keyring" ]] && cmp -- "$temporary/keyring" "$keyring"
else
  sudo install -m 0644 "$temporary/keyring" "$keyring"
fi
printf 'deb [signed-by=%s] https://apt.repos.intel.com/oneapi all main\n' "$keyring" |
  sudo tee "$source_list" >/dev/null
sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  intel-oneapi-runtime-opencl=2026.2.0-863 intel-oneapi-runtime-dpcpp-sycl-opencl-cpu=2026.2.0-863
for package in intel-oneapi-runtime-opencl intel-oneapi-runtime-dpcpp-sycl-opencl-cpu; do
  test "$(dpkg-query -W -f='${Version}' "$package")" = '2026.2.0-863'
done
icd='/etc/OpenCL/vendors/intel64.icd'
driver='/opt/intel/oneapi/redist/lib/libintelocl.so'
test -r "$icd" && test -L "$driver" && test -f "$driver"
test "$(tr -d '\r\n' <"$icd")" = "$driver"
resolved=$(readlink -e -- "$driver")
case "$resolved" in /opt/intel/oneapi/redist/lib/libintelocl.so.*) ;; *) echo "Unexpected Intel CPU ICD target: $resolved" >&2; exit 1 ;; esac
