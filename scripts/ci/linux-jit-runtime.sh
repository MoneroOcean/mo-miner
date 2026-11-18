#!/usr/bin/env bash
# Execute the linked source-JIT proof fixture in an isolated CPU-only container:
#   linux-jit-runtime.sh run              run the 11-case runtime suite under fixed docker/cgroup limits
#   linux-jit-runtime.sh bundle-evidence  tar the bounded evidence directory after a failure
# `run` reads MOM_NATIVE_JIT_RUNTIME_FIXTURE (the admitted executable) and
# MOM_NATIVE_JIT_RUNTIME_EXPECTED_SHA256 (its upstream pin). The fixture bytes must not change.
set -Eeuo pipefail

repo="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
image="ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
inspect_format='memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}} cpu={{.HostConfig.NanoCpus}} pids={{.HostConfig.PidsLimit}} network={{.HostConfig.NetworkMode}} readonly={{.HostConfig.ReadonlyRootfs}} security={{json .HostConfig.SecurityOpt}}'
expected_limits='memory=4294967296 swap=4294967296 cpu=2000000000 pids=128 network=none readonly=true'

# docker --mount uses commas between fields; a comma or line break inside one path would inject options.
validate_mount_paths() {
  local input_path
  for input_path in "$@"; do
    case "$input_path" in *','*|*$'\n'*|*$'\r'*) return 2 ;; esac
  done
}

# Run the fixed 11-case suite inside the keeper container; the inner script re-checks the cgroup
# limits from inside, and only the last 1 MiB of its combined output is kept as container.log.
# Reads $name and $evidence; sets exec_status.
dispatch_runtime() {
  set +e
  # These substitutions must run inside the contained process, not on the host.
  # shellcheck disable=SC2016
  timeout --signal=TERM --kill-after=10s 21m docker exec --user 10001:10001 \
    --env MOM_NATIVE_JIT_RUNTIME_FIXTURE=/proof/native-dpcpp-jit "$name" /bin/bash -ec '
      test "$(cat /sys/fs/cgroup/memory.max)" = 4294967296
      test "$(cat /sys/fs/cgroup/memory.swap.max)" = 0
      test "$(cat /sys/fs/cgroup/cpu.max)" = "200000 100000"
      test "$(cat /sys/fs/cgroup/pids.max)" = 128
      test "$(cat /sys/fs/cgroup/memory.oom.group)" = 1
      set +e
      /usr/bin/timeout --signal=TERM --kill-after=10s 20m /qa/node --test --test-concurrency=1 --test-reporter=tap /qa/tests/native-jit-runtime.js 2>&1 | tee /tmp/native-jit-runtime.tap
      test_status=${PIPESTATUS[0]}
      set -e
      if [ "$test_status" -eq 124 ]; then exit 124; fi
      test "$test_status" -eq 0
      grep -Fxq "# tests 11" /tmp/native-jit-runtime.tap
      grep -Fxq "# pass 11" /tmp/native-jit-runtime.tap
      grep -Fxq "# fail 0" /tmp/native-jit-runtime.tap
      grep -Fxq "# skipped 0" /tmp/native-jit-runtime.tap
      for case_name in "spirv" "sm_61" "sm_120" "sm_61 kawpow" "sm_61 firopow" "sm_61 evrprogpow" "sm_61 meowpow" "sm_120 kawpow" "sm_120 firopow" "sm_120 evrprogpow" "sm_120 meowpow"; do
        grep -Fq "# Subtest: $case_name preserves source, cache, error recovery and image lifecycle" /tmp/native-jit-runtime.tap
      done
    ' 2>&1 | tail -c 1048576 > "$evidence/container.log"
  exec_status=${PIPESTATUS[0]}
  set -e
}

# Sample cgroup memory telemetry once per second while the container's cgroup exists.
sampler() {
  while [ -r "$cgroup_path/memory.peak" ]; do
    {
      printf '%s ' "$SECONDS"
      for key in memory.current memory.peak memory.events memory.swap.current; do
        if [ -r "$cgroup_path/$key" ]; then printf '%s=' "$key"; tr '\n' ',' < "$cgroup_path/$key"; printf ' '; fi
      done
      printf '\n'
    } >> "$evidence/cgroup-samples.txt"
    sleep 1
  done
}

# EXIT trap: capture final cgroup and container evidence while the keeper still holds the cgroup,
# remove the container, and fail unless the fixture bytes are unchanged and every limit held.
finish() {
  rc=$?
  trap - EXIT
  set +e
  if [ "$sampler_pid" -gt 0 ]; then kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null; fi
  if [ "$created" -eq 1 ]; then
    docker inspect --format "$inspect_format" "$name" > "$evidence/container-inspect.txt"
    if [ -n "$cgroup_path" ] && [ -r "$cgroup_path/memory.peak" ]; then
      for key in memory.current memory.peak memory.max memory.events memory.oom.group memory.swap.current memory.swap.max cpu.max pids.max; do
        if [ -r "$cgroup_path/$key" ]; then
          { printf '%s=' "$key"; tr '\n' ' ' < "$cgroup_path/$key"; printf '\n'; } >> "$evidence/cgroup-final.txt"
        fi
      done
      cp "$cgroup_path/memory.events" "$evidence/memory.events"
    else
      printf 'final cgroup telemetry unavailable\n' > "$evidence/cgroup-final.txt"
    fi
    docker stop --time 5 "$name" > "$evidence/container-stop.txt" 2>&1
    docker inspect --format 'oom={{.State.OOMKilled}} exit={{.State.ExitCode}}' "$name" > "$evidence/container-after-stop.txt"
    container_exit="$(docker inspect -f '{{.State.ExitCode}}' "$name")"
    docker rm -f "$name" > /dev/null 2>&1
  fi
  proof_after="$(sha256sum "$MOM_NATIVE_JIT_RUNTIME_FIXTURE" 2>/dev/null | awk '{print $1}')"
  printf 'expected_sha256=%s\nbefore_sha256=%s\nafter_sha256=%s\ntimed_out=%s\ncontainer_exit=%s\n' \
    "$MOM_NATIVE_JIT_RUNTIME_EXPECTED_SHA256" "$proof_before" "$proof_after" "$timed_out" "$container_exit" > "$evidence/proof-integrity.txt"
  test "$proof_after" = "$proof_before" || rc=1
  printf 'exec_status=%s\ncontainer_id=%s\nhost_pid=%s\ncgroup=%s\n' "$exec_status" "$container_id" "$host_pid" "$cgroup_path" >> "$evidence/proof-integrity.txt"
  if [ "$created" -eq 1 ]; then
    grep -Fq "$expected_limits" "$evidence/container-inspect.txt" || rc=1
    grep -Fq 'no-new-privileges:true' "$evidence/container-inspect.txt" || rc=1
    grep -Fxq 'memory.max=4294967296 ' "$evidence/cgroup-final.txt" || rc=1
    grep -Fxq 'memory.swap.max=0 ' "$evidence/cgroup-final.txt" || rc=1
    grep -Fxq 'memory.swap.current=0 ' "$evidence/cgroup-final.txt" || rc=1
    grep -Fxq 'cpu.max=200000 100000 ' "$evidence/cgroup-final.txt" || rc=1
    grep -Fxq 'pids.max=128 ' "$evidence/cgroup-final.txt" || rc=1
    grep -Fxq 'memory.oom.group=1 ' "$evidence/cgroup-final.txt" || rc=1
    peak="$(awk -F= '$1 == "memory.peak" {gsub(/ /, "", $2); print $2}' "$evidence/cgroup-final.txt")"
    [[ "$peak" =~ ^[0-9]+$ ]] && [ "$peak" -le 4294967296 ] || rc=1
    awk '$1 == "oom" || $1 == "oom_kill" || $1 == "oom_group_kill" {if ($2 != 0) bad=1} END {exit bad}' "$evidence/memory.events" || rc=1
    grep -Fq 'oom=false' "$evidence/container-after-stop.txt" || rc=1
    [ "$exec_status" -eq 0 ] && [ "$timed_out" -eq 0 ] || rc=1
  fi
  exit "$rc"
}

# Create the bounded keeper container, verify its host cgroup limits, then dispatch the suite.
run_proof() {
  evidence="$RUNNER_TEMP/native-linux-jit-runtime-${MOM_NATIVE_CI_SUFFIX}"
  mkdir -m 700 "$evidence"
  exec > >(tee "$evidence/host.log") 2>&1
  test "$(node --version)" = "v24.15.0"
  test -f "$MOM_NATIVE_JIT_RUNTIME_FIXTURE"
  test -x "$MOM_NATIVE_JIT_RUNTIME_FIXTURE"
  node_bin="$(node -p 'process.execPath')"
  test -f "$node_bin"
  test -x "$node_bin"
  test_script="$repo/tests/native-jit-runtime.js"
  test -f "$test_script"
  validate_mount_paths "$node_bin" "$test_script" "$MOM_NATIVE_JIT_RUNTIME_FIXTURE" || exit 2
  proof_before="$(sha256sum "$MOM_NATIVE_JIT_RUNTIME_FIXTURE" | awk '{print $1}')"
  test "$proof_before" = "$MOM_NATIVE_JIT_RUNTIME_EXPECTED_SHA256"
  name="mom-native-linux-jit-runtime-${MOM_NATIVE_CI_SUFFIX}"
  created=0
  cgroup_path=""
  container_id=""
  host_pid=""
  sampler_pid=0
  exec_status=125
  container_exit=125
  timed_out=0
  trap finish EXIT
  docker create --name "$name" --network none --cpus 2 --memory 4g --memory-swap 4g --pids-limit 128 \
    --read-only --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges:true \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=134217728,mode=1777 \
    --mount "type=bind,source=$node_bin,target=/qa/node,readonly" \
    --mount "type=bind,source=$test_script,target=/qa/tests/native-jit-runtime.js,readonly" \
    --mount "type=bind,source=$MOM_NATIVE_JIT_RUNTIME_FIXTURE,target=/proof/native-dpcpp-jit,readonly" \
    --workdir /tmp --entrypoint /bin/bash "$image" -ec 'while :; do sleep 3600; done' > "$evidence/container-id.txt"
  created=1
  docker start "$name" > /dev/null
  container_id="$(docker inspect -f '{{.Id}}' "$name")"
  host_pid="$(docker inspect -f '{{.State.Pid}}' "$name")"
  [[ "$host_pid" =~ ^[1-9][0-9]*$ ]]
  cg_rel="$(awk -F: '$1 == "0" {print $3; exit}' "/proc/$host_pid/cgroup")"
  [[ "$cg_rel" == *"$container_id"* ]]
  cgroup_path="$(realpath -m "/sys/fs/cgroup$cg_rel")"
  case "$cgroup_path" in /sys/fs/cgroup/*) ;; *) exit 3 ;; esac
  for key in memory.max memory.swap.max cpu.max pids.max memory.oom.group; do test -r "$cgroup_path/$key"; done
  test "$(cat "$cgroup_path/memory.max")" = 4294967296
  test "$(cat "$cgroup_path/memory.swap.max")" = 0
  test "$(cat "$cgroup_path/cpu.max")" = "200000 100000"
  test "$(cat "$cgroup_path/pids.max")" = 128
  printf 1 | sudo tee "$cgroup_path/memory.oom.group" > /dev/null
  test "$(cat "$cgroup_path/memory.oom.group")" = 1
  printf 'container_id=%s\nhost_pid=%s\ncgroup=%s\n' "$container_id" "$host_pid" "$cgroup_path" > "$evidence/host-gate.txt"
  docker inspect --format "$inspect_format" "$name" > "$evidence/container-inspect.txt"
  grep -Fq "$expected_limits" "$evidence/container-inspect.txt"
  grep -Fq 'no-new-privileges:true' "$evidence/container-inspect.txt"
  sampler & sampler_pid=$!
  dispatch_runtime
  if [ "$exec_status" -eq 124 ] || [ "$exec_status" -eq 137 ]; then timed_out=1; fi
  printf 'exec_status=%s\n' "$exec_status" > "$evidence/status.txt"
  cat "$evidence/container.log"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  sampler_pid=0
  test "$exec_status" -eq 0
}

# Pack the evidence directory, if one was created, for the failure-only upload.
bundle_evidence() {
  local name="native-linux-jit-runtime-${MOM_NATIVE_CI_SUFFIX}"
  if [ -d "$RUNNER_TEMP/$name" ]; then
    tar -cf "$RUNNER_TEMP/native-linux-jit-runtime-failure-${MOM_NATIVE_CI_SUFFIX}.tar" -C "$RUNNER_TEMP" "$name"
  fi
}

main() {
  case "${1:-}" in
    run) run_proof ;;
    bundle-evidence) bundle_evidence ;;
    *) printf '[native-ci] Usage: linux-jit-runtime.sh run|bundle-evidence\n' >&2; exit 2 ;;
  esac
}

# Tests source this file to exercise single functions; only direct execution runs a command.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
