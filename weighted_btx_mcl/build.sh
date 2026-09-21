#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${WBTX_BUILD_DIR:-"$here/build"}
mcl=${MCL_SOURCE:-"$here/../experiments/mcl/build/mcl"}
revision=cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf
mkdir -p "$build"
if [[ ! -f "$mcl/lib/libmcl.a" ]]; then
  if [[ -n "${MCL_SOURCE:-}" ]]; then
    printf 'MCL_SOURCE requires a prebuilt pinned checkout; missing %s/lib/libmcl.a\n' "$mcl" >&2
    exit 1
  fi
  MCL_BUILD_DIR="$here/../experiments/mcl/build" bash "$here/../experiments/mcl/build.sh"
fi
test "$(git -C "$mcl" rev-parse HEAD)" = "$revision"
for source in benchmark tests; do
  "${CXX:-c++}" -O3 -DNDEBUG -std=c++17 -DMCL_FP_BIT=384 -DMCL_FR_BIT=256 \
    -Wall -Wextra -I"$mcl/include" "$here/$source.cpp" "$mcl/lib/libmcl.a" \
    -lpthread -o "$build/$source"
done
python3 "$here/prepare_profile.py" > "$build/profile.txt" 2> "$build/profile.json"
git -C "$mcl" rev-parse HEAD > "$build/mcl_commit.txt"
"${CXX:-c++}" --version > "$build/compiler.txt"
printf '%s\n' 'Protocol: -O3 -DNDEBUG -std=c++17 MCL_FP_BIT=384 MCL_FR_BIT=256; std::thread pool; MCL static library built with MCL_MSM=1' > "$build/build_settings.txt"
