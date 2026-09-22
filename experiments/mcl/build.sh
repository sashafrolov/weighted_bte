#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=${MCL_BUILD_DIR:-"$here/build"}
revision=cbb18eb08b86129cf936a6436b5e6c68a2ce8ddf
mkdir -p "$build"
if [[ ! -d "$build/mcl/.git" ]]; then
  git clone https://github.com/herumi/mcl.git "$build/mcl"
fi
git -C "$build/mcl" checkout --detach "$revision"
test "$(git -C "$build/mcl" rev-parse HEAD)" = "$revision"
# MCL make does not track compiler/option changes in its object dependencies.
make -C "$build/mcl" clean
make -C "$build/mcl" -j"${JOBS:-4}" lib/libmcl.a MCL_MSM=1 CXX="${CXX:-c++}"
"${CXX:-c++}" -O3 -DNDEBUG -std=c++17 -DMCL_FP_BIT=384 -DMCL_FR_BIT=256 \
  -I"$build/mcl/include" "$here/kernel_bench.cpp" "$build/mcl/lib/libmcl.a" \
  -lpthread -o "$build/kernel_bench"
python3 "$here/prepare_profile.py" > "$build/weights.txt" 2> "$build/profile.json"
git -C "$build/mcl" rev-parse HEAD > "$build/mcl_commit.txt"
"${CXX:-c++}" --version > "$build/compiler.txt"
printf '%s\n' 'MCL_MSM=1 MCL_FP_BIT=384 MCL_FR_BIT=256; kernel: -O3 -DNDEBUG -std=c++17; no OpenMP' > "$build/build_settings.txt"
printf 'Built %s\n' "$build/kernel_bench"
