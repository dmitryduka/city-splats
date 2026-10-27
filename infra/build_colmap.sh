#!/usr/bin/env bash
# Build COLMAP (with ONNX for ALIKED/LightGlue, CUDA when available) and pycolmap against it.
#
# Same script on macOS arm64 (CPU + ONNX) and Linux aarch64/x86_64 (DGX Spark: CUDA + ONNX).
# Installs into tools/colmap-install (no sudo) and pip-installs pycolmap into .venv.
#
# usage: infra/build_colmap.sh [colmap_tag]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-4.2.1}"
SRC="$ROOT/tools/colmap-src"
BUILD="$SRC/build"
PREFIX="$ROOT/tools/colmap-install"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -d "$SRC" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/colmap/colmap.git "$SRC"
fi

# Local patches (e.g. allow ONNX extractors/matchers on CoreML without CUDA/OpenGL).
for patch in "$ROOT"/infra/patches/colmap-*.patch; do
  if git -C "$SRC" apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $(basename "$patch")"
  else
    git -C "$SRC" apply "$patch"
    echo "applied: $(basename "$patch")"
  fi
done

CMAKE_ARGS=(
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DONNX_ENABLED=ON
  -DFETCH_ONNX=ON
  -DGUI_ENABLED=OFF
  -DTESTS_ENABLED=OFF
)

case "$(uname -s)" in
  Darwin)
    BREW="$(brew --prefix)"
    # No OpenGL on headless macOS builds: SIFT runs on CPU here.
    CMAKE_ARGS+=(
      -DOPENGL_ENABLED=OFF
      -DCUDA_ENABLED=OFF
      -DOpenMP_ROOT="$BREW/opt/libomp"
      -DCMAKE_PREFIX_PATH="$BREW"
    )
    ;;
  Linux)
    if command -v nvcc >/dev/null 2>&1; then
      # DGX Spark (GB10) is sm_121; native picks whatever GPU is present.
      CMAKE_ARGS+=(-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=native)
    else
      CMAKE_ARGS+=(-DCUDA_ENABLED=OFF)
    fi
    ;;
esac

cmake -S "$SRC" -B "$BUILD" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD"
cmake --install "$BUILD"

# pycolmap: python/CMakeLists.txt does find_package(colmap), so point it at our install.
uv pip install --python "$PYTHON" scikit-build-core "pybind11==3.0.4" numpy
CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
  uv pip install --python "$PYTHON" --reinstall --no-build-isolation \
  -C cmake.define.GENERATE_STUBS=OFF \
  -C cmake.define.CMAKE_PREFIX_PATH="$PREFIX;${BREW:-}" \
  ${BREW:+-C cmake.define.OpenMP_ROOT="$BREW/opt/libomp"} \
  -C cmake.define.CMAKE_POLICY_DEFAULT_CMP0074=NEW \
  -C cmake.define.pybind11_DIR="$("$PYTHON" -m pybind11 --cmakedir)" \
  "$SRC"

"$PYTHON" - <<'EOF'
import pycolmap
print("pycolmap", pycolmap.__version__, "cuda:", pycolmap.has_cuda)
EOF
