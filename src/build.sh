#!/usr/bin/env bash
# Reproducible Linux build of iBEM 1.0.0 - 3D ellipsoidal path (calculationType 1)
# Run from inside build_linux/ (it lives next to the iBEM sources).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"          # iBEM-1.0.0 source root
EIGEN="$HERE/eigen3"                    # vendored Eigen headers
BLD="$(mktemp -d)"

# 1) stage sources
cp "$SRC"/*.h "$BLD"/ 2>/dev/null || true
cp "$SRC"/*.cpp "$BLD"/ 2>/dev/null || true
# 2) apply the 3D-only workarounds / fixes
cp "$HERE/classFinder_3D.h"        "$BLD/classFinder.h"        # dispatcher: 3D only
cp "$HERE/inputGenerator_3D.h"     "$BLD/inputGenerator.h"     # dispatcher: 3D only
cp "$HERE/inputGenerator_3D.cpp"   "$BLD/inputGenerator.cpp"   # case 1 only
cp "$HERE/configElastic3D_patched.h" "$BLD/configElastic3D.h"  # setUp() ordering fix
cp "$HERE/postProcessorElastic3D_patched.cpp" "$BLD/postProcessorElastic3D.cpp"  # dump T.txt

# 3) compile only the 3D ellipsoidal translation units
cd "$BLD"
SRCS="main.cpp ibemRunner.cpp inputGenerator.cpp configElastic3D.cpp \
BEMBuilderElastic3D.cpp BEMElasticHelper3D.cpp integratorElastic3D.cpp \
integratorElastic3DHelper.cpp integratorhelper.cpp postProcessorElastic3D.cpp"
g++ -O2 -std=c++17 -I. -I"$EIGEN" $SRCS -o "$HERE/iBEM_run"
echo "Built: $HERE/iBEM_run"
