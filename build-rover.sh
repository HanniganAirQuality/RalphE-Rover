#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root
cd "$(dirname "$0")"

# Extract the bundled ARM64 Zig compiler if needed
if [ ! -x zig/zig_compiler/zig ]; then
    mkdir -p zig/zig_compiler
    tar -xf zig/zig-arm64.tar.xz \
        -C zig/zig_compiler --strip-components=1
fi

# Build ZigMaps
(
    cd zig
    ./zig_compiler/zig build -Doptimize=ReleaseFast
)

# Build ROS 2 packages
source /opt/ros/jazzy/setup.bash
colcon build

echo "RalphE build complete"