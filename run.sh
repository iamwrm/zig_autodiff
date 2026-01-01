#!/bin/bash
set -e

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo "pixi not found, installing..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

pixi run zig build run
