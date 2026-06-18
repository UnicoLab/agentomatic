#!/usr/bin/env bash
# build_studio.sh — Build the Agentomatic Studio frontend and bundle into the Python package.
#
# Usage:
#   ./scripts/build_studio.sh [path-to-agentomatic-studio]
#
# This script:
#   1. Builds the React app in agentomatic-studio (npm run build)
#   2. Copies the built assets to src/agentomatic/studio/static/
#   3. The static files are then included in the Python wheel via hatch
#
# Requirements:
#   - Node.js >= 18 and npm
#   - agentomatic-studio directory (default: ../agentomatic-studio)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STUDIO_DIR="${1:-$PROJECT_ROOT/../agentomatic-studio}"
TARGET_DIR="$PROJECT_ROOT/src/agentomatic/studio/static"

echo "🎨 Building Agentomatic Studio..."
echo "   Studio source: $STUDIO_DIR"
echo "   Target:        $TARGET_DIR"
echo ""

# Check studio directory exists
if [ ! -d "$STUDIO_DIR" ]; then
    echo "❌ Studio directory not found: $STUDIO_DIR"
    echo "   Pass the path as argument: ./scripts/build_studio.sh /path/to/agentomatic-studio"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install Node.js >= 18."
    exit 1
fi

echo "📦 Installing dependencies..."
cd "$STUDIO_DIR"
npm ci --silent 2>/dev/null || npm install --silent

echo "🔨 Building production bundle..."
# Set the PUBLIC_URL so React Router works at /studio/ui/
PUBLIC_URL="/studio/ui" npm run build

echo "📁 Copying build to Python package..."
# Clean target
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Copy build output
cp -r "$STUDIO_DIR/build/"* "$TARGET_DIR/"

echo ""
echo "✅ Studio UI bundled successfully!"
echo "   Files: $(find "$TARGET_DIR" -type f | wc -l | tr -d ' ')"
echo "   Size:  $(du -sh "$TARGET_DIR" | cut -f1)"
echo ""
echo "   The studio will be served at /studio/ui/ when running:"
echo "   agentomatic run --studio"
