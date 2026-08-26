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
#   - Node.js >=20.19.0 <22, or >=22.12.0, and npm
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

# Check Node.js.  Keep this in sync with agentomatic-studio/package.json and
# Vite: unsupported Node versions can appear to build successfully while
# producing an unverified package.
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install a supported Node.js version."
    exit 1
fi

# Keep npm on the exact Node installation we validate below. In shells with
# multiple Node managers it is otherwise possible for ``node`` to pass this
# check while the ``npm`` launcher resolves a different, unsupported runtime.
NODE_BIN_DIR="$(dirname "$(command -v node)")"
NPM_BIN="$NODE_BIN_DIR/npm"
if [ ! -x "$NPM_BIN" ]; then
    echo "❌ npm was not found next to Node.js at $NODE_BIN_DIR."
    exit 1
fi
export PATH="$NODE_BIN_DIR:$PATH"

if ! node -e '
const [major, minor] = process.versions.node.split(".").map(Number);
const supported =
  (major === 20 && minor >= 19) ||
  major === 21 ||
  (major === 22 && minor >= 12) ||
  major >= 23;
process.exit(supported ? 0 : 1);
'; then
    echo "❌ Node.js $(node --version) is unsupported. Use >=20.19.0 <22 or >=22.12.0."
    exit 1
fi

echo "📦 Installing dependencies..."
cd "$STUDIO_DIR"
"$NPM_BIN" ci --silent 2>/dev/null || "$NPM_BIN" install --silent

echo "🔨 Building production bundle..."
# Set the Vite base so assets and React Router work at /studio/ui/.
VITE_BASE_URL="/studio/ui/" "$NPM_BIN" run build

echo "📁 Copying build to Python package..."
# Clean target
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Copy build output
cp -r "$STUDIO_DIR/dist/"* "$TARGET_DIR/"

# Vite occasionally leaves trailing spaces in otherwise minified chunks. They
# are semantically inert but make ``git diff --check`` fail for a freshly
# bundled release. Normalize only generated text assets so the package tree
# remains release-clean across macOS and Linux builders.
while IFS= read -r -d '' asset; do
    perl -pi -e 's/[ \t]+$//' "$asset"
done < <(find "$TARGET_DIR" -type f \( -name '*.html' -o -name '*.css' -o -name '*.js' \) -print0)

echo ""
echo "✅ Studio UI bundled successfully!"
echo "   Files: $(find "$TARGET_DIR" -type f | wc -l | tr -d ' ')"
echo "   Size:  $(du -sh "$TARGET_DIR" | cut -f1)"
echo ""
echo "   The studio will be served at /studio/ui/ when running:"
echo "   agentomatic run --studio"
