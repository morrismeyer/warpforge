#!/bin/bash
#
# Test that Python and Java logo generators produce identical SVG output.
# This is a polyglot verification test - a core WarpForge principle.
#
# Usage:
#     ./test-logo-generators.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create temp directories for each implementation
PYTHON_OUT=$(mktemp -d)
JAVA_OUT=$(mktemp -d)

cleanup() {
    rm -rf "$PYTHON_OUT" "$JAVA_OUT"
}
trap cleanup EXIT

echo "=============================================="
echo "  Polyglot Logo Generator Verification Test"
echo "=============================================="
echo ""
echo "Testing that Python and Java implementations"
echo "produce identical SVG output."
echo ""

# Run Python generator
echo "Running Python generator..."
python3 generate-logo.py --all --svg-only --output "$PYTHON_OUT" > /dev/null 2>&1
echo "  Output: $PYTHON_OUT"

# Run Java generator
echo "Running Java generator..."
java GenerateLogo.java --all --svg-only --output "$JAVA_OUT" > /dev/null 2>&1
echo "  Output: $JAVA_OUT"

echo ""
echo "Comparing outputs..."
echo ""

# Compare each SVG file
PASS=0
FAIL=0
TOTAL=0

for py_file in "$PYTHON_OUT"/*.svg; do
    filename=$(basename "$py_file")
    java_file="$JAVA_OUT/$filename"

    TOTAL=$((TOTAL + 1))

    if [ ! -f "$java_file" ]; then
        echo "  FAIL: $filename - missing from Java output"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Compare files (normalize whitespace for comparison)
    py_normalized=$(cat "$py_file" | tr -s '[:space:]' ' ' | sed 's/> </></g')
    java_normalized=$(cat "$java_file" | tr -s '[:space:]' ' ' | sed 's/> </></g')

    if [ "$py_normalized" = "$java_normalized" ]; then
        echo "  PASS: $filename"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $filename - content differs"
        FAIL=$((FAIL + 1))

        # Show diff for debugging
        echo "    --- Python ---"
        head -5 "$py_file" | sed 's/^/    /'
        echo "    --- Java ---"
        head -5 "$java_file" | sed 's/^/    /'
        echo ""
    fi
done

# Check for extra files in Java output
for java_file in "$JAVA_OUT"/*.svg; do
    filename=$(basename "$java_file")
    py_file="$PYTHON_OUT/$filename"

    if [ ! -f "$py_file" ]; then
        echo "  FAIL: $filename - extra file in Java output"
        FAIL=$((FAIL + 1))
        TOTAL=$((TOTAL + 1))
    fi
done

echo ""
echo "=============================================="
echo "  Results: $PASS/$TOTAL passed"
echo "=============================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  All tests PASSED!"
    echo "  Python and Java generators produce identical output."
    echo ""
    exit 0
else
    echo ""
    echo "  $FAIL test(s) FAILED!"
    echo "  Python and Java outputs differ."
    echo ""
    exit 1
fi
