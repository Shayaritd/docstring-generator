#!/usr/bin/env bash
set -uo pipefail

API_URL="${1:-http://localhost:8000}"
MAX_WAIT_SECONDS=120
FAILURES=0

check() {
    local description="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  [PASS] $description"
    else
        echo "  [FAIL] $description"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "=== Testing deployment at ${API_URL} ==="

echo ""
echo "Waiting for model to load (up to ${MAX_WAIT_SECONDS}s)..."
elapsed=0
model_loaded=false
while [ "$elapsed" -lt "$MAX_WAIT_SECONDS" ]; do
    health_response=$(curl -sf "${API_URL}/health" 2>/dev/null || echo "")
    if echo "$health_response" | grep -q '"model_loaded"[[:space:]]*:[[:space:]]*true'; then
        model_loaded=true
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "  ...still waiting (${elapsed}s elapsed)"
done

if $model_loaded; then
    check "Model loaded within ${MAX_WAIT_SECONDS}s" 0
else
    check "Model loaded within ${MAX_WAIT_SECONDS}s" 1
    echo "  Last /health response: ${health_response:-<no response>}"
fi

health_status=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health" 2>/dev/null || echo "000")
[ "$health_status" = "200" ]; check "/health returns HTTP 200" $?

echo ""
echo "Testing /generate with a valid function..."
generate_response=$(curl -sf -X POST "${API_URL}/generate" \
    -H "Content-Type: application/json" \
    -d '{"function_code": "def add(a, b):\n    return a + b", "max_length": 100}' 2>/dev/null || echo "")

if echo "$generate_response" | grep -q '"docstring"'; then
    check "/generate returns a docstring for valid input" 0
    echo "  Response: $(echo "$generate_response" | head -c 200)..."
else
    check "/generate returns a docstring for valid input" 1
    echo "  Got: ${generate_response:-<no response>}"
fi

echo ""
echo "Testing /generate with invalid input (expect HTTP 422)..."
invalid_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${API_URL}/generate" \
    -H "Content-Type: application/json" \
    -d '{"function_code": "not valid python :("}' 2>/dev/null || echo "000")
[ "$invalid_status" = "422" ]; check "/generate rejects invalid Python with HTTP 422" $?

docs_status=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/docs" 2>/dev/null || echo "000")
[ "$docs_status" = "200" ]; check "/docs (Swagger UI) is reachable" $?

echo ""
echo "=== Summary ==="
if [ "$FAILURES" -eq 0 ]; then
    echo "All checks passed."
    exit 0
else
    echo "${FAILURES} check(s) failed."
    exit 1
fi
