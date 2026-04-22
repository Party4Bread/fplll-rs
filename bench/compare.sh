#!/usr/bin/env bash
# Compare fplll (C++) vs fplll-rs (Rust) LLL speed on 20-dim, 1024-bit integer relations.
# Uses reference fplll's latticegen to generate the lattices so both tools reduce the
# same inputs (seeded by fplll's gmp_randstate).
set -euo pipefail

cd "$(dirname "$0")/.."

FPLLL_BIN=./fplll/fplll
LATGEN_BIN=./fplll/latticegen
RUST_FPLLL=./target/release/fplll

RUNS=5
DIM=20
BITS=1024

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Generating $RUNS lattices: ${DIM}-dim, ${BITS}-bit integer relations ==="
for i in $(seq 0 $((RUNS-1))); do
  $LATGEN_BIN -randseed $((42 + i)) r $DIM $BITS > $TMPDIR/lat_$i.txt
done

run_one() {
  local label="$1"; shift
  local total_ns=0
  for i in $(seq 0 $((RUNS-1))); do
    local t0=$(date +%s%N)
    "$@" < $TMPDIR/lat_$i.txt > $TMPDIR/out_$i.txt
    local t1=$(date +%s%N)
    local dt=$((t1 - t0))
    total_ns=$((total_ns + dt))
    printf "  %-28s seed %d : %.3fs\n" "$label" $i "$(echo "scale=3; $dt/1000000000" | bc -l)"
  done
  printf "  %-28s MEAN     : %.3fs\n" "$label" "$(echo "scale=3; $total_ns/$RUNS/1000000000" | bc -l)"
}

echo
echo "=== fplll (C++, default = wrapper) ==="
run_one "fplll -a lll" $FPLLL_BIN -a lll

echo
echo "=== fplll (C++, -m fast -f double) ==="
run_one "fplll -m fast -f double" $FPLLL_BIN -a lll -m fast -f double

echo
echo "=== fplll (C++, -m proved -f mpfr -p 128) ==="
run_one "fplll -m proved -f mpfr" $FPLLL_BIN -a lll -m proved -f mpfr -p 128

echo
echo "=== fplll-rs (Rust, f64 + ROW_EXPO) ==="
run_one "fplll-rs -a lll" $RUST_FPLLL -a lll
