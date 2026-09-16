# Test123

Sparse distributed representations in one C file (`sdr.c`).

Build: `cc -std=c17 -O2 -Wall -Wextra sdr.c -o sdr_demo -lm`
Test: `./sdr_demo` (exit 0 = all checks pass)
UBSan: `cc -std=c17 -O1 -g -fsanitize=undefined sdr.c -o sdr_ubsan -lm && ./sdr_ubsan`
Strict ISO C (no GNU __int128): `cc -std=c17 -O2 -Wall -Wextra -Wpedantic -DSDR_NO_INT128 sdr.c -o sdr_demo -lm`

Define `SDR_NO_MAIN` to use as a library without the demo main.
