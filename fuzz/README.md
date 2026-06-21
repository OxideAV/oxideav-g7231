# oxideav-g7231 fuzz harness

libFuzzer (`cargo-fuzz`) panic-freedom harness for the ITU-T G.723.1
dual-rate (6.3 kbit/s MP-MLQ / 5.3 kbit/s ACELP) decoder and encoder.
The contract under test on every target is the same: **no panic, no
debug-build integer overflow, no out-of-bounds index** on any input,
irrespective of how malformed. Output correctness is the integration
tests' and bench harness's job — the fuzzers only assert *return*.

## Targets

| Target      | Surface                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `decode`    | `Decoder::send_packet`/`receive_frame`/`flush`/`reset` on a mixed-rate (`00`/`01`/`10`/`11`) attacker stream. |
| `roundtrip` | Closed-loop `Encoder` (adversarial 16-bit PCM) → `Decoder` on the encoder's self-produced bitstreams.         |
| `bitstream` | Field-targeted corruption + boundary truncation of structurally near-legal frames, below the trait surface.   |
| `params`    | `make_encoder` / `make_decoder` parameter-validation surface + sustained SID/erasure-concealment decay.       |

Each target's per-byte input layout is documented at the top of its
`fuzz_targets/<name>.rs`.

## Seed corpus

`fuzz/seeds/<target>/` holds a small, version-controlled set of
spec-shaped seed inputs (the running `corpus/` directory is generated
and git-ignored). Each seed is hand-constructed to the corresponding
target's input layout so the fuzzer starts from coverage-rich inputs
rather than random noise — mixed-rate packet streams, sustained
erasure runs, both encoder rates, saturated / alternating-pattern
field bodies, and frames truncated at field boundaries.

## Running

`cargo-fuzz` needs a nightly toolchain. libFuzzer writes newly
discovered inputs into its **first** corpus directory, so name the
git-ignored `corpus/<target>/` dir first (its growth stays untracked)
and the committed `seeds/<target>/` dir second as read-only seed
material — this keeps the seed corpus pristine:

```sh
cargo +nightly fuzz run decode    corpus/decode    fuzz/seeds/decode
cargo +nightly fuzz run roundtrip corpus/roundtrip fuzz/seeds/roundtrip
cargo +nightly fuzz run bitstream corpus/bitstream fuzz/seeds/bitstream
cargo +nightly fuzz run params    corpus/params    fuzz/seeds/params
```

Replay just the committed seeds (no fuzzing, no corpus growth) as a
fast regression gate:

```sh
cargo +nightly fuzz run decode fuzz/seeds/decode -- -runs=0
```

## Clean-room note

The harness, the targets, and the seed corpus are constructed solely
from the ITU-T G.723.1 Recommendation frame-shape definitions (§3.7
rate discriminator, §3.9 postfilter/AGC, §3.10 erasure concealment)
and this crate's own public API.
