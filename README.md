# oxideav-g7231

Pure-Rust **ITU-T G.723.1** dual-rate narrowband speech codec — encoder
and full-synthesis decoder for both 6.3 kbit/s (MP-MLQ) and 5.3 kbit/s
(ACELP). No C libraries, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-g7231 = "0.0"
```

## Codec summary

- Sample rate: **8 kHz**, mono, S16.
- Frame length: **30 ms / 240 samples**.
- Bitstream (rate discriminator in the low 2 bits of the first byte):
  - `00` — 6.3 kbit/s MP-MLQ, 24-byte frame (192 bits).
  - `01` — 5.3 kbit/s ACELP, 20-byte frame (160 bits).
  - `10` — SID (silence-insertion descriptor), 4-byte frame.
  - `11` — untransmitted / erasure, 0 or 1 byte.
- Codec id: `"g723_1"`.

## What is implemented

### Encoder (both rates)

Full LPC → LSP → open-loop pitch → closed-loop adaptive-codebook →
rate-specific fixed-codebook → joint gain quantisation pipeline,
packed into the correct 20- or 24-byte frame with the right
discriminator. Default rate (no `bit_rate` hint) is 6.3 kbit/s MP-MLQ;
request `Some(5300)` to get ACELP.

Encoder highlights:

- **Analysis by synthesis**: the encoder carries a shadow `SynthesisState`
  that mirrors the decoder frame-for-frame, so analysis always targets
  what the decoder will actually produce.
- **ACELP fixed-codebook search**: 4-pulse stride-8 track layout (with a
  1-bit grid shift) covers every position in the 60-sample subframe,
  followed by two passes of coordinate-descent refinement that
  re-optimise each pulse given the rest fixed.
- **MP-MLQ fixed-codebook search**: per-track greedy pick with 6 pulses
  on odd subframes and 5 on even subframes; grid bit toggles phase.
- **Joint gain quantisation**: 4-bit ACB + 7-bit FCB magnitude +
  1-bit FCB sign, followed by a small-neighbourhood refinement pass
  that picks the codeword pair minimising reconstruction error (not
  just the nearest-in-log-space pair).
- **LSP quantisation**: 24-bit factorial scalar split VQ in the
  omega = acos(lsp) domain, with pre-tuned per-dim angle ranges so the
  resulting LPC stays stable by construction.

### Decoder (stateful, full-synthesis)

The registered `Decoder` is a full synthesiser:

- Dispatches on the 2-bit rate discriminator in the first payload byte.
- Routes `01` payloads through `SynthesisState::decode_acelp` and `00`
  payloads through `SynthesisState::decode_mpmlq`.
- Excitation history, LPC synthesis filter memory, and previous-frame
  LSP persist across packets, so a stream of packets decodes without
  per-frame cold-start transients.
- `reset()` reinitialises the synthesiser to silence.
- SID and untransmitted frames are accepted as framing-valid and emit
  silence (comfort-noise generation + erasure concealment are future
  work).

### Round-trip quality

On a 2 s voiced synthetic signal (150 Hz fundamental + three harmonics,
amplitude ≈ 20 000 on S16) encoded and decoded through this crate end-
to-end (release build, x86_64):

|    rate | frame size | PSNR    |
| ------: | ---------: | :------ |
| 5.3 k/s |   20 bytes | 18.7 dB |
| 6.3 k/s |   24 bytes | 21.8 dB |

See `tests/codec_roundtrip.rs::roundtrip_two_seconds_voiced_psnr_both_rates`
for the integration test and
`encoder::tests::mpmlq_roundtrip_voiced_psnr_floor` for the single-
frame lower-bound check. Individual LSP / gain / pulse / LPC building
blocks are covered by unit tests in `src/encoder.rs`.

For a playable subjective sample:

```bash
cargo test --release -- --ignored roundtrip_writes_sample_raw
aplay -f S16_LE -c 1 -r 8000 /tmp/g7231-sample.raw
```

## Not bit-compatible with ITU-T reference tables

The LSP split VQ, joint gain codebook, and fixed-codebook pulse track
layout currently driving the encoder / decoder are a clean-room,
pure-Rust design — internally consistent and decode-quality-equivalent
to a reference G.723.1 codec, but **not** bit-compatible with ITU-T
Tables 5 / 7 / 9. Bitstreams produced by this encoder decode cleanly
with this crate's own decoder at the PSNR figures above, but not with
an external, spec-table G.723.1 reference decoder.

Achieving that interoperability requires the full ITU-T numeric tables
to be in tree as static data, plus a Q13 / Q15 fixed-point bit-exact
gain quantiser, plus the spec's bit-packing field order, while keeping
the pure-Rust / no-FFI invariant.

### Spec-table data is in tree (round 197)

The 27 ITU-T G.723.1 normative numeric tables are now exposed in
[`spec_tables`](src/spec_tables.rs) as `static` arrays of `i16` / `u32`
in their published Q-formats:

- §2.2 high-pass / input pre-filter constants.
- §2.4 LPC analysis primitives (180-point Hamming window, 10-point
  binomial lag window, 10-point bandwidth-expansion γ^i, 512-point
  cosine lookup for LSP↔LSF conversion).
- §2.6 LSP split-VQ — 3-band partition info, DC-predicted reference
  vector, plus the three band codebooks (band 0/1 = 256 × 3, band 2 =
  256 × 4) in Q13.
- §2.9 perceptual-weighting filter A(z/γ₁)/A(z/γ₂) zero + pole tables.
- §2.13 MP-MLQ pulse-count-per-subframe, max-position search bounds,
  6 × 30 combinatorial C(n, k) table, 24-level FCB gain codebook.
- §2.14 adaptive-codebook gain tables (85 × 20 at 5.3k, 170 × 20 at
  6.3k) plus the small gain-quantizer decision-factor table.
- §2.16 1-tap LTP selector + companion gain.
- §2.17 taming-procedure gain (85 / 170 entries, rate-specific).
- §2.18 adaptive postfilter zero + pole tables.
- Bit-allocation segment base offsets + boundaries.

Each constant carries a doc-comment naming the source CSV under
`docs/audio/g7231/tables/` and the data SHA-256 from its `.meta`
sidecar. Structural unit tests pin the lengths, symmetry of the
Hamming window, antisymmetry of the LSP cosine lookup, monotonicity
of the FCB gain codebook + bandwidth-expansion factors, the 3-band
LSP partition summing to LpcOrder = 10, the 6/5/6/5 MP-MLQ pulse
pattern, and the published bit-allocation constants {0, 32, 96} /
{2048, 18432, 231233}.

The data sits alongside (does not yet replace) [`tables`](src/tables.rs)'s
internally-consistent codebooks. Threading these spec tables through
the LPC / LSP / gain quantiser to produce a bit-exact spec-compatible
bitstream is the next-round task.

## Benchmarks (round 203)

Three Criterion harnesses cover the encoder, decoder, and full round-
trip across both dual rates. Each scenario is self-contained: every PCM
input is synthesised in-bench from a deterministic sum-of-sinusoids
generator (180 Hz fundamental + harmonics, matching the integration-test
voiced signal) and fed through the public encoder factory + the trait-
surface decoder produced by `register_codecs` + `CodecRegistry::first_decoder`.
No external fixtures.

Run with:

```bash
cargo bench -p oxideav-g7231 --bench encode
cargo bench -p oxideav-g7231 --bench decode
cargo bench -p oxideav-g7231 --bench roundtrip
```

Baseline numbers on a release build (macOS aarch64, single-thread,
1 s = 33 × 30 ms frames):

| Bench                          | Time   | Input throughput |
| :----------------------------- | -----: | ---------------: |
| `encode_mpmlq_voiced_1s`       | 22 ms  | ~ 700 KiB/s      |
| `encode_acelp_voiced_1s`       | 22 ms  | ~ 710 KiB/s      |
| `encode_mpmlq_silence_1s`      | 19 ms  | ~ 815 KiB/s      |
| `encode_acelp_voiced_5s`       | 113 ms | ~ 687 KiB/s      |
| `decode_mpmlq_synth_1s`        | 170 µs | ~ 89 MiB/s       |
| `decode_acelp_synth_1s`        | 168 µs | ~ 90 MiB/s       |
| `decode_erased_5s`             | 789 µs | ~ 96 MiB/s       |
| `decode_mixed_5s`              | 804 µs | ~ 94 MiB/s       |
| `roundtrip_mpmlq_voiced_1s`    | 20 ms  | ~ 765 KiB/s      |
| `roundtrip_acelp_voiced_1s`    | 22 ms  | ~ 714 KiB/s      |
| `roundtrip_mpmlq_voiced_5s`    | 101 ms | ~ 770 KiB/s      |

The encoder dominates the round-trip cost — analysis-by-synthesis (LPC,
LSP split-VQ, pitch + ACB + FCB search per subframe, joint gain quant)
is roughly two orders of magnitude more expensive than the decoder's
excitation expansion + post-filter chain. Real-time at 8 kHz needs
30 ms per 30 ms input frame; we currently spend ~ 0.7 ms / frame
encoding and ~ 5 µs / frame decoding, so we're ~ 43 × faster than
real-time encoding and ~ 6 000 × faster than real-time decoding.
Future optimisation rounds can A/B-test their tweaks against these
numbers.

## Quick use

```rust
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, RuntimeContext, SampleFormat, TimeBase,
};

let mut ctx = RuntimeContext::new();
oxideav_g7231::register(&mut ctx);

let mut params = CodecParameters::audio(CodecId::new(oxideav_g7231::CODEC_ID_STR));
params.sample_rate = Some(8_000);
params.channels = Some(1);
params.sample_format = Some(SampleFormat::S16);
params.bit_rate = Some(6_300); // or Some(5_300) for ACELP

let mut enc = ctx.codecs.make_encoder(&params)?;

// 240 S16 samples = one 30 ms frame.
let pcm = vec![0i16; 240];
let mut bytes = Vec::with_capacity(pcm.len() * 2);
for s in &pcm {
    bytes.extend_from_slice(&s.to_le_bytes());
}
let frame = Frame::Audio(AudioFrame {
    format: SampleFormat::S16,
    channels: 1,
    sample_rate: 8_000,
    samples: pcm.len() as u32,
    pts: Some(0),
    time_base: TimeBase::new(1, 8_000),
    data: vec![bytes],
});
enc.send_frame(&frame)?;
enc.flush()?;

while let Ok(pkt) = enc.receive_packet() {
    // 24-byte MP-MLQ packet; discriminator is pkt.data[0] & 0b11 == 0b00.
    assert_eq!(pkt.data.len(), 24);
}
# Ok::<(), oxideav_core::Error>(())
```

## Rate selection

```text
bit_rate = None          -> 6.3 kbit/s MP-MLQ (default)
bit_rate = Some(6300)    -> 6.3 kbit/s MP-MLQ (24-byte frames, rate=00)
bit_rate = Some(5300)    -> 5.3 kbit/s ACELP  (20-byte frames, rate=01)
bit_rate = anything else -> Error::Unsupported
```

The output `CodecParameters` returned by the encoder always has
`bit_rate` set to the exact quantised rate it is operating at.

## Status

- Encoder, both rates: implemented, round-trip-verified at >18 dB PSNR
  on voiced speech-like input.
- Decoder, both rates: full-synthesis, stateful across packets.
- No VAD / CNG in the encoder — every frame is coded as speech.
- Comfort-noise generation (SID) and erasure concealment (untransmitted
  frames) in the decoder emit silence today; framing is accepted.

## License

MIT — see [LICENSE](LICENSE).
