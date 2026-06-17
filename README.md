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

- **Analysis by synthesis**: the encoder carries a shadow
  `SynthesisState` mirroring the decoder frame-for-frame, so analysis
  always targets what the decoder will actually produce.
- **ACELP fixed-codebook search**: the §2.16 Table 1 4-pulse stride-8
  track geometry (with the global 1-bit odd-shift grid), followed by
  two passes of coordinate-descent refinement.
- **MP-MLQ fixed-codebook search**: per-track greedy pick with 6 pulses
  on odd subframes and 5 on even subframes; grid bit toggles phase.
- **Joint gain quantisation**: 4-bit ACB + 7-bit FCB magnitude + 1-bit
  FCB sign, with a small-neighbourhood refinement pass minimising
  reconstruction error.
- **LSP quantisation**: 24-bit factorial scalar split VQ in the
  omega = acos(lsp) domain, with per-dim angle ranges keeping the LPC
  stable by construction.

### Decoder (stateful, full-synthesis)

The registered `Decoder` is a full synthesiser:

- Dispatches on the 2-bit rate discriminator and routes `01` / `00`
  payloads through `SynthesisState::decode_acelp` / `decode_mpmlq`.
- Excitation history, LPC synthesis filter memory, and previous-frame
  LSP persist across packets, so a stream decodes without per-frame
  cold-start transients. The previous-frame LSP cold-starts at the
  long-term DC vector `p_DC` per §3.11.
- **LSP stability**: the spec's iterative ordering procedure (§2.6,
  eq. 6–7.3) with `Δ_min = 31.25 Hz` on the normal path and the wider
  `62.5 Hz` on the erasure path (§3.10.1).
- **Frame-erasure concealment** (§3.10.2): a voiced/unvoiced
  classifier cross-correlates a trailing 120-sample window, regenerates
  a periodic (voiced) or pseudo-random (unvoiced) excitation,
  attenuates 2.5 dB per consecutive erased frame, and mutes after 3
  frames. The erasure LSP path leaks toward `p_DC` per §3.10.1.
- **Post-filter chain**: §3.6 pitch (long-term) post-filter with
  per-side prediction-gain weighting and rate-specific γ_ltp — the
  forward cross-correlation reads across the subframe boundary into the
  whole-frame synthesis signal (§3.6 / trace §8) rather than truncating
  the window at sample 60; §3.8 formant filter `A(z/γ₁)/A(z/γ₂)` running
  on the per-subframe interpolated synthesis LPC, with the γ₁ = 0.65 /
  γ₂ = 0.75 tap weighting taken verbatim from the spec's exact Q15
  `PostFiltZeroTable` / `PostFiltPoleTable` (§2.18) instead of a
  recomputed float `gamma^i`; §3.8 signal-adaptive tilt compensation;
  §3.9 leaky-integrator adaptive gain scaling.
- `reset()` reinitialises the synthesiser to silence.
- SID and untransmitted frames are accepted as framing-valid and feed
  the §3.10.2 concealment path; comfort-noise generation (Annex A SID
  parameter parsing) is future work.

### Round-trip quality

On a 2 s voiced synthetic signal (150 Hz fundamental + three harmonics)
encoded and decoded end-to-end (release build):

|    rate | frame size | PSNR      |
| ------: | ---------: | :-------- |
| 5.3 k/s |   20 bytes | ≈ 17.1 dB |
| 6.3 k/s |   24 bytes | ≈ 21.1 dB |

See `tests/codec_roundtrip.rs` for the integration test and
`encoder::tests` for the single-frame lower-bound checks. For a
playable subjective sample:

```bash
cargo test --release -- --ignored roundtrip_writes_sample_raw
aplay -f S16_LE -c 1 -r 8000 /tmp/g7231-sample.raw
```

## Bitstream interoperability

The LSP split VQ, joint gain codebook, and MP-MLQ pulse track layout
driving the encoder / decoder are a clean-room, pure-Rust design —
internally consistent and decode-quality-equivalent, but **not**
bit-compatible with the ITU-T Tables 5 / 7 / 9 bitstream layout. (The
5.3 kbit/s ACELP fixed-codebook *positions* do follow §2.16 Table 1,
but the surrounding gain word and bit packing remain the clean-room
design.) Bitstreams produced by this encoder decode cleanly with this
crate's own decoder at the PSNR figures above, but not with a
spec-table-layout G.723.1 decoder. Achieving that interoperability
requires the full ITU-T numeric tables driving a Q13 / Q15 fixed-point
bit-exact gain quantiser plus the spec's bit-packing field order.

### Spec-table data in tree

The 27 ITU-T G.723.1 normative numeric tables are exposed in
[`spec_tables`](src/spec_tables.rs) as `static` arrays of `i16` / `u32`
in their published Q-formats (§2.2 pre-filter constants, §2.4 LPC
analysis primitives, §2.6 LSP split-VQ codebooks, §2.9 perceptual
weighting, §2.13 MP-MLQ tables, §2.14 ACB gain tables, §2.16 LTP
selector, §2.17 taming gain, §2.18 postfilter tables, bit-allocation
offsets). Each constant carries a doc-comment naming its source CSV
under `docs/audio/g7231/tables/` and the data SHA-256 from its `.meta`
sidecar. Typed accessor helpers (`LspBand`, `SpecRate`,
`fixed_codebook_gain`, `mpmlq_combinatorial`, …) sit on top, with
structural unit tests pinning lengths, symmetry, monotonicity, and the
published bit-allocation constants. This data sits alongside (does not
yet replace) the internally-consistent codebooks in
[`tables`](src/tables.rs); threading it through the gain quantiser to
produce a bit-exact spec-layout stream is future work.

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

### Rate selection

```text
bit_rate = None          -> 6.3 kbit/s MP-MLQ (default)
bit_rate = Some(6300)    -> 6.3 kbit/s MP-MLQ (24-byte frames, rate=00)
bit_rate = Some(5300)    -> 5.3 kbit/s ACELP  (20-byte frames, rate=01)
bit_rate = anything else -> Error::Unsupported
```

The output `CodecParameters` returned by the encoder always has
`bit_rate` set to the exact quantised rate it is operating at. The
encoder has no VAD / CNG — every frame is coded as speech.

## Benchmarks

Three Criterion harnesses cover the encoder, decoder, and full
round-trip across both rates. Each scenario is self-contained: PCM is
synthesised in-bench from a deterministic sum-of-sinusoids generator
and fed through the public encoder factory + the trait-surface
decoder. No external fixtures.

```bash
cargo bench -p oxideav-g7231 --bench {encode,decode,roundtrip}
```

The encoder dominates round-trip cost — analysis-by-synthesis is
roughly two orders of magnitude more expensive than the decoder's
excitation expansion + post-filter chain. On a release build both
directions run comfortably faster than real time (~0.7 ms / 30 ms
frame encoding, ~5 µs / frame decoding).

## Fuzzing

Three `cargo-fuzz` targets live under `fuzz/fuzz_targets/`, each an
ASan-instrumented panic-freedom fuzzer:

- **`decode`** — attacker-supplied byte packets through
  `Decoder::send_packet`, driving the cross-packet state machine
  (rate-discriminator transitions, erasure run counter, postfilter /
  AGC memory) with `flush()` / `reset()` injected mid-stream.
- **`roundtrip`** — arbitrary 16-bit PCM through the registered
  `Encoder` at a fuzzer-chosen rate, then every emitted packet back
  through the `Decoder`.
- **`bitstream`** — structured corruption of near-legal frames (one
  field surgically corrupted or the payload truncated at a field
  boundary) probing the `BitReader` out-of-bits guard.

```bash
cargo fuzz run {decode,roundtrip,bitstream}   # nightly toolchain
```

## License

MIT — see [LICENSE](LICENSE).
