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
- SID and untransmitted frames are accepted as framing-valid and feed
  the spec-aligned §3.10.2 concealment path described below; comfort-
  noise generation (Annex A SID parameter parsing) is future work, but
  packets carrying only the discriminator no longer emit raw silence.

### LSP stability check — G.723.1 §3.1 / 2.6 (round 216)

Decoded-LSP post-processing now matches the spec's iterative ordering
procedure (eq. 6–7.3) instead of the previous ad-hoc cosine-domain
`gap ≥ 0.01` clamp:

- `enforce_lsp_stability(lsp_cos, Δ_min_hz)` converts the cosine LSPs
  back to angular frequencies via `acos`, scans for the first pair
  `(ω_j, ω_{j+1})` with `ω_{j+1} − ω_j < Δω_min`, spreads it around its
  midpoint by `±Δω_min/2`, and iterates up to
  `LSP_STABILITY_MAX_ITERATIONS = 10` passes — exactly the procedure
  prescribed by §2.6.
- `Δω_min` is `2π · Δ_min_hz / SAMPLE_RATE_HZ`, with
  `Δ_min = 31.25 Hz` (`LSP_STABILITY_DELTA_MIN_HZ`) for the normal
  decode path.
- The frame-erasure / SID concealment path now applies the same
  procedure with the spec's wider `Δ_min = 62.5 Hz`
  (`LSP_STABILITY_DELTA_MIN_ERASURE_HZ`) per §3.10.1, pulling the
  extrapolated previous-frame LSP back into a stable configuration
  when repeated erasures drift its pairs closer together.

Round-trip PSNR is unchanged at the headline figures below; the
stability check is a numeric-discipline change in the LSP→LPC link,
not a quality knob. Five new unit tests pin the procedure (no-op on
already-stable input; one-pass convergence and midpoint-spread shape
on a single inversion; erasure variant widens beyond the normal
variant when the input violates the wider floor; every dequantised
LSP from a probe set of indices hits the 31.25 Hz floor and stays
strictly monotone in cosine domain; degenerate all-equal input still
yields a finite LPC).

### Formant-postfilter tilt + adaptive gain scaling — G.723.1 §3.8 / 3.9 (round 229)

The decoder's tilt-compensation stage and adaptive post-filter gain scaling
now both follow the spec's per-subframe shape instead of fixed shortcuts:

- **§3.8 tilt-compensation (eq. 49.2).** The tilt transfer
  `1 − μ · z⁻¹` is no longer applied with a constant `μ = 0.25`. Each
  subframe now computes `k = r(1)/r(0)` on the synthesis input `sy[n]`,
  smooths it across subframes by
  `k1 = (1 − POSTFILTER_TILT_SMOOTH_ALPHA)·k1_prev +
   POSTFILTER_TILT_SMOOTH_ALPHA·k` with `α = 1/4`, and uses
  `μ = POSTFILTER_TILT_BASE · k1` with `POSTFILTER_TILT_BASE = 0.25`. A
  silent subframe leaves `μ = 0` (eq. 50 degenerate path); a strongly
  low-pass synthesis subframe drives `μ` toward `≈ 0.25`. `k1` is bounded
  to `[−1, 1]` per `r(1)/r(0)` Cauchy-Schwarz.
- **§3.9 adaptive gain scaling (eq. 50–52).** The previous AGC used a
  per-sample chase with `α = 0.85` toward `sqrt(e_in/e_out)`. The spec
  shape replaces both legs: per subframe `g_s = sqrt(Σ sy²[n] / Σ pf²[n])`
  (set to `1` if the denominator is zero, per eq. 50); per sample the
  smoothed gain runs as a leaky integrator `g[n] = (1 − α)·g[n − 1] +
  α·g_s` with `α = POSTFILTER_AGC_ALPHA = 1/16`; the output is
  `q[n] = pf[n]·g[n]·(1 + α)` so the `(1 + 1/16)` boost undoes the
  average attenuation introduced by the integrator. `g[−1]` initialises
  to `POSTFILTER_AGC_INIT_GAIN = 1` per §3.11.

Round-trip PSNR on the integration signal improves modestly (ACELP
≈ +0.2 dB, MP-MLQ unchanged inside its ~0.01 dB measurement floor), but
the post-filter is now signal-adaptive: tilt tracks the synthesis input's
spectral tilt instead of always cutting at a fixed factor, and the AGC
follows the spec's leaky-integrator shape with the same `(1 + α)`
compensation factor. Five new unit tests pin the new behaviour
(`pf_tilt_k1` smooths across consecutive low-pass subframes and stays
inside `[−1, 1]`; zero input decays `pf_tilt_k1` by `1 − α`; silence
through the AGC stays at unity gain; the AGC's per-sample leaky
integrator matches the closed-form `g[N − 1] = g₀ + (g_s − g₀)·(1 − (1 − α)^N)`
after one subframe).

### Frame-erasure concealment — G.723.1 §3.10.2 (round 222)

The decoder's frame-erasure path (triggered by `0b10` SID and `0b11`
untransmitted discriminators) now follows the spec's two-stage
interpolation rather than the previous fixed-decay-plus-random scheme:

- A trailing 120-sample window of post-filtered output is kept across
  good frames (`ERASURE_CLASSIFIER_HISTORY_LEN = 120`) along with the
  saved third-subframe lag `L_2` and the average fixed-codebook gain
  over subframes 2 and 3.
- On erasure, `SynthesisState::classify_erasure_voicing` cross-
  correlates the trailing window with itself shifted by `L_2 ± 3`
  (`ERASURE_CLASSIFIER_LAG_RADIUS = 3`) and converts the best ratio
  `C² / (E_lag · T_en)` to a prediction gain in dB. Above
  `ERASURE_VOICED_THRESHOLD_DB = 0.58 dB` the trailing window is
  deemed *voiced* and concealment regenerates a periodic excitation at
  the classifier's pitch period through the adaptive codebook with the
  fixed-codebook contribution suppressed; below the threshold the
  trailing window is *unvoiced* and concealment regenerates a uniform
  pseudo-random excitation scaled by the saved average gain.
- Sustained erasure attenuates the regenerated excitation by an extra
  `ERASURE_ATTENUATION_DB_PER_FRAME = 2.5 dB` per consecutive erased
  frame; the run counter past `ERASURE_MUTE_AFTER_FRAMES = 3`
  produces exact silence.
- The §3.10.1 LSP-extrapolation stability pass with the wider
  `Δ_min = 62.5 Hz` floor continues to apply on the previous-frame LSP
  before the per-subframe LSP interpolation runs.

Two new unit tests pin the behaviour:
`erasure_classifier_distinguishes_voiced_and_unvoiced` confirms a pure
100 Hz sinusoid is reported voiced with a lag inside `80 ± 3` while
broadband-LCG noise is reported unvoiced and an empty history falls
back to unvoiced; `decode_erased_attenuation_schedule_matches_spec`
confirms the erased-run counter advances and that the first frame past
the mute threshold emits exact zero samples. Both pre-existing
integration tests (`erasure_in_middle_of_stream_is_concealed`,
`sustained_erasure_run_decays_to_silence`) continue to pass against
the new path without modification.

### Pitch (long-term) post-filter — G.723.1 §3.6 (round 211)

The pitch post-filter applied between synthesis and the formant /
tilt / AGC stages now matches the spec's §3.6 shape:

- Forward and backward cross-correlations are maximised over the
  seven-lag window `M ∈ [L − 3, L + 3]` around the reference lag `L`
  (`L = L_0` drives subframes 0 + 1; `L = L_2` drives 2 + 3, per
  §3.6 prose), per `SynthesisState::ltp_search_forward` /
  `SynthesisState::ltp_search_backward`.
- The single-side weighting `(w_f, w_b) ∈ {(0,0), (0,1), (1,0)}` is
  picked by per-side prediction gain (eq. 45–46), with the weaker
  side ignored.
- A 1.25 dB pitch-prediction-gain gate (`POSTFILTER_LTP_PRED_GAIN_DB_MIN`)
  bypasses the LTP postfilter on subframes that wouldn't benefit
  (broadband / unvoiced segments).
- The LTP weighting `γ_ltp` switches by rate per §3.6: **0.1875** for
  6.3 kbit/s MP-MLQ, **0.25** for 5.3 kbit/s ACELP, threaded through
  `Rate::{High, Low}` from each decode entry point.
- The output is energy-normalised by `g_p ≤ 1` (eq. 47), so the LTP
  comb cannot inflate the subframe energy past the synthesis input.

The decoded-PCM PSNR on the round-trip integration test is unchanged
at ~17.4 dB (ACELP) / ~20.7 dB (MP-MLQ) on the synthetic voiced
signal — the new postfilter is signal-adaptive (gates off where the
fixed-`β` predecessor would have done harm; engages with the right
γ_ltp where the predecessor was conservative). The shape now matches
the spec; future rounds can tighten gain quantisation against the
in-tree §2.14 / §2.17 Q-format codebooks for measurable bit-exactness
gains.

### Round-trip quality

On a 2 s voiced synthetic signal (150 Hz fundamental + three harmonics,
amplitude ≈ 20 000 on S16) encoded and decoded through this crate end-
to-end (release build, x86_64):

|    rate | frame size | PSNR    |
| ------: | ---------: | :------ |
| 5.3 k/s |   20 bytes | ≈ 17.6 dB |
| 6.3 k/s |   24 bytes | ≈ 20.7 dB |

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

### Typed accessor primitives + deeper invariants (round 265)

The `spec_tables` module now exposes typed accessor helpers on top of
the raw arrays:

- `LspBand` (`Band0` / `Band1` / `Band2`) with `start_and_length()`
  pulling each band's `(start, dim)` pair out of `LSP_BAND_INFO`;
  `lsp_codebook_entry(band, idx)` slicing one codeword row of the
  correct dimension out of the 3-band split VQ.
- `SpecRate::{High, Low}` driving `adaptive_codebook_gain_row` (returns
  the 20-sample row, `None` past the rate-specific row count) and
  `taming_gain` (returns the published i16 entry, `None` past the
  table).
- `fixed_codebook_gain(idx)` surfacing the 24 published levels.
- `mpmlq_combinatorial(row, col)` exposing the C(n, k) table as a
  typed 2-D lookup with bounds checks; `mpmlq_pulse_count(subframe)`
  / `mpmlq_max_position(subframe)` returning the per-subframe published
  values.

Fourteen new unit tests pin previously-unverified structural invariants
of the published data alongside the accessor behaviour:

- LSP DC-predicted frequencies are strictly increasing and bounded in
  Q15.
- The perceptual-weighting pole table is exactly halving
  (`p[i] = p[i − 1] / 2`).
- The postfilter pole table is a 3/4-geometric sequence (matched to
  within ±1 Q15 unit of rounding).
- The postfilter zero table is strictly decreasing positive.
- The fixed-codebook gain codebook is log-spaced with a per-step
  ratio inside `[1.30, 1.70]` after the first few Q15-rounded entries,
  spanning >1000× across the 24 levels.
- The MP-MLQ combinatorial table satisfies the Pascal-rule recurrence
  `T[r][c] = T[r][c+1] + T[r+1][c+1]` across its positive-support
  window — confirming it is a contiguous binomial table.
- Both taming-gain tables (5p3 and 6p3) are monotonically
  non-decreasing with the published 1024 Q-unit floor.
- `LspBand::ALL` covers indices 0..10 contiguously.

Total lib-test count: **85** (up from 71).

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

## Fuzzing (round 236)

A single `cargo-fuzz` target lives under `fuzz/fuzz_targets/decode.rs`
and exercises the attacker surface of the registered G.723.1 decoder.
The target drives attacker-supplied bytes through `Decoder::send_packet`
as a sequence of up to 16 variable-length packets (capped at 64 B each
to bound per-iteration allocation), with the packet size deterministic-
ally drawn from the spec-legal `{0, 1, 4, 20, 24}` ladder plus an
attacker-chosen length. Each packet's first body byte is fed verbatim
so the 2-bit rate discriminator (`00` high / `01` low / `10` SID / `11`
untransmitted, per G.723.1 §3.7) is attacker-controlled, and the
decoder's cross-packet state machine — `pending` VecDeque, `next_pts`
advance, `drained` flag, `SynthesisState`, frame-erasure run counter
(§3.10.2), formant-postfilter / AGC memory (§3.9) — is forced through
the discriminator transitions a single-rate harness never reaches.
`flush()` and `reset()` are also injected mid-stream at deterministic
hook points so the post-flush `Eof` path and the silence re-seed
behaviour are covered.

The contract under test is purely panic-freedom on every
`Decoder` entry point. Output frames are discarded — PCM-shape sanity
remains the domain of the integration test (`tests/codec_roundtrip.rs`)
and the bench harness.

Run with a nightly toolchain:

```bash
cargo fuzz run decode
```

Headline coverage on the empty corpus: ~200 000 runs in 13 s on
macOS aarch64, no crashes, no leaks. The recommended dictionary
libFuzzer emits at the end of a short session features the four rate
discriminators in the low 2 bits of the first body byte, confirming
the target is steering the input toward the rate-dispatch surface
rather than getting stuck on a single branch.

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
- Erasure / SID concealment in the decoder now follows G.723.1 §3.10.2
  (voiced/unvoiced classifier, periodic-vs-random regenerated
  excitation, 2.5 dB/frame attenuation, mute after 3 frames). Annex A
  SID parameter parsing (encoder-side comfort-noise descriptor
  generation, decoder-side noise-fill seeding from saved SID
  parameters) is still future work; raw silence is no longer the
  fallback.

## License

MIT — see [LICENSE](LICENSE).
