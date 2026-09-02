# oxideav-g7231

[![CI](https://github.com/OxideAV/oxideav-g7231/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-g7231/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-g7231.svg)](https://crates.io/crates/oxideav-g7231) [![docs.rs](https://docs.rs/oxideav-g7231/badge.svg)](https://docs.rs/oxideav-g7231) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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

The wire format is the **ITU-T clause-4 spec layout** at both rates:
Table 5 (high rate, 24 octets) / Table 6 (low rate, 20 octets) octet
maps carrying the published quantiser indices.

### Encoder (both rates)

The full clause-2 analysis pipeline on the published tables, packed
into the clause-4 octet layout: §2.2 framer (the input is coded
delayed by one subframe — the spec's 7.5 ms lookahead, 37.5 ms total
delay) → §2.3 high-pass (fixed point, r455) → §2.4 per-subframe
windowed LPC → §2.5 LSP quantisation → §2.8 formant weighting → §2.9
open-loop pitch → §2.11 harmonic noise shaping → §2.12/§2.13
combined-filter target → §2.14 closed-loop adaptive codebook →
§2.15/§2.16 fixed codebook → eq. 36/39/40 gain words → §2.19 memory
update — plus the **Annex A** silence compression (VAD + SID frames)
on request. Default rate (no `bit_rate` hint) is 6.3 kbit/s MP-MLQ;
request `Some(5300)` for ACELP. The §2.3 DC-removal filter (eq. 1)
defaults ON per §2.2; `encoder::SpecEncoder` exposes the rate,
high-pass and VAD switches the ITU encoder-test configurations
require (`set_rate` may change the rate at any frame boundary).

- **Analysis by synthesis**: the encoder carries a shadow
  `SynthesisState` committed through the *exact* decode kernel, so
  analysis always targets what the decoder will actually produce.
- **§2.3 high-pass** (r455, first stage on the saturating fixed-point
  chain): Word32 recursion state holding the half-scale output in Q16,
  emitted as `round16` on the Word16 rail — the half-scale analysis
  domain the decoder's doubled pulse amplitudes already implied.
- **LPC analysis** (§2.4): four LPC sets per frame, each from the
  published Q15 Hamming window over 180 samples centered on its
  subframe, `1025/1024` white-noise correction, published binomial
  lag window, Levinson-Durbin (also yielding `k[2]` for the Annex A
  sine detector and the corrected autocorrelation for COD-CNG).
- **LSP quantisation** (§2.5): the 7.5 Hz bandwidth-expanded A3(z)
  through a predictive 3+3+4 split VQ over the published 256-entry
  band codebooks, MA predictor `b = 12/32`, DC removal, and the eq. 5
  inverse-neighbour-gap weighted error.
- **Weighted-domain targets** (§2.8–§2.13): per-subframe formant
  weighting `W(z) = A(z/0.9)/A(z/0.5)` on the unquantised LPC
  (published Q15 tap weights); two half-frame open-loop pitch
  estimates on the weighted speech (eq. 12 over the **positive**
  correlations only, with the smaller-lag 1.25 dB preference — r455,
  vector-arbitrated); harmonic noise shaping `P(z) = 1 − β·z^−L` gated
  by the eq. 17 2.0 dB prediction-gain test; and ringing subtraction
  of the combined filter `S(z) = Ã(z)·W(z)·P(z)`.
- **Closed-loop pitch** (§2.14): lag candidates around the §2.9
  open-loop estimate (±1 on subframes 0/2, the −1..+2 delta window on
  1/3) jointly searched with the published 85-/170-row gain-vector
  codebook — each 20-entry row `[β_i, −β_i²/2, −β_iβ_j/2]` (Q13) is
  applied *verbatim* as one dot product against the filtered-basis
  correlations (r455).
- **MP-MLQ fixed-codebook search** (§2.15): eq. 24/25 `G_max`
  estimate on the 24-level table, the vector-arbitrated five-level
  gain neighbourhood × both grids × the short-lag Dirac-train mode,
  greedy sequential pulse placement, each candidate scored at its own
  level; positions transmitted as `C(30,M)` combinatorial codes with
  the 13-bit `MSBPOS` word.
- **ACELP fixed-codebook search** (§2.16): the Recommendation's own
  procedure (r455) — eq. 28 `d[j]` and eq. 29 even-position
  covariance of the pitch-enhanced impulse response, eq. 32 sign
  folding (on magnitudes), four nested Table 1 track loops with the
  odd grid on the even-shifted energy, the eq. 35 focused-search
  threshold with the 600-entries-per-frame cap, `argmax C²/ε`, and the
  last-step gain `min |G − G̃_j|`.
- **Gain words** (eq. 36/39/40): combined 12-bit
  `PGIndex·24 + MGIndex` words, with the high-rate short-lag (L < 58)
  85-row layout carrying the impulse-train bit in the MSB.
- **Annex A silence compression** (r455, `SpecEncoder::set_vad`):
  the A.2 VAD (adaptation-enable flag from the open-loop lags and the
  `k[2] ≥ 0.95` sine detector, noise-inverse-filtered energy,
  slow-attack / fast-decay noise level in `[128, 131071]`, the A-5
  logarithmic threshold, 6-frame hangover after bursts ≥ 2 frames),
  the A.4 COD-CNG decision (first inactive frame → SID; otherwise SID
  when the Itakura distance to the SID filter exceeds `thr1 = 1.2136`
  or the coded energy moves by more than `thr2 = 3` levels), the
  A.4.4 SID filter (current vs. three-frame past-average LPC) through
  the §2.5 quantiser, the A.4.3 6-bit pseudo-log gain quantiser
  (segment bases `{0, 32, 96}` and squared boundaries from the staged
  tables), Table A.1 packing (4-octet SID, 1-octet untransmitted),
  and the A.4.5 comfort-noise excitation feeding the local decoder.
  The annex does not define its random generator (`rseed = 12345`),
  so this crate uses its own documented LCG: the comfort noise, and
  hence the active frames following a silence, are not bit-exact.
  The registry encoder keeps the VAD off (every frame coded as
  speech); the decoder conceals SID / untransmitted frames per §3.10
  rather than synthesising Annex A comfort noise (follow-up).

### Decoder (stateful, full-synthesis, fixed-point)

The registered `Decoder` ships the r391 **saturating fixed-point
pipeline** (`qdec::QSynthesis`): Q15 LSP domain, Q14 cosine lookup →
Q13 LPC, wide-accumulator excitation reconstruction, and integer
post-filters, built on the `basicop` saturating operator layer. The
§3.1 chain on the published tables:

- Clause-4 unpack (Table 5/6, MSBPOS split, field validation), then
  §3.2 LSP decode (MA predictor + DC + split rows, §2.6 stability with
  the previous-vector fallback), eq. 37/38 lag decode, the eq. 39/40
  gain-word split (85-row rule keyed off the subframe pair's reference
  lag), the eq. 41.1–41.2 **fifth-order adaptive codebook**, and the
  rate-specific fixed-codebook reconstruction (MP-MLQ combinatorial +
  Dirac trains; ACELP Table 1 + the §2.16 1-tap pitch enhancement).
- **Post-filter chain in the §3.1 order**: the §3.6 pitch (long-term)
  post-filter runs on the **whole-frame decoded excitation**
  (eq. 42–47: forward/backward search in `[L±3]`, the forward-reach
  availability rule, the 1.25 dB prediction-gain gate, `g = C/D`
  weighted by the rate-specific γ_ltp, the attenuate-only eq. 47
  `g_p`), its output feeds the §3.7 synthesis filter, and §3.8/§3.9
  run on the synthesis output: formant filter `A(z/γ₁)/A(z/γ₂)` on the
  per-subframe interpolated LPC with the exact Q15 §2.18 tap tables,
  signal-adaptive tilt compensation, leaky-integrator adaptive gain
  scaling. `SynthesisState::set_postfilter(false)` disables the chain
  (the ITU decoder-test configurations require the switch).
- **Frame-erasure concealment** (§3.10): voiced/unvoiced classifier,
  periodic or pseudo-random regeneration, 2.5 dB/frame attenuation,
  mute after 3 frames; §3.10.1 LSP extrapolation toward `p_DC` with
  the erasure predictor `b_e = 23/32` and the wider 62.5 Hz ordering
  floor. §3.11 cold start (previous LSP = `p_DC`, AGC gain = 1).
- SID and untransmitted frames feed the concealment path; Annex A SID
  parsing / CNG is future work (Annex A is not in the 1996 base
  edition).

### Round-trip quality

On 2 s of voiced synthetic speech (180 Hz fundamental + three
harmonics) through the full encode → decode pipeline (release,
peak-referenced PSNR):

|    rate | frame size | PSNR      | signal SNR |
| ------: | ---------: | :-------- | :--------- |
| 5.3 k/s |   20 bytes | ≈ 30.8 dB | ≈ 18.8 dB  |
| 6.3 k/s |   24 bytes | ≈ 31.1 dB | ≈ 19.1 dB  |

See `tests/codec_roundtrip.rs` for the integration tests. For a
playable subjective sample:

```bash
cargo test --release -- --ignored roundtrip_writes_sample_raw
aplay -f S16_LE -c 1 -r 8000 /tmp/g7231-sample.raw
```

## Bitstream interoperability

Frames follow the Recommendation's clause-4 octet maps (Tables 5/6)
over the published quantiser tables — and as of r388 the wire format
is **verified against the official ITU conformance vectors**
(`docs/audio/g7231/conformance/` in the OxideAV umbrella): all 2 816
frames of the 13 main-body reference bitstreams unpack and repack
byte-identically through [`linepack`](src/linepack.rs), so the
crate's MSBPOS mixed-radix combine (subframe-major, most significant
first), `C(30,M)` combinatorial position codec and field layout are
the reference's own. The three deliberate transmission-error frames
in `PATHD63P.TCO` correctly fail field validation and are concealed
as erasures. Decoded pulse positions were additionally verified
against Â(z)-deconvolved reference decoder output.

The intra-word conventions are pinned by the vectors: high-rate
`PSIG` stores signs MSB-first over ascending pulse order with a set
bit meaning **negative**; low-rate `PSIG` bit `t` is the track-`t`
sign with a set bit meaning **positive**; and the 24-bit `LPC` word
carries **band 0 in its most-significant byte** (r406 — with the
LSB-first reading, every reference stream decodes its two edge LSP
bands through the wrong codebooks).

Where the clause 1–4 prose leaves a model choice open, it is
arbitrated against the vectors (the clause-5 reference C stays
outside the clean-room wall). Current model (r391, re-arbitrated r406
after the LSP band-order fix): the eq. 41.1 `e′` view is the
contiguous history slice from `e[−L−2]`; fixed-codebook pulses land
at twice the published gain-table level with the synthesis output
emitted **unshifted** (whole-file least-squares scale vs the
reference = 0.9995; r391's halved-output stage compensated the
band-swapped LSP distortion); gain-vector rows act at an effective
/16384; the stored excitation saturates at the plain Word16 rail; and
the §2.2 framer codes the input **delayed by one subframe** — the
7.5 ms lookahead alignment at which (and only at which) the encoder's
LSP decisions lock to the reference.

Measured whole-file decoder tracking against the reference `.ROU`
outputs (r406 fixed pipeline, corr / SNR): **PATHD53 1.0000 /
+54.4 dB** (max sample error 27 LSB), OVERD53 0.9993 / +28.1 dB,
INEQD53 0.9811 / +12.5 dB (50.8% of samples bit-exact), PATHD63P
0.9123 / +7.6 dB, OVERD63P 0.9715 / +12.5 dB, TAMED63P 0.9643 /
+11.5 dB. What remains between this and a *bit-exact* claim is the
reference's per-stage rounding and overflow protocol (dominant on the
post-filter-ON and OVER/TAME saturation-torture classes), which the
Recommendation specifies only in the clause-5 C.

Encoder parameter agreement against the reference `.RCO` bitstreams
(r455; the teacher-forced columns score each stage from the
reference's own state, the free-running column the whole stream):

| vector   | LSP word (forced / free) | ACL0/2 exact (forced) | PGIndex (forced) | FCB subframe exact (forced) | MG exact (free) |
|----------|--------------------------|-----------------------|------------------|-----------------------------|-----------------|
| CODEC63  | 89.5 / 77.0%             | 95.2%                 | 91.0%            | 55.6%                       | 65.3%           |
| PATHC63H | 97.0 / 94.1%             | 73.3%                 | 86.0%            | 57.2%                       | 63.6%           |
| OVERC63  | 90.0 / 85.0%             | 87.5%                 | 76.2%            | 52.5%                       | 68.8%           |
| TAMEC63H | 53.0 / 29.0%             | 98.5%                 | 56.8%            | 16.8%                       | 29.5%           |
| PATHC53  | 93.9 / 90.4%             | 72.4%                 | 78.4%            | 69.6%                       | 73.3%           |
| INEQC53  | 46.0 / 22.2%             | 94.4%                 | 25.8%            | 60.3%                       | 29.0%           |
| OVERC53H | 95.2 / 90.5%             | 95.2%                 | 81.0%            | 60.7%                       | 69.0%           |

Free-running ACL0/ACL2 within ±1: 84–100% on every vector (PATHC63H
53.5 → 85.8%, TAMEC63H 49 → 99% in r455). What the r455 campaign
established, with every decision arbitrated against the vectors: the
§2.9 search counts positive correlations only; the §2.16 ACELP search
must be the printed focused nested-loop procedure with magnitude sign
folding (whole-subframe agreement on PATHC53 27 → 70%); the §2.15
gain neighbourhood is five levels around the nearest level to the
estimate in this crate's doubled excitation domain; the §2.14 rows
act verbatim; and the §2.3 output lives on a half-scale Word16 rail
with a wide recursion state. What remains is precision-level: the
LSP-word mismatches are near-ties (the reference's index is our
second-best at a 4–10% error margin, and the INEQ / TAME / sine
segments cycle through rows the exact fixed-point form of the eq. 4.5
weighted error decides), the remaining lag / gain-vector mismatches
are rank-1 near-ties, and TAME's gain vectors follow the encoder
taming procedure the Recommendation text does not describe. The
Annex A frame-type sequence tracks the DTX vectors at 94.4% (DTX63),
93.3% (DTX53) and 93.3% (DTXMIX, per-frame rate schedule), with the
coded SID gain within ±1 of the reference's on 96–100% of common SID
frames. See [`tests/itu_conformance.rs`](tests/itu_conformance.rs)
for all pinned floors.

### Spec-table data in tree

The 27 ITU-T G.723.1 normative numeric tables are exposed in
[`spec_tables`](src/spec_tables.rs) as `static` arrays of `i16` / `u32`
in their published Q-formats, each citing its source CSV under
`docs/audio/g7231/tables/` and the data SHA-256. On top of the raw
data sit typed accessors plus three codec layers that now **drive the
codec**:

- [`linepack`](src/linepack.rs) — clause-4 Table 5/6 frame
  pack/unpack + the 13-bit MSBPOS combine.
- [`spec_lsp`](src/spec_lsp.rs) — the §2.5/§2.6 predictive split-VQ
  LSP codec (Q15 normalised-frequency domain).
- [`spec_exc`](src/spec_exc.rs) — eq. 36/39/40 gain words, the
  eq. 41 five-tap adaptive codebook, MP-MLQ/ACELP fixed-vector
  reconstruction, and the §2.16 pitch enhancement. The `C(30,M)`
  combinatorial position codec (`fcbk_pack_positions` /
  `fcbk_unpk_positions`) is exhaustively verified bijective over both
  codeword spaces (`C(30,5) = 142 506`, `C(30,6) = 593 775`).
- [`basicop`](src/basicop.rs) — the saturating Word16/Word32 DSP
  operator layer (r391), and [`qdec`](src/qdec.rs) — the fixed-point
  §3.1 decode pipeline the registry decoder ships.

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
registry encoder codes every frame as speech; Annex A silence
compression is available on `encoder::SpecEncoder::set_vad`. Per §2.2
the coded signal is the input delayed by one subframe (7.5 ms — the
spec's lookahead), so decoded sample `n` renders input sample
`n − 60`.

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

Five `cargo-fuzz` targets live under `fuzz/fuzz_targets/`, each an
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
- **`params`** — the `make_encoder` / `make_decoder` parameter-
  validation surface (rejected sample rates / channel counts / sample
  formats, 5.3 / 6.3 kbit/s band edges) plus a sustained SID /
  untransmitted erasure-concealment decay→recovery→reset drive.
- **`dtx`** — Annex A: arbitrary PCM through `SpecEncoder` with the
  VAD on and a fuzzer-driven per-frame rate schedule, every active /
  SID / untransmitted packet back through the fixed-point decoder.

A version-controlled seed corpus shaped to each target's input layout
lives under `fuzz/seeds/<target>/`; see `fuzz/README.md`.

```bash
cargo fuzz run {decode,roundtrip,bitstream,params,dtx}   # nightly toolchain
```

## License

MIT — see [LICENSE](LICENSE).
