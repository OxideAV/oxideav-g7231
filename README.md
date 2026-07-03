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

The wire format is the **ITU-T clause-4 spec layout** at both rates:
Table 5 (high rate, 24 octets) / Table 6 (low rate, 20 octets) octet
maps carrying the published quantiser indices.

### Encoder (both rates)

Full §2.3 high-pass → LPC → LSP → open-loop pitch → closed-loop
adaptive-codebook → rate-specific fixed-codebook → gain-word pipeline
on the published tables, packed into the clause-4 octet layout.
Default rate (no `bit_rate` hint) is 6.3 kbit/s MP-MLQ; request
`Some(5300)` for ACELP. The §2.3 DC-removal filter (eq. 1) defaults ON
per §2.2; `encoder::SpecEncoder` exposes the rate + high-pass switches
the ITU encoder-test configurations require.

- **Analysis by synthesis**: the encoder carries a shadow
  `SynthesisState` committed through the *exact* decode kernel, so
  analysis always targets what the decoder will actually produce.
- **LSP quantisation** (§2.5): predictive 3+3+4 split VQ over the
  published 256-entry band codebooks, MA predictor `b = 12/32`, DC
  removal, and the eq. 5 inverse-neighbour-gap weighted error.
- **Closed-loop pitch** (§2.14): lag candidates around the open-loop
  estimate (±1 on subframes 0/2, the −1..+2 delta window on 1/3)
  jointly searched with the published 85-/170-row 5-tap gain-vector
  codebook by maximising the error reduction `2·βᵀd − βᵀRβ` over the
  filtered eq. 41 basis vectors.
- **MP-MLQ fixed-codebook search** (§2.15): eq. 24/25 `G_max`
  estimate, the `[Ĝ − 3.2 dB, Ĝ + 6.4 dB]` quantised-gain
  neighbourhood × both grids × the short-lag Dirac-train mode, greedy
  sequential pulse placement, and an exact 24-level MMSE gain re-pick;
  positions transmitted as `C(30,M)` combinatorial codes with the
  13-bit `MSBPOS` word.
- **ACELP fixed-codebook search** (§2.16): the Table 1 4-pulse
  stride-8 track geometry with coordinate-descent refinement, run
  against the pitch-enhanced impulse response
  (`h′[n] = h[n] + β·h′[n − L − ε]`), least-squares gain folded into
  the pulse signs and quantised on the 24-level table.
- **Gain words** (eq. 36/39/40): combined 12-bit
  `PGIndex·24 + MGIndex` words, with the high-rate short-lag (L < 58)
  85-row layout carrying the impulse-train bit in the MSB.

### Decoder (stateful, full-synthesis)

The registered `Decoder` runs the §3.1 pipeline on the published
tables:

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
| 5.3 k/s |   20 bytes | ≈ 23.9 dB | ≈ 12.0 dB  |
| 6.3 k/s |   24 bytes | ≈ 26.4 dB | ≈ 14.4 dB  |

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

The intra-word sign conventions are pinned by the vectors: high-rate
`PSIG` stores signs MSB-first over ascending pulse order with a set
bit meaning **negative**; low-rate `PSIG` bit `t` is the track-`t`
sign with a set bit meaning **positive** (flipping it takes the
whole-file `OVERD53` waveform correlation from −0.97 to +0.97).

What remains between this and a *bit-exact* conformance claim is
fixed-point arithmetic: this is a floating-point implementation of
the clause 2/3 mathematical description, while the Recommendation
makes the clause-5 fixed-point behaviour normative. Measured decoder
tracking against the reference `.ROU` outputs (r388): `OVERD53`
whole-file correlation 0.973 / mean per-frame 0.985 / SNR 7.3 dB
(post-filter OFF per the test config); `OVERD63P` cold-start frames
0–3 correlation 0.61/0.89/0.83/0.74 (post-filter ON). The `OVER..` /
`TAME..` classes deliberately drive sustained Word16-saturation
chains that only a bit-exact fixed-point pipeline tracks long-range —
that rebuild (Q15 saturating basic ops through analysis + synthesis)
is the remaining conformance work. See
[`tests/itu_conformance.rs`](tests/itu_conformance.rs) for the pinned
floors.

Bitstreams produced by this encoder decode with this crate's own
decoder at the PSNR figures above, carry spec-semantic indices
throughout, and every encoder-emitted frame on the ITU test inputs is
a legal clause-4 stream.

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

Four `cargo-fuzz` targets live under `fuzz/fuzz_targets/`, each an
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

A version-controlled seed corpus shaped to each target's input layout
lives under `fuzz/seeds/<target>/`; see `fuzz/README.md`.

```bash
cargo fuzz run {decode,roundtrip,bitstream,params}   # nightly toolchain
```

## License

MIT — see [LICENSE](LICENSE).
