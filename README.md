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
layout in this crate are a clean-room, pure-Rust design — internally
consistent and decode-quality-equivalent to a reference G.723.1 codec,
but **not** bit-compatible with ITU-T Tables 5 / 7 / 9. Bitstreams
produced by this encoder decode cleanly with this crate's own decoder
at the PSNR figures above, but not with an external, spec-table G.723.1
reference decoder.

Achieving that interoperability would mean porting the ITU-T tables
verbatim (~6-8 KB of codebook data plus the Q13/Q15 fixed-point gain
quantiser) while keeping the pure-Rust / no-FFI invariant. That's a
separate piece of work from what this crate provides today.

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase,
};

let mut reg = CodecRegistry::new();
oxideav_g7231::register(&mut reg);

let mut params = CodecParameters::audio(CodecId::new(oxideav_g7231::CODEC_ID_STR));
params.sample_rate = Some(8_000);
params.channels = Some(1);
params.sample_format = Some(SampleFormat::S16);
params.bit_rate = Some(6_300); // or Some(5_300) for ACELP

let mut enc = reg.make_encoder(&params)?;

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
