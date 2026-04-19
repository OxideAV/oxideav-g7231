# oxideav-g7231

Pure-Rust **ITU-T G.723.1** dual-rate narrowband speech codec — encoder
for both 6.3 kbit/s (MP-MLQ) and 5.3 kbit/s (ACELP), plus a framing-
aware decoder. No C libraries, no FFI, no `*-sys` crates.

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

The LSP split-VQ, gain VQ and lag encoding here are locally consistent
but **not** bit-compatible with ITU-T Tables 5/7/9 — a bitstream
produced by this encoder will not decode cleanly on a reference
(e.g. C-language) G.723.1 decoder. Encoder output does decode cleanly
through this crate's reference inverses (`encoder::decode_acelp_local`
and `encoder::decode_mpmlq_local`) and is verified by round-trip tests
that assert finite output with non-trivial reconstructed energy on
synthetic voiced input.

### Decoder (shipped `Decoder` impl)

The registered decoder today validates packet framing — rate
discriminator, advertised payload length, SID / untransmitted handling —
and emits 240-sample S16 silence frames with monotonic PTS. This is
the framing contract any future full-synthesis decoder must satisfy.
The LSP-VQ lookup, adaptive/fixed-codebook excitation reconstruction,
MP-MLQ pulse decoding, post-filter and comfort-noise generation are
deliberately stubbed.

For local round-trip testing of the encoder against its own analysis
pipeline, use `encoder::decode_acelp_local` or
`encoder::decode_mpmlq_local` — these are kept `pub` exactly for that
purpose.

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

- Encoder, both rates: implemented, round-trip-verified against the
  crate's own reference decoders.
- Decoder: framing-only (silence PCM out). Full synthesis path is
  future work; the module is laid out (bitreader, LSP tables, 10th-
  order synthesis filter) so the DSP blocks can land incrementally.
- No VAD / CNG in the encoder — every frame is coded as speech.

## License

MIT — see [LICENSE](LICENSE).
