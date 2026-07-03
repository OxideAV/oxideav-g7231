//! ITU-T G.723.1 dual-rate (6.3 / 5.3 kbit/s) speech codec.
//!
//! Pure-Rust encoder + decoder covering both rates of G.723.1:
//!
//! - 6.3 kbit/s MP-MLQ (24-byte frames, discriminator `00`)
//! - 5.3 kbit/s ACELP  (20-byte frames, discriminator `01`)
//! - SID / untransmitted framing handled by a shared erasure concealment
//!   path (attenuated gains, extrapolated lag, pseudo-random innovation)
//!
//! # Encoder pipeline
//!
//! For each 30 ms frame of 240 S16 samples at 8 kHz:
//!
//! ```text
//!  PCM → LPC (autocorr + Levinson + lag window)
//!      → LSP conversion (Chebyshev root-finding)
//!      → §2.5 predictive split VQ on the published 3-band codebooks
//!        (24-bit LPC word, MA predictor b = 12/32, eq. 5 weights)
//!      → 4 × subframe loop:
//!            open-loop pitch on the ZIR-subtracted synthesis target
//!          → §2.14 closed-loop lag (±1 / −1..+2 windows) jointly
//!            searched with the 85-/170-row 5-tap gain-vector codebook
//!          → rate-specific FCB search at quantised §2.15 gain levels
//!               * ACELP: §2.16 Table 1 tracks + grid against the
//!                 pitch-enhanced impulse response
//!               * MP-MLQ: greedy multipulse over the Ĝ ± neighbourhood,
//!                 both grids, and the short-lag Dirac-train mode
//!          → eq. 36/39/40 combined 12-bit gain word
//!      → canonical decode_spec_params() pass commits decoder state
//!      → clause-4 Table 5/6 octet packing (24 B rate=00 / 20 B rate=01)
//! ```
//!
//! The encoder's analysis loop runs against an internal [`encoder::SynthesisState`]
//! that mirrors the real decoder, so encoder and decoder reconstruct
//! sample-identical PCM from the same bitstream.
//!
//! # Decoder
//!
//! [`Decoder::send_packet`] dispatches on the rate discriminator and
//! routes ACELP / MP-MLQ payloads through the matching
//! [`encoder::SynthesisState`] entry points (clause-4 unpack + the §3.1
//! spec-table pipeline). The chain is
//! synthesis → pitch post-filter → formant post-filter
//! (A(z/γ₁)/A(z/γ₂)) → first-order tilt compensation → smoothed AGC.
//! SID / untransmitted packets feed [`encoder::SynthesisState::decode_erased`],
//! which extrapolates the last lag, attenuates the gains on a per-frame
//! schedule, and seeds a pseudo-random innovation so repeated loss decays
//! without audible clicks.  Decoder state persists across packets and
//! `reset()` reinitialises to silence.
//!
//! # Wire format
//!
//! Frames are the ITU-T clause-4 spec layout: the Table 5 (high rate) /
//! Table 6 (low rate) octet maps over the published quantiser tables —
//! see [`linepack`] for the packing layer and the README for the two
//! remaining documented derivation choices (MSBPOS digit order and the
//! intra-word pulse/sign bit conventions) plus the conformance-vector
//! caveat.
//!
//! Reference: ITU-T Recommendation G.723.1 (03/96).

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod encoder;
pub mod header;
pub mod linepack;
pub mod spec_exc;
pub mod spec_lsp;
pub mod spec_tables;
pub mod tables;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, CodecTag, Error, Frame, Packet, Result,
};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder};

use crate::header::{parse_frame_type, FrameType};
use crate::tables::{FRAME_SIZE_SAMPLES, SAMPLE_RATE_HZ};

pub const CODEC_ID_STR: &str = "g723_1";

pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("g723_1_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(SAMPLE_RATE_HZ);
    // AVI / WAVEFORMATEX tag: WAVE_FORMAT_MSG723 = 0x0014.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(encoder::make_encoder)
            .tag(CodecTag::wave_format(0x0014)),
    );
}

/// Unified registration entry point — installs G.723.1 into the codec
/// sub-registry of the supplied [`oxideav_core::RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut oxideav_core::RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("g7231", register);

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(G7231Decoder::new()))
}

/// Full-synthesis G.723.1 decoder. Dispatches on the 2-bit rate
/// discriminator in the first payload byte:
///
/// - `00` → 6.3 kbit/s MP-MLQ, routed through [`encoder::SynthesisState::decode_mpmlq`]
/// - `01` → 5.3 kbit/s ACELP,  routed through [`encoder::SynthesisState::decode_acelp`]
/// - `10` → SID (silence-insertion descriptor) — handled by the same
///          concealment path as erasures ([`encoder::SynthesisState::decode_erased`])
///          since we do not yet parse SID parameters
/// - `11` → Untransmitted (erasure) — [`encoder::SynthesisState::decode_erased`]
///          extrapolates the last frame with attenuated gains and a
///          pseudo-random innovation so repeated loss fades smoothly
///
/// State (excitation history, previous-frame LSP, synthesis filter
/// memory, post-filter memory) persists across packets via
/// [`encoder::SynthesisState`] so a steady stream of packets decodes
/// without per-frame cold-start transients. `reset()` reinitialises the
/// synthesiser to silence.
struct G7231Decoder {
    codec_id: CodecId,
    synthesis: encoder::SynthesisState,
    pending: std::collections::VecDeque<Frame>,
    drained: bool,
    next_pts: i64,
}

impl G7231Decoder {
    fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            synthesis: encoder::SynthesisState::new(),
            pending: std::collections::VecDeque::new(),
            drained: false,
            next_pts: 0,
        }
    }

    fn audio_frame_from_pcm(&self, pcm: &[i16; FRAME_SIZE_SAMPLES], pts: Option<i64>) -> Frame {
        let mut bytes = Vec::with_capacity(FRAME_SIZE_SAMPLES * 2);
        for &s in pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        // Stream-level shape (S16 / mono / 8 kHz / time_base) lives on
        // the stream's CodecParameters now, not per-frame. The
        // decoder's `output_params` (synthesised in `Decoder::params`)
        // surfaces it.
        Frame::Audio(AudioFrame {
            samples: FRAME_SIZE_SAMPLES as u32,
            pts,
            data: vec![bytes],
        })
    }
}

impl Decoder for G7231Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::invalid("G.723.1: empty packet"));
        }
        let frame_type = parse_frame_type(&packet.data)?;
        let expected = frame_type.frame_size();
        match frame_type {
            FrameType::Untransmitted => {}
            _ if packet.data.len() < expected => {
                return Err(Error::invalid(format!(
                    "G.723.1: {} needs {} bytes, got {}",
                    frame_type.bit_rate_label(),
                    expected,
                    packet.data.len(),
                )));
            }
            _ => {}
        }

        let pts = packet.pts.or(Some(self.next_pts));
        self.next_pts = pts.unwrap_or(self.next_pts) + FRAME_SIZE_SAMPLES as i64;

        let frame = match frame_type {
            FrameType::HighRate => {
                // A size-valid frame whose *content* fails validation
                // (e.g. an out-of-range pulse-position code from
                // transmission errors — the PATHD63P conformance vector
                // carries three such frames) is concealed like an
                // erasure (§3.10) instead of killing the stream.
                let pcm = self
                    .synthesis
                    .decode_mpmlq(&packet.data)
                    .unwrap_or_else(|_| self.synthesis.decode_erased());
                self.audio_frame_from_pcm(&pcm, pts)
            }
            FrameType::LowRate => {
                let pcm = self
                    .synthesis
                    .decode_acelp(&packet.data)
                    .unwrap_or_else(|_| self.synthesis.decode_erased());
                self.audio_frame_from_pcm(&pcm, pts)
            }
            FrameType::SidFrame | FrameType::Untransmitted => {
                // Frame erasure / comfort-noise concealment: reuse the
                // previous-frame LSP, extrapolate the last pitch lag,
                // attenuate the gains per frame of the erased run, and
                // drive the synthesis with a pseudo-random innovation so
                // there's no audible click at the seam. After ~5 erased
                // frames in a row the concealment fades to silence.
                let pcm = self.synthesis.decode_erased();
                self.audio_frame_from_pcm(&pcm, pts)
            }
        };
        self.pending.push_back(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(f);
        }
        if self.drained {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.drained = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.synthesis.reset();
        self.pending.clear();
        self.drained = false;
        self.next_pts = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::packet::PacketFlags;
    use oxideav_core::{Rational, TimeBase};

    fn packet(data: Vec<u8>) -> Packet {
        Packet {
            stream_index: 0,
            time_base: TimeBase(Rational::new(1, SAMPLE_RATE_HZ as i64)),
            pts: None,
            dts: None,
            duration: None,
            flags: PacketFlags::default(),
            data,
        }
    }

    #[test]
    fn registers_in_registry() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(reg.has_encoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = oxideav_core::RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&id),
            "decoder factory not installed via RuntimeContext"
        );
        assert!(
            ctx.codecs.has_encoder(&id),
            "encoder factory not installed via RuntimeContext"
        );
    }

    #[test]
    fn decodes_high_rate_frame_shape() {
        // All-zero 24-byte high-rate payload is valid framing; the
        // decoder runs the full MP-MLQ synthesis and emits a 240-sample
        // S16 frame. The output samples are small but not strictly zero
        // (the zero bitstream dequantises to a default LSP + near-zero
        // gains + first-pulse-slot pulses) — here we only assert frame
        // shape.
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0u8; 24]);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        // Stream-level shape (sample_rate / channels / format) used to
        // be on the frame; with the slim it lives on the stream's
        // `CodecParameters`. The factory builds those off
        // `params_for_codec()` (8 kHz mono S16) — checked separately
        // below. Here we only assert the per-frame sample contract.
        assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), FRAME_SIZE_SAMPLES * 2);
    }

    #[test]
    fn decodes_low_rate_silence() {
        let mut dec = G7231Decoder::new();
        // Discriminator bits = 01 → low rate, 20 bytes.
        let mut data = vec![0u8; 20];
        data[0] = 0b01;
        let pkt = packet(data);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        // The pre-slim assertion was
        // `f.time_base().as_rational().den == SAMPLE_RATE_HZ`. With
        // the slim there's no per-frame time_base and no
        // `time_base()` accessor. The semantic equivalent is "the
        // decoder produces a 240-sample frame at 8 kHz" — checked
        // here via `samples` (which the decoder carries on every
        // emitted AudioFrame). The encoder-side params ctor
        // (`params_for_codec`) confirms 8 kHz / mono / S16 lives on
        // the stream's CodecParameters.
        assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
    }

    #[test]
    fn rejects_short_high_rate_frame() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0u8; 10]); // high-rate needs 24
        assert!(dec.send_packet(&pkt).is_err());
    }

    #[test]
    fn accepts_untransmitted_single_byte() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0b11]);
        dec.send_packet(&pkt).unwrap();
        assert!(matches!(dec.receive_frame().unwrap(), Frame::Audio(_)));
    }

    #[test]
    fn flush_then_eof() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0u8; 24]);
        dec.send_packet(&pkt).unwrap();
        dec.flush().unwrap();
        // One buffered frame, then EOF.
        assert!(dec.receive_frame().is_ok());
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn pts_increments_by_frame_size() {
        let mut dec = G7231Decoder::new();
        dec.send_packet(&packet(vec![0u8; 24])).unwrap();
        dec.send_packet(&packet(vec![0u8; 24])).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        assert_eq!(f0.pts(), Some(0));
        assert_eq!(f1.pts(), Some(FRAME_SIZE_SAMPLES as i64));
    }
}
