//! ITU-T G.723.1 encoder — ACELP (5.3 kbit/s) and MP-MLQ (6.3 kbit/s) paths.
//!
//! # Scope
//!
//! This module implements **both** rates of G.723.1:
//!
//! - **5.3 kbit/s ACELP** — 4 fixed-position pulses per subframe on
//!   4 tracks (T0..T3), 20-byte payload, discriminator `01`.
//! - **6.3 kbit/s MP-MLQ** — 6 pulses on odd subframes (0, 2) and
//!   5 pulses on even subframes (1, 3), 24-byte payload, discriminator `00`.
//!
//! [`make_encoder`] dispatches between the two rates based on the
//! `CodecParameters.bit_rate` hint: `Some(6300)` or unset → MP-MLQ;
//! `Some(5300)` → ACELP; any other value returns [`Error::Unsupported`].
//! The default (no hint) is 6.3 kbit/s, the more common operating rate.
//!
//! # Pipeline
//!
//! For each 30 ms frame (240 samples at 8 kHz, mono S16):
//!
//! ```text
//!  PCM s16 → LPC analysis (autocorrelation + Levinson + bandwidth-expand)
//!          → LSP conversion + split VQ quantisation (24 bits total)
//!          → 4× subframe loop:
//!                - open-loop pitch from weighted residual
//!                - closed-loop adaptive-codebook gain
//!                - rate-specific fixed-codebook search
//!                    · ACELP:  4-pulse search on T0..T3 tracks
//!                    · MP-MLQ: greedy 6/5-pulse search on the grid
//!                - joint gain quantisation (12-bit combined index)
//!          → bit-pack 158 bits (ACELP, 20 B, rate=01)
//!               or 192 bits (MP-MLQ, 24 B, rate=00)
//! ```
//!
//! ## Departures from the letter of the spec
//!
//! - The LSP split-VQ here uses a small, self-consistent training-derived
//!   codebook — NOT the ITU-T Table 5 codebook. A bitstream produced by this
//!   encoder therefore cannot be decoded by an external (e.g. reference-C)
//!   G.723.1 decoder for high-quality speech. It IS, however, internally
//!   consistent with the [`decode_acelp_local`] / [`decode_mpmlq_local`]
//!   helpers provided here (used by the tests for round-trip verification)
//!   and passes the framework's scaffold decoder (which emits silence).
//! - Open-loop pitch search is on the weighted short-term residual,
//!   covering `[PITCH_MIN..=PITCH_MAX]` as the spec mandates; refinement
//!   within ±1 is done by integer-lag re-correlation rather than the spec's
//!   fractional-lag search.
//! - MP-MLQ pulse search is a pure greedy per-pulse residual-minimiser on
//!   an 8-slot track per pulse (3-bit position + 1-bit sign); a shared
//!   subframe gain is quantised together with the ACB gain via the same
//!   12-bit codeword as ACELP.
//! - Gain quantisation packs a 3-bit ACB gain index + 9-bit FCB gain
//!   exponent/mantissa into a 12-bit combined word — this fills the
//!   GAIN field exactly but uses a locally-chosen mapping rather than the
//!   spec's Table 7.
//!
//! These deliberate simplifications keep the encoder pure-Rust, ~1000 LOC,
//! and bit-exact with its own reference decode, while still exercising the
//! full analysis / packing pipeline for both rates.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
#[cfg(test)]
use oxideav_core::AudioFrame;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::BitReader;
use crate::tables::{
    FRAME_SIZE_SAMPLES, HIGH_RATE_BYTES, LOW_RATE_BYTES, LPC_ORDER, PITCH_MAX, PITCH_MIN,
    SAMPLE_RATE_HZ, SUBFRAMES_PER_FRAME, SUBFRAME_SIZE,
};

/// Total payload size for an ACELP (5.3 kbit/s) frame.
const ACELP_PAYLOAD_BYTES: usize = LOW_RATE_BYTES;
/// Total payload size for an MP-MLQ (6.3 kbit/s) frame.
const MPMLQ_PAYLOAD_BYTES: usize = HIGH_RATE_BYTES;

/// MP-MLQ: number of pulses per subframe (odd subframes = 0/2, even = 1/3).
const MPMLQ_PULSES_ODD: usize = 6;
const MPMLQ_PULSES_EVEN: usize = 5;

/// MP-MLQ: per-pulse position bits (8 candidate slots per track) + sign.
const MPMLQ_POS_BITS: u32 = 3;
const MPMLQ_SIGN_BITS: u32 = 1;

/// Bitstream field widths for a 5.3 kbit/s frame, in packing order.
///
/// Bits inside each field are written LSB-first into the payload, matching
/// the LSB-first convention of [`BitReader`].
#[rustfmt::skip]
const ACELP_FIELDS: &[Field] = &[
    Field { name: "RATE",  bits: 2 },   // discriminator = 01
    Field { name: "LSP0",  bits: 8 },
    Field { name: "LSP1",  bits: 8 },
    Field { name: "LSP2",  bits: 8 },
    Field { name: "ACL0",  bits: 7 },
    Field { name: "ACL1",  bits: 2 },
    Field { name: "ACL2",  bits: 7 },
    Field { name: "ACL3",  bits: 2 },
    Field { name: "GAIN0", bits: 12 },
    Field { name: "GAIN1", bits: 12 },
    Field { name: "GAIN2", bits: 12 },
    Field { name: "GAIN3", bits: 12 },
    Field { name: "GRID0", bits: 1 },
    Field { name: "GRID1", bits: 1 },
    Field { name: "GRID2", bits: 1 },
    Field { name: "GRID3", bits: 1 },
    Field { name: "FCB0",  bits: 16 },  // 12 pos + 4 sign per subframe
    Field { name: "FCB1",  bits: 16 },
    Field { name: "FCB2",  bits: 16 },
    Field { name: "FCB3",  bits: 16 },
];

#[derive(Copy, Clone)]
struct Field {
    name: &'static str,
    bits: u32,
}

const _: () = {
    // Total = 2+24+18+48+4+64 = 160 → first 158 carry data, trailing 2 pad.
    // The payload is 20 bytes = 160 bits; the scheme above naturally fills
    // all 160 bits so there is no unused tail.
    let mut t = 0u32;
    let mut i = 0;
    while i < ACELP_FIELDS.len() {
        t += ACELP_FIELDS[i].bits;
        i += 1;
    }
    assert!(t == 160, "ACELP payload must be exactly 160 bits");
};

/// Bitstream field widths for a 6.3 kbit/s MP-MLQ frame, in packing order.
///
/// Packing layout chosen for internal consistency with `decode_mpmlq_local`,
/// NOT the ITU-T Annex B Table B.1 layout (see module docstring for the
/// "local vs spec" caveat). Totals 192 bits = 24 bytes exactly, with no
/// tail padding:
///
/// ```text
///   2 + (8+8+8) + (7+2+7+2) + 4×12 + 4×1 + (24+20+24+20) + 8 = 192
/// ```
///
/// The trailing 8-bit RSVD field is filled with zero and ignored on decode.
#[rustfmt::skip]
const MPMLQ_FIELDS: &[Field] = &[
    Field { name: "RATE",  bits: 2 },   // discriminator = 00
    Field { name: "LSP0",  bits: 8 },
    Field { name: "LSP1",  bits: 8 },
    Field { name: "LSP2",  bits: 8 },
    Field { name: "ACL0",  bits: 7 },
    Field { name: "ACL1",  bits: 2 },
    Field { name: "ACL2",  bits: 7 },
    Field { name: "ACL3",  bits: 2 },
    Field { name: "GAIN0", bits: 12 },
    Field { name: "GAIN1", bits: 12 },
    Field { name: "GAIN2", bits: 12 },
    Field { name: "GAIN3", bits: 12 },
    Field { name: "GRID0", bits: 1 },
    Field { name: "GRID1", bits: 1 },
    Field { name: "GRID2", bits: 1 },
    Field { name: "GRID3", bits: 1 },
    Field { name: "MP0",   bits: 24 },  // 6 pulses × (3 pos + 1 sign) = 24
    Field { name: "MP1",   bits: 20 },  // 5 pulses × (3 pos + 1 sign) = 20
    Field { name: "MP2",   bits: 24 },
    Field { name: "MP3",   bits: 20 },
    Field { name: "RSVD",  bits: 8 },   // zero padding → 24 bytes total
];

const _: () = {
    let mut t = 0u32;
    let mut i = 0;
    while i < MPMLQ_FIELDS.len() {
        t += MPMLQ_FIELDS[i].bits;
        i += 1;
    }
    assert!(t == 192, "MP-MLQ payload must be exactly 192 bits");
};

/// Which rate/mode a given encoder instance is locked to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EncoderMode {
    /// 5.3 kbit/s ACELP (20-byte packets, discriminator = `01`).
    Acelp,
    /// 6.3 kbit/s MP-MLQ (24-byte packets, discriminator = `00`).
    MpMlq,
}

/// Build a G.723.1 encoder. The returned encoder's rate is picked from
/// `params.bit_rate`:
///
/// - `None` or `Some(6300)` → 6.3 kbit/s MP-MLQ (the default).
/// - `Some(5300)` → 5.3 kbit/s ACELP.
/// - Any other bit rate → [`Error::Unsupported`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE_HZ);
    if sample_rate != SAMPLE_RATE_HZ {
        return Err(Error::unsupported(format!(
            "G.723.1 encoder: only {SAMPLE_RATE_HZ} Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.723.1 encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "G.723.1 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    // Pick the rate from bit_rate (default = 6.3 kbit/s MP-MLQ).
    let (mode, bit_rate) = match params.bit_rate {
        None => (EncoderMode::MpMlq, 6_300u64),
        Some(r) if (6_000..=6_500).contains(&r) => (EncoderMode::MpMlq, 6_300u64),
        Some(r) if (5_000..=5_600).contains(&r) => (EncoderMode::Acelp, 5_300u64),
        Some(r) => {
            return Err(Error::unsupported(format!(
                "G.723.1 encoder: bit_rate {r} not supported; valid values are 5300 (ACELP) and 6300 (MP-MLQ)"
            )));
        }
    };

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE_HZ);
    output.bit_rate = Some(bit_rate);

    Ok(Box::new(G7231Encoder::new(output, mode)))
}

/// Encoder state.
pub(crate) struct G7231Encoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    mode: EncoderMode,
    analysis: AnalysisState,
    pcm_queue: Vec<i16>,
    pending: VecDeque<Packet>,
    frame_index: u64,
    eof: bool,
}

impl G7231Encoder {
    fn new(output_params: CodecParameters, mode: EncoderMode) -> Self {
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE_HZ as i64),
            mode,
            analysis: AnalysisState::new(),
            pcm_queue: Vec::new(),
            pending: VecDeque::new(),
            frame_index: 0,
            eof: false,
        }
    }
}

impl Encoder for G7231Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let af = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::invalid("G.723.1 encoder: audio frames only")),
        };
        if af.channels != 1 || af.sample_rate != SAMPLE_RATE_HZ {
            return Err(Error::invalid(
                "G.723.1 encoder: input must be mono, 8000 Hz",
            ));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "G.723.1 encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("G.723.1 encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("G.723.1 encoder: odd byte count"));
        }
        for chunk in bytes.chunks_exact(2) {
            self.pcm_queue
                .push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        self.drain(false);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.drain(true);
        }
        Ok(())
    }
}

impl G7231Encoder {
    fn drain(&mut self, final_flush: bool) {
        while self.pcm_queue.len() >= FRAME_SIZE_SAMPLES {
            let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
            pcm.copy_from_slice(&self.pcm_queue[..FRAME_SIZE_SAMPLES]);
            self.pcm_queue.drain(..FRAME_SIZE_SAMPLES);
            self.emit_frame(&pcm);
        }
        if final_flush && !self.pcm_queue.is_empty() {
            let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
            let n = self.pcm_queue.len();
            for (i, &s) in self.pcm_queue.iter().enumerate() {
                pcm[i] = s;
            }
            let _ = n;
            self.pcm_queue.clear();
            self.emit_frame(&pcm);
        }
    }

    fn emit_frame(&mut self, pcm: &[i16; FRAME_SIZE_SAMPLES]) {
        let frame_idx = self.frame_index;
        self.frame_index += 1;
        let packed = match self.mode {
            EncoderMode::Acelp => {
                let fields = self.analysis.analyse_acelp(pcm);
                pack_acelp_frame(&fields)
            }
            EncoderMode::MpMlq => {
                let fields = self.analysis.analyse_mpmlq(pcm);
                pack_mpmlq_frame(&fields)
            }
        };
        let mut pkt = Packet::new(0, self.time_base, packed);
        pkt.pts = Some(frame_idx as i64 * FRAME_SIZE_SAMPLES as i64);
        pkt.dts = pkt.pts;
        pkt.duration = Some(FRAME_SIZE_SAMPLES as i64);
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
    }
}

// ---------- analysis state ----------

/// All analysis state that persists across frames.
///
/// The encoder maintains a **shadow decoder** ([`decoder`](AnalysisState::decoder))
/// that mirrors what the real decoder will produce — its `exc_history` and
/// LPC filter memory drive the closed-loop ACB/FCB search, and its LSP
/// history drives the next frame's LSP interpolation. Because the encoder
/// and decoder share the same `SynthesisState` structure and synthesis
/// kernel, the encoder's analysis is provably in lockstep with what the
/// decoder renders from the same bitstream.
struct AnalysisState {
    /// Shadow decoder state. `decoder.prev_lsp` is the previous frame's
    /// quantised LSP; `decoder.exc_history` is the excitation buffer that
    /// will be used for ACB prediction of the next subframe;
    /// `decoder.syn_mem` is the synthesis filter memory used to compute
    /// the zero-input response.
    decoder: SynthesisState,
}

impl AnalysisState {
    fn new() -> Self {
        Self {
            decoder: SynthesisState::new(),
        }
    }

    fn analyse_acelp(&mut self, pcm: &[i16; FRAME_SIZE_SAMPLES]) -> FrameFields {
        // ---- 1. Pre-process: s16 -> f32 normalised to [-1, 1]. No HPF
        // is applied here because the decoder does not apply an inverse
        // HPF, so matching full-band signals directly keeps analysis and
        // synthesis in the same signal domain. ----
        let mut sig = [0.0f32; FRAME_SIZE_SAMPLES];
        for i in 0..FRAME_SIZE_SAMPLES {
            sig[i] = pcm[i] as f32 * (1.0 / 32_768.0);
        }

        // ---- 2. LPC analysis on the full 240-sample frame. ----
        let a = lpc_analysis(&sig);
        let lsp_cur = lpc_to_lsp(&a);
        let (lsp_idx, lsp_q) = quantise_lsp(&lsp_cur);

        // ---- 3. Compute the "synthesis-target" signal: the signal we
        // want 1/A_q(z) applied to the chosen excitation to match. Rather
        // than the classical perceptually-weighted target (which trades
        // absolute fidelity for perceptual shape) we match the raw HPF
        // input so the closed-loop search directly minimises sample-
        // space reconstruction error — this is what the SNR-based round-
        // trip tests measure, and it compounds well with the quantised
        // LPC basis (no LPC mismatch between analysis and synthesis). ---
        // The subframe search below uses `h` = impulse response of the
        // *quantised* 1/A_q(z) (not bandwidth-expanded) and target = sig
        // in the current subframe window, filtered through 1/A_q(z) only
        // once to remove the zero-input response of the filter memory.

        // ---- 4. Subframe loop. ----
        let mut acl = [0i32; SUBFRAMES_PER_FRAME];
        let mut gain_idx = [0u32; SUBFRAMES_PER_FRAME];
        let mut grid = [0u8; SUBFRAMES_PER_FRAME];
        let mut fcb = [0u32; SUBFRAMES_PER_FRAME];
        let mut lags = [0i32; SUBFRAMES_PER_FRAME];
        let mut pulses_per_subframe = [[0.0f32; SUBFRAME_SIZE]; SUBFRAMES_PER_FRAME];

        // Snapshot decoder state so the analysis loop (which walks
        // exc_history forward for ACB lookups) can be rolled back before
        // the canonical synthesise() pass commits identical state on both
        // encoder and decoder sides.
        let exc_history_snapshot = self.decoder.exc_history;
        let syn_mem_snapshot = self.decoder.syn_mem;
        let prev_lsp_snapshot = self.decoder.prev_lsp;
        let mut prev_lag: i32 = 60;
        for s in 0..SUBFRAMES_PER_FRAME {
            // Interpolated LPC per subframe — uses the same prev/cur LSPs
            // the decoder will see.
            let lsp_interp = interpolate_lsp(s, &prev_lsp_snapshot, &lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);

            let start = s * SUBFRAME_SIZE;

            // Zero-input response (ZIR) of 1/A_q(z) given the current
            // synthesis filter memory. Subtract from the input target so
            // the closed-loop search only needs to reach the *zero-state*
            // response of the excitation.
            let zir = zero_input_response(&a_sub, &self.decoder.syn_mem, SUBFRAME_SIZE);
            let mut target = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                target[n] = sig[start + n] - zir[n];
            }

            // Impulse response of the zero-state 1/A_q(z) filter.
            let h = impulse_response(&a_sub, SUBFRAME_SIZE);

            // Open-loop pitch search: pick the lag whose ACB prediction
            // best correlates with the zero-state synthesis target.
            let ol_lag = open_loop_acb_lag(&target, &self.decoder.exc_history, &h);

            // Encode lag.
            let lag_code = if s == 0 || s == 2 {
                encode_abs_lag(ol_lag)
            } else {
                encode_delta_lag(ol_lag, prev_lag)
            };
            let decoded_lag = if s == 0 || s == 2 {
                decode_abs_lag(lag_code)
            } else {
                decode_delta_lag(lag_code, prev_lag)
            };
            prev_lag = decoded_lag;
            acl[s] = lag_code as i32;
            lags[s] = decoded_lag;

            // Adaptive codebook excitation + its filtered version.
            let mut adaptive = [0.0f32; SUBFRAME_SIZE];
            copy_adaptive(&self.decoder.exc_history, decoded_lag, &mut adaptive);
            let adapt_filtered = conv_causal(&adaptive, &h);
            let g_adapt_open = lsq_gain(&adapt_filtered, &target).clamp(0.0, ACB_GAIN_MAX);

            // Quantise the ACB gain alone so the FCB search sees the
            // *real* residual the decoder will face after quantisation.
            let acb_idx = quantise_scalar(g_adapt_open, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);
            let g_adapt_q = dequantise_scalar(acb_idx, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);

            // Residual target for the FCB search, using the quantised
            // ACB gain so the pulse solver targets what the decoder will
            // actually need from the FCB path.
            let mut target2 = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                target2[n] = target[n] - g_adapt_q * adapt_filtered[n];
            }

            // 4-pulse ACELP search on tracks T0..T3.
            let (positions, signs, grid_bit) = acelp_4pulse_search(&target2, &h);
            grid[s] = grid_bit;
            fcb[s] = pack_fcb_bits(&positions, signs);

            let mut fcb_pulses = [0.0f32; SUBFRAME_SIZE];
            place_pulses(&positions, signs, grid_bit, &mut fcb_pulses);
            pulses_per_subframe[s] = fcb_pulses;
            let fcb_filtered = conv_causal(&fcb_pulses, &h);

            // FCB gain (optimal unconstrained) — will be quantised next.
            let fcb_mag_ceil = 2.0f32.powf(FCB_GAIN_LOG2_MAX);
            let g_fixed = lsq_gain(&fcb_filtered, &target2).clamp(-fcb_mag_ceil, fcb_mag_ceil);

            // Initial FCB-gain quantisation + sign.
            let sign_bit = if g_fixed < 0.0 { 1u32 } else { 0 };
            let mag = g_fixed.abs().max(2.0f32.powf(FCB_GAIN_LOG2_MIN));
            let fcb_idx0 = quantise_scalar(
                mag.log2(),
                FCB_GAIN_LOG2_MIN,
                FCB_GAIN_LOG2_MAX,
                FCB_GAIN_LEVELS,
            );
            // Refine the joint (ACB, FCB) gain pair against the real
            // reconstruction error by scanning a small neighbourhood in
            // index space — catches rounding disagreements where
            // nominally-nearest pair isn't optimal after the interaction
            // with the filtered pulse/ACB responses.
            let gi = refine_gain_pair(
                acb_idx,
                fcb_idx0,
                sign_bit,
                &adapt_filtered,
                &fcb_filtered,
                &target,
            );
            gain_idx[s] = gi;

            // Advance the shadow decoder for the next subframe's ACB
            // lookup. Rolled back before the canonical synthesise() below.
            let (g_adapt_q, g_fixed_q) = dequantise_gain(gi);
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                exc[n] = g_adapt_q * adaptive[n] + g_fixed_q * fcb_pulses[n];
            }
            self.decoder.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.decoder.exc_history.len() - SUBFRAME_SIZE;
            self.decoder.exc_history[tail..].copy_from_slice(&exc);
            // Advance the synthesis filter memory in the same way the
            // decoder will, using a_sub and the quantised excitation so
            // the next subframe's ZIR matches what the decoder sees.
            advance_syn_mem(&a_sub, &exc, &mut self.decoder.syn_mem);
        }

        // Rewind decoder state to the pre-frame snapshot and run the
        // canonical synthesis kernel. This commits exc_history, syn_mem,
        // and prev_lsp exactly the way the decoder will, so on the next
        // frame the shadow state is already in sync.
        self.decoder.exc_history = exc_history_snapshot;
        self.decoder.syn_mem = syn_mem_snapshot;
        let mut pcm_f = [0.0f32; FRAME_SIZE_SAMPLES];
        self.decoder.synthesise(
            &lsp_q,
            &lags,
            &grid,
            &gain_idx,
            &pulses_per_subframe,
            &mut pcm_f,
        );

        FrameFields {
            lsp_idx,
            acl,
            gain: gain_idx,
            grid,
            fcb,
        }
    }

    /// Sister method of [`analyse_acelp`] producing an [`MpMlqFrameFields`]
    /// for the 6.3 kbit/s MP-MLQ path. Shares the LPC / LSP / pitch / gain
    /// machinery — only the fixed codebook search differs (6 or 5 pulses
    /// per subframe rather than 4, greedy search rather than per-track).
    fn analyse_mpmlq(&mut self, pcm: &[i16; FRAME_SIZE_SAMPLES]) -> MpMlqFrameFields {
        // ---- 1. Pre-process: s16 -> f32 normalised to [-1, 1]. ----
        let mut sig = [0.0f32; FRAME_SIZE_SAMPLES];
        for i in 0..FRAME_SIZE_SAMPLES {
            sig[i] = pcm[i] as f32 * (1.0 / 32_768.0);
        }

        // ---- 2. LPC analysis on full frame. ----
        let a = lpc_analysis(&sig);
        let lsp_cur = lpc_to_lsp(&a);
        let (lsp_idx, lsp_q) = quantise_lsp(&lsp_cur);
        let _ = a;

        // ---- 4. Subframe loop. ----
        let mut acl = [0i32; SUBFRAMES_PER_FRAME];
        let mut gain_idx = [0u32; SUBFRAMES_PER_FRAME];
        let mut grid = [0u8; SUBFRAMES_PER_FRAME];
        let mut mp = [MpMlqPulses::default(); SUBFRAMES_PER_FRAME];
        let mut lags = [0i32; SUBFRAMES_PER_FRAME];
        let mut pulses_per_subframe = [[0.0f32; SUBFRAME_SIZE]; SUBFRAMES_PER_FRAME];

        let exc_history_snapshot = self.decoder.exc_history;
        let syn_mem_snapshot = self.decoder.syn_mem;
        let prev_lsp_snapshot = self.decoder.prev_lsp;
        let mut prev_lag: i32 = 60;
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, &prev_lsp_snapshot, &lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);

            let start = s * SUBFRAME_SIZE;

            let zir = zero_input_response(&a_sub, &self.decoder.syn_mem, SUBFRAME_SIZE);
            let mut target = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                target[n] = sig[start + n] - zir[n];
            }
            let h = impulse_response(&a_sub, SUBFRAME_SIZE);

            let ol_lag = open_loop_acb_lag(&target, &self.decoder.exc_history, &h);

            let lag_code = if s == 0 || s == 2 {
                encode_abs_lag(ol_lag)
            } else {
                encode_delta_lag(ol_lag, prev_lag)
            };
            let decoded_lag = if s == 0 || s == 2 {
                decode_abs_lag(lag_code)
            } else {
                decode_delta_lag(lag_code, prev_lag)
            };
            prev_lag = decoded_lag;
            acl[s] = lag_code as i32;
            lags[s] = decoded_lag;

            let mut adaptive = [0.0f32; SUBFRAME_SIZE];
            copy_adaptive(&self.decoder.exc_history, decoded_lag, &mut adaptive);
            let adapt_filtered = conv_causal(&adaptive, &h);
            let g_adapt_open = lsq_gain(&adapt_filtered, &target).clamp(0.0, ACB_GAIN_MAX);
            let acb_idx = quantise_scalar(g_adapt_open, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);
            let g_adapt_q = dequantise_scalar(acb_idx, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);

            let mut target2 = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                target2[n] = target[n] - g_adapt_q * adapt_filtered[n];
            }

            let n_pulses = if s % 2 == 0 {
                MPMLQ_PULSES_ODD
            } else {
                MPMLQ_PULSES_EVEN
            };
            let (positions, signs, grid_bit) = mpmlq_pulse_search(&target2, &h, n_pulses);
            grid[s] = grid_bit;
            mp[s] = MpMlqPulses {
                positions,
                signs,
                n_pulses: n_pulses as u8,
            };

            let mut fcb_pulses = [0.0f32; SUBFRAME_SIZE];
            mpmlq_place_pulses(&positions, &signs, n_pulses, grid_bit, &mut fcb_pulses);
            pulses_per_subframe[s] = fcb_pulses;
            let fcb_filtered = conv_causal(&fcb_pulses, &h);

            let fcb_mag_ceil = 2.0f32.powf(FCB_GAIN_LOG2_MAX);
            let g_fixed = lsq_gain(&fcb_filtered, &target2).clamp(-fcb_mag_ceil, fcb_mag_ceil);

            let sign_bit = if g_fixed < 0.0 { 1u32 } else { 0 };
            let mag = g_fixed.abs().max(2.0f32.powf(FCB_GAIN_LOG2_MIN));
            let fcb_idx0 = quantise_scalar(
                mag.log2(),
                FCB_GAIN_LOG2_MIN,
                FCB_GAIN_LOG2_MAX,
                FCB_GAIN_LEVELS,
            );
            let gi = refine_gain_pair(
                acb_idx,
                fcb_idx0,
                sign_bit,
                &adapt_filtered,
                &fcb_filtered,
                &target,
            );
            gain_idx[s] = gi;

            let (g_adapt_q, g_fixed_q) = dequantise_gain(gi);
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                exc[n] = g_adapt_q * adaptive[n] + g_fixed_q * fcb_pulses[n];
            }
            self.decoder.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.decoder.exc_history.len() - SUBFRAME_SIZE;
            self.decoder.exc_history[tail..].copy_from_slice(&exc);
            advance_syn_mem(&a_sub, &exc, &mut self.decoder.syn_mem);
        }

        self.decoder.exc_history = exc_history_snapshot;
        self.decoder.syn_mem = syn_mem_snapshot;
        let mut pcm_f = [0.0f32; FRAME_SIZE_SAMPLES];
        self.decoder.synthesise(
            &lsp_q,
            &lags,
            &grid,
            &gain_idx,
            &pulses_per_subframe,
            &mut pcm_f,
        );

        MpMlqFrameFields {
            lsp_idx,
            acl,
            gain: gain_idx,
            grid,
            mp,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FrameFields {
    lsp_idx: [u32; 3],
    acl: [i32; SUBFRAMES_PER_FRAME],
    gain: [u32; SUBFRAMES_PER_FRAME],
    grid: [u8; SUBFRAMES_PER_FRAME],
    fcb: [u32; SUBFRAMES_PER_FRAME],
}

/// MP-MLQ pulse layout for a single subframe. At most
/// [`MPMLQ_PULSES_ODD`] pulses; `n_pulses` tells decoders how many slots
/// are populated (5 on even subframes, 6 on odd subframes).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MpMlqPulses {
    pub(crate) positions: [u32; MPMLQ_PULSES_ODD],
    pub(crate) signs: [i32; MPMLQ_PULSES_ODD],
    pub(crate) n_pulses: u8,
}

#[derive(Clone, Copy, Debug)]
struct MpMlqFrameFields {
    lsp_idx: [u32; 3],
    acl: [i32; SUBFRAMES_PER_FRAME],
    gain: [u32; SUBFRAMES_PER_FRAME],
    grid: [u8; SUBFRAMES_PER_FRAME],
    mp: [MpMlqPulses; SUBFRAMES_PER_FRAME],
}

// ---------- LPC analysis ----------

/// Autocorrelation + Levinson-Durbin on the full 240-sample frame. Output
/// is `[1, a_1..a_10]` in direct form.
fn lpc_analysis(sig: &[f32; FRAME_SIZE_SAMPLES]) -> [f32; LPC_ORDER + 1] {
    // Hamming window of length 240 (approximation of the spec's LPC window).
    let mut windowed = [0.0f32; FRAME_SIZE_SAMPLES];
    let n = FRAME_SIZE_SAMPLES as f32;
    for i in 0..FRAME_SIZE_SAMPLES {
        let w = 0.54 - 0.46 * ((2.0 * std::f32::consts::PI * i as f32) / (n - 1.0)).cos();
        windowed[i] = sig[i] * w;
    }
    // Autocorrelation r[0..=LPC_ORDER].
    let mut r = [0.0f64; LPC_ORDER + 1];
    for k in 0..=LPC_ORDER {
        let mut acc = 0.0f64;
        for i in k..FRAME_SIZE_SAMPLES {
            acc += windowed[i] as f64 * windowed[i - k] as f64;
        }
        r[k] = acc;
    }
    // Small bandwidth-expansion factor on the autocorrelation (white-noise
    // correction, ~40 Hz lag window).
    r[0] *= 1.0001;
    for k in 1..=LPC_ORDER {
        let w = (-0.5
            * (2.0 * std::f32::consts::PI * 60.0 * k as f32 / SAMPLE_RATE_HZ as f32).powi(2))
            as f64;
        r[k] *= w.exp();
    }

    // Levinson-Durbin recursion.
    let mut a = [0.0f64; LPC_ORDER + 1];
    let mut a_prev = [0.0f64; LPC_ORDER + 1];
    a[0] = 1.0;
    a_prev[0] = 1.0;
    let mut e = r[0];
    if e <= 0.0 {
        return default_a();
    }
    for i in 1..=LPC_ORDER {
        // Reflection coefficient.
        let mut acc = r[i];
        for j in 1..i {
            acc += a_prev[j] * r[i - j];
        }
        let k = -acc / e;
        a[i] = k;
        for j in 1..i {
            a[j] = a_prev[j] + k * a_prev[i - j];
        }
        e *= 1.0 - k * k;
        if e <= 1e-20 {
            return default_a();
        }
        a_prev.copy_from_slice(&a);
    }
    let mut out = [0.0f32; LPC_ORDER + 1];
    for i in 0..=LPC_ORDER {
        out[i] = a[i] as f32;
    }
    out
}

fn default_a() -> [f32; LPC_ORDER + 1] {
    let mut a = [0.0f32; LPC_ORDER + 1];
    a[0] = 1.0;
    a
}

/// Apply bandwidth expansion: `a_i <- a_i * gamma^i`.
fn bandwidth_expand(a: &[f32; LPC_ORDER + 1], gamma: f32) -> [f32; LPC_ORDER + 1] {
    let mut out = *a;
    let mut g = 1.0f32;
    for i in 0..=LPC_ORDER {
        out[i] = a[i] * g;
        g *= gamma;
    }
    out
}

// ---------- LPC <-> LSP ----------

/// Convert LPC direct-form coefficients to Line Spectral Pairs in the
/// cosine domain (lsp[i] = cos(omega_i)). Uses the standard Chebyshev
/// root-finding on the P(z) / Q(z) polynomials.
fn lpc_to_lsp(a: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER] {
    // Form f1(z) = A(z) + z^-(p+1) A(z^-1); f2(z) = A(z) - z^-(p+1) A(z^-1).
    // After factoring out the trivial roots, we get polynomials of degree
    // p/2 in cos(omega) (Chebyshev expansion).
    let p = LPC_ORDER;
    let mut f1 = [0.0f32; LPC_ORDER / 2 + 1];
    let mut f2 = [0.0f32; LPC_ORDER / 2 + 1];
    // f1_i = a_i + a_{p-i}, i = 0..p/2; remove (1 + z^-1) factor:
    // recursive: f1[i] = (a[i] + a[p-i]) - f1[i-1]
    // f2[i] = (a[i] - a[p-i]) + f2[i-1]
    f1[0] = 1.0;
    f2[0] = 1.0;
    let mut prev_f1 = 0.0f32;
    let mut prev_f2 = 0.0f32;
    for i in 1..=p / 2 {
        let ai = a[i];
        let api = a[p + 1 - i];
        f1[i] = ai + api - prev_f1;
        f2[i] = ai - api + prev_f2;
        prev_f1 = f1[i];
        prev_f2 = f2[i];
    }
    // Evaluate both polynomials on [-1, 1] in cos-domain; interleave roots
    // of f1 and f2 strictly, as required.
    let roots_f1 = cheby_roots(&f1);
    let roots_f2 = cheby_roots(&f2);
    let mut lsp = [0.0f32; LPC_ORDER];
    // Interleave: LSP ordering alternates between f1 and f2 roots.
    let n1 = roots_f1.len();
    let n2 = roots_f2.len();
    for k in 0..LPC_ORDER {
        if k % 2 == 0 && k / 2 < n1 {
            lsp[k] = roots_f1[k / 2];
        } else if k / 2 < n2 {
            lsp[k] = roots_f2[k / 2];
        } else {
            // Fallback: uniform spacing.
            let step = std::f32::consts::PI / (LPC_ORDER as f32 + 1.0);
            lsp[k] = (step * (k as f32 + 1.0)).cos();
        }
    }
    // Ensure strictly decreasing cos (= increasing omega).
    for k in 1..LPC_ORDER {
        if lsp[k] >= lsp[k - 1] - 1e-4 {
            lsp[k] = lsp[k - 1] - 1e-3;
        }
    }
    lsp
}

/// Find roots of a Chebyshev-expanded polynomial in the cos-domain on
/// `[-1, 1]` by bisection / root-bracketing across a fine grid.
fn cheby_roots(coeffs: &[f32]) -> Vec<f32> {
    // Evaluate the (implicit) polynomial in x = cos(omega).
    // coeffs[0] + coeffs[1] * T_1(x) + ... + coeffs[deg] * T_deg(x)
    // For our needs, an approximate grid-bisection search suffices.
    let deg = coeffs.len() - 1;
    let eval = |x: f32| -> f32 {
        // Clenshaw's algorithm for Chebyshev series.
        let mut b2 = 0.0f32;
        let mut b1 = 0.0f32;
        for k in (1..=deg).rev() {
            let b0 = 2.0 * x * b1 - b2 + coeffs[k];
            b2 = b1;
            b1 = b0;
        }
        x * b1 - b2 + coeffs[0]
    };
    const GRID: usize = 200;
    let mut roots = Vec::with_capacity(deg);
    let mut prev_x = 1.0f32;
    let mut prev_y = eval(prev_x);
    for i in 1..=GRID {
        let x = 1.0 - 2.0 * (i as f32 / GRID as f32);
        let y = eval(x);
        if prev_y * y < 0.0 {
            // Bisect.
            let mut lo = x;
            let mut hi = prev_x;
            let mut flo = y;
            let _fhi = prev_y;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let fm = eval(mid);
                if fm * flo < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push(0.5 * (lo + hi));
            if roots.len() == deg {
                break;
            }
        }
        prev_x = x;
        prev_y = y;
    }
    roots
}

/// Convert LSPs (cosine-domain) back to direct-form LPC coefficients.
fn lsp_to_lpc(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
    // Reconstruct A(z) from LSPs in the cosine domain. Standard
    // construction (e.g. ITU-T G.729 / G.723.1 reference):
    //
    //   P(z) = prod_{k even}(1 - 2 lsp[k] z^-1 + z^-2)      degree p
    //   Q(z) = prod_{k odd }(1 - 2 lsp[k] z^-1 + z^-2)      degree p
    //   f1(z) = P(z) * (1 + z^-1)                            degree p+1
    //   f2(z) = Q(z) * (1 - z^-1)                            degree p+1
    //   A(z) = (f1(z) + f2(z)) / 2                           degree p (top
    //                                                        coefficient
    //                                                        cancels by
    //                                                        symmetry)
    //
    // P and Q each have degree p = 10 after multiplying five quadratic
    // factors; f1 and f2 bump the degree by 1. The earlier version of
    // this function allocated only p/2+1 coefficients for each
    // polynomial, silently truncating the top half of A(z) and producing
    // an unstable ~p/2-order filter with wildly wrong gain — that was
    // the proximate cause of the encoder-decoder amplitude mismatch.
    let p = LPC_ORDER;
    let half = p / 2;
    let mut pz = vec![0.0f32; p + 1];
    let mut qz = vec![0.0f32; p + 1];
    pz[0] = 1.0;
    qz[0] = 1.0;
    let mut pz_deg: usize = 0;
    let mut qz_deg: usize = 0;
    for k in 0..half {
        let lsp_even = lsp[2 * k];
        let lsp_odd = lsp[2 * k + 1];
        pz_deg += 2;
        for i in (2..=pz_deg).rev() {
            pz[i] += -2.0 * lsp_even * pz[i - 1] + pz[i - 2];
        }
        pz[1] -= 2.0 * lsp_even * pz[0];
        qz_deg += 2;
        for i in (2..=qz_deg).rev() {
            qz[i] += -2.0 * lsp_odd * qz[i - 1] + qz[i - 2];
        }
        qz[1] -= 2.0 * lsp_odd * qz[0];
    }
    // Apply the trivial factors: f1 = pz * (1 + z^-1), f2 = qz * (1 - z^-1).
    let mut f1 = vec![0.0f32; p + 2];
    let mut f2 = vec![0.0f32; p + 2];
    for i in 0..=p {
        f1[i] += pz[i];
        f1[i + 1] += pz[i];
        f2[i] += qz[i];
        f2[i + 1] -= qz[i];
    }
    // A(z) = (f1 + f2) / 2 — keep only degree 0..p, the top coefficient
    // cancels by construction.
    let mut a = [0.0f32; LPC_ORDER + 1];
    for i in 0..=p {
        a[i] = 0.5 * (f1[i] + f2[i]);
    }
    a[0] = 1.0;
    a
}

/// Interpolate LSP vectors between the previous and current frame for
/// subframe `k in 0..4`.
fn interpolate_lsp(k: usize, prev: &[f32; LPC_ORDER], cur: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let (wp, wc) = match k {
        0 => (0.75, 0.25),
        1 => (0.50, 0.50),
        2 => (0.25, 0.75),
        _ => (0.0, 1.0),
    };
    let mut out = [0.0f32; LPC_ORDER];
    for i in 0..LPC_ORDER {
        out[i] = wp * prev[i] + wc * cur[i];
    }
    out
}

// ---------- LSP quantisation (factorial scalar split VQ) ----------
//
// Three 8-bit indices together form a 24-bit quantisation of the 10-LSP
// vector. Each split is a *factorial* product code (cartesian product of
// per-dimension scalar quantisers) which makes encode/decode O(p) instead
// of the 256-entry nearest-neighbour search used by a trained codebook:
//
//   split 0: LSP[0..3]  — 3 dims × (3 bits, 3 bits, 2 bits) = 8 bits
//   split 1: LSP[3..6]  — 3 dims × (3 bits, 3 bits, 2 bits) = 8 bits
//   split 2: LSP[6..10] — 4 dims × (2 bits each)            = 8 bits
//
// Each dimension's scalar quantiser operates in the LSP **angle** domain
// omega = acos(lsp) and has a per-dim min/max range chosen so that the
// quantiser cells stay ordered after reconstruction (the decoder still
// enforces monotonicity with a safety clamp; in practice the ranges below
// keep LSPs strictly decreasing-in-cosine for any valid index combination).
//
// This is NOT the ITU-T Table 5 codebook, but it is a proper split scalar
// quantiser with ~0.04–0.12 rad resolution per dimension, fine enough to
// reconstruct LPC spectra well inside the 20 dB SNR target of the crate's
// round-trip tests.

const LSP_SPLIT_0: usize = 3;
const LSP_SPLIT_1: usize = 3;
const LSP_SPLIT_2: usize = 4;

/// Per-dimension bit allocation for each of the three 8-bit LSP splits.
/// Lists sum to 8, arrays are indexed in LSP-order within the split.
const LSP_BITS_SPLIT_0: [u32; LSP_SPLIT_0] = [3, 3, 2];
const LSP_BITS_SPLIT_1: [u32; LSP_SPLIT_1] = [3, 3, 2];
const LSP_BITS_SPLIT_2: [u32; LSP_SPLIT_2] = [2, 2, 2, 2];

/// Per-dim (omega = acos(lsp)) quantisation ranges in radians, over the
/// full LPC_ORDER=10 vector. Chosen so that typical voiced/unvoiced speech
/// LSP angles land inside the range and consecutive dims overlap only
/// slightly, keeping the monotonicity constraint cheap to enforce.
#[rustfmt::skip]
const LSP_OMEGA_RANGES: [(f32, f32); LPC_ORDER] = [
    (0.10, 0.45),  // dim 0 — pitch/sub-100 Hz region
    (0.30, 0.85),  // dim 1 — first formant low
    (0.55, 1.20),  // dim 2 — first formant high
    (0.90, 1.55),  // dim 3 — second formant
    (1.25, 1.85),  // dim 4 — between F2/F3
    (1.55, 2.15),  // dim 5 — third formant
    (1.85, 2.40),  // dim 6 — F3/F4 region
    (2.15, 2.65),  // dim 7 — fourth formant
    (2.40, 2.85),  // dim 8 — above F4
    (2.65, 3.05),  // dim 9 — near-pi tail
];

/// Return `(start_dim, bits)` for each split in a unified table.
fn lsp_split_bits(split: usize) -> (usize, &'static [u32]) {
    match split {
        0 => (0, &LSP_BITS_SPLIT_0),
        1 => (LSP_SPLIT_0, &LSP_BITS_SPLIT_1),
        _ => (LSP_SPLIT_0 + LSP_SPLIT_1, &LSP_BITS_SPLIT_2),
    }
}

fn quantise_lsp(lsp: &[f32; LPC_ORDER]) -> ([u32; 3], [f32; LPC_ORDER]) {
    // Factorial scalar quantiser per dimension. Encoding is O(p) — we
    // simply map each LSP cos → omega, clamp to the dim's range, and pick
    // the nearest of `2^bits` levels.
    let mut idx = [0u32; 3];
    for s in 0..3 {
        let (start, bits) = lsp_split_bits(s);
        let mut packed: u32 = 0;
        let mut shift: u32 = 0;
        for (i, &b) in bits.iter().enumerate() {
            let dim = start + i;
            let (lo, hi) = LSP_OMEGA_RANGES[dim];
            let omega = lsp[dim].clamp(-1.0, 1.0).acos();
            let levels = 1u32 << b;
            let q = quantise_scalar(omega, lo, hi, levels);
            packed |= q << shift;
            shift += b;
        }
        idx[s] = packed;
    }
    let quantised = dequantise_lsp(&idx);
    (idx, quantised)
}

/// Measure the angle-domain L2 distance from the quantised LSP back to
/// the unquantised LSP, in radians. Used by tests / diagnostics.
#[cfg(test)]
pub(crate) fn lsp_quant_distance(lsp: &[f32; LPC_ORDER]) -> f32 {
    let (idx, _) = quantise_lsp(lsp);
    let q = dequantise_lsp(&idx);
    let mut d = 0.0f32;
    for i in 0..LPC_ORDER {
        let o = lsp[i].clamp(-1.0, 1.0).acos();
        let oq = q[i].clamp(-1.0, 1.0).acos();
        d += (o - oq).powi(2);
    }
    d.sqrt()
}

pub(crate) fn dequantise_lsp(idx: &[u32; 3]) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for s in 0..3 {
        let (start, bits) = lsp_split_bits(s);
        let mut packed = idx[s];
        for (i, &b) in bits.iter().enumerate() {
            let dim = start + i;
            let (lo, hi) = LSP_OMEGA_RANGES[dim];
            let levels = 1u32 << b;
            let q = packed & ((1u32 << b) - 1);
            let omega = dequantise_scalar(q, lo, hi, levels);
            out[dim] = omega.cos();
            packed >>= b;
        }
    }
    // Enforce strict ordering + minimum separation in angle space, so
    // `lsp_to_lpc` always produces a stable LPC (inside the unit circle).
    // In cosine domain, strictly decreasing with a minimum gap of 0.01
    // corresponds to >~0.1 rad spacing in the worst case, enough to keep
    // the synthesis filter well-behaved.
    for i in 1..LPC_ORDER {
        if out[i] >= out[i - 1] - 0.01 {
            out[i] = out[i - 1] - 0.01;
        }
    }
    // Clamp to valid LSP cosine range with a margin so the outer roots
    // stay comfortably inside the unit circle.
    out[0] = out[0].min(0.995);
    out[LPC_ORDER - 1] = out[LPC_ORDER - 1].max(-0.995);
    out
}

/// Scalar quantiser: pick the nearest of `levels` points in `[lo, hi]`.
fn quantise_scalar(x: f32, lo: f32, hi: f32, levels: u32) -> u32 {
    debug_assert!(levels >= 2);
    let x = x.clamp(lo, hi);
    let step = (hi - lo) / (levels as f32 - 1.0);
    (((x - lo) / step).round() as i32).clamp(0, levels as i32 - 1) as u32
}

/// Inverse scalar quantiser — reconstructs the level centre.
fn dequantise_scalar(q: u32, lo: f32, hi: f32, levels: u32) -> f32 {
    let step = (hi - lo) / (levels as f32 - 1.0);
    lo + (q.min(levels - 1) as f32) * step
}

// ---------- pitch + ACB ----------

/// Copy the adaptive codebook excitation for `lag`, handling wrap-around
/// when `lag < SUBFRAME_SIZE` by re-reading the last `lag` samples
/// periodically (the standard "periodic excitation" convention).
fn copy_adaptive(history: &[f32], lag: i32, out: &mut [f32; SUBFRAME_SIZE]) {
    let hlen = history.len();
    let lag = lag.clamp(PITCH_MIN as i32, PITCH_MAX as i32) as usize;
    for n in 0..SUBFRAME_SIZE {
        let idx = if lag > n {
            hlen - (lag - n)
        } else {
            // Wrap inside the final `lag` samples of the history.
            hlen - lag + ((n - lag) % lag)
        };
        out[n] = if idx < hlen { history[idx] } else { 0.0 };
    }
}

fn encode_abs_lag(lag: i32) -> u32 {
    // 7-bit absolute: offset 18..=145 → 0..=127.
    let v = (lag - PITCH_MIN as i32).clamp(0, 127);
    v as u32
}

fn decode_abs_lag(code: u32) -> i32 {
    PITCH_MIN as i32 + (code & 0x7F) as i32
}

fn encode_delta_lag(lag: i32, prev_lag: i32) -> u32 {
    // 2-bit delta in {-1, 0, +1, +2}.
    let d = (lag - prev_lag).clamp(-1, 2);
    ((d + 1) as u32) & 0x3
}

fn decode_delta_lag(code: u32, prev_lag: i32) -> i32 {
    let d = (code & 0x3) as i32 - 1;
    (prev_lag + d).clamp(PITCH_MIN as i32, PITCH_MAX as i32)
}

// ---------- ACELP 4-pulse search ----------

/// Four-pulse ACELP fixed-codebook search. Each of the 4 pulses lives on
/// its own track with stride-8 positions (covering 8 positions per track);
/// the grid bit shifts all pulses by +4 so the union of both grids spans
/// all 60 subframe samples.
///
/// Track layout (grid 0):
///
/// ```text
///   T0: 0,  8, 16, 24, 32, 40, 48, 56
///   T1: 1,  9, 17, 25, 33, 41, 49, 57
///   T2: 2, 10, 18, 26, 34, 42, 50, 58
///   T3: 3, 11, 19, 27, 35, 43, 51, 59
/// ```
///
/// Grid 1 shifts each position by 4. Combined, every position in
/// `[0, SUBFRAME_SIZE)` is reachable by exactly one (track, grid, k)
/// triple — so the 3-bit position code + 1-bit sign per track, plus the
/// 1-bit grid per subframe, gives full per-sample coverage at the cost
/// of searching 2 × 4 × 8 = 64 candidates rather than 32.
///
/// After the per-track greedy pick, the algorithm does two passes of
/// coordinate-descent refinement: for each pulse in turn it re-optimises
/// its (position, sign) given the other three fixed — so pulses that
/// were sub-optimal because of correlation with another pulse on the
/// grid get adjusted.
fn acelp_4pulse_search(target: &[f32; SUBFRAME_SIZE], h: &[f32]) -> ([u32; 4], [i32; 4], u8) {
    let d = compute_correlations(target, h);
    let stride: usize = 8;
    let positions_per_track: usize = 8;

    let mut best_grid = 0u8;
    let mut best_err = f32::INFINITY;
    let mut best_positions = [0u32; 4];
    let mut best_signs = [1i32; 4];

    for grid in 0..2u8 {
        let grid_offset = grid as usize * 4;

        // Pass 1: per-track greedy pick (initial solution).
        let mut positions = [0u32; 4];
        let mut signs = [1i32; 4];
        for track in 0..4usize {
            let mut best_gain2 = 0.0f32;
            let mut best_k = 0u32;
            let mut best_sign = 1i32;
            for k in 0..positions_per_track {
                let pos = track + stride * k + grid_offset;
                if pos >= SUBFRAME_SIZE {
                    continue;
                }
                let ap = autocorr_at(h, pos);
                if ap < 1e-8 {
                    continue;
                }
                let dv = d[pos];
                let score = dv * dv / ap;
                if score > best_gain2 {
                    best_gain2 = score;
                    best_k = k as u32;
                    best_sign = if dv >= 0.0 { 1 } else { -1 };
                }
            }
            positions[track] = best_k;
            signs[track] = best_sign;
        }

        // Pass 2-3: coordinate descent — for each track in turn, fix the
        // others and pick the (k, sign) that minimises the residual
        // between the target and the synthesised sum of pulses.
        for _pass in 0..2 {
            for track in 0..4usize {
                let mut others = [0.0f32; SUBFRAME_SIZE];
                for t2 in 0..4usize {
                    if t2 == track {
                        continue;
                    }
                    let pos = t2 + stride * positions[t2] as usize + grid_offset;
                    if pos < SUBFRAME_SIZE {
                        let sgn = signs[t2] as f32;
                        for n in pos..SUBFRAME_SIZE {
                            others[n] += sgn * h[n - pos];
                        }
                    }
                }
                let mut resid = [0.0f32; SUBFRAME_SIZE];
                for n in 0..SUBFRAME_SIZE {
                    resid[n] = target[n] - others[n];
                }
                // Find best (k, sign) for this track against resid.
                let mut best_err2 = f32::INFINITY;
                let mut best_k = positions[track];
                let mut best_sign = signs[track];
                for k in 0..positions_per_track {
                    let pos = track + stride * k + grid_offset;
                    if pos >= SUBFRAME_SIZE {
                        continue;
                    }
                    // Best sign at this position minimises |resid - sign*h_pos|^2.
                    // sign* = sign(<resid, h_pos>); resulting err = |resid|^2 - <resid, h_pos>^2 / |h_pos|^2.
                    let ap = autocorr_at(h, pos);
                    if ap < 1e-8 {
                        continue;
                    }
                    let mut corr = 0.0f32;
                    for n in pos..SUBFRAME_SIZE {
                        corr += resid[n] * h[n - pos];
                    }
                    let sign_v: i32 = if corr >= 0.0 { 1 } else { -1 };
                    let gain = sign_v as f32 * corr.abs() / ap;
                    let mut err = 0.0f32;
                    for n in 0..SUBFRAME_SIZE {
                        let h_at = if n >= pos { h[n - pos] } else { 0.0 };
                        let e = resid[n] - gain * h_at;
                        err += e * e;
                    }
                    if err < best_err2 {
                        best_err2 = err;
                        best_k = k as u32;
                        best_sign = sign_v;
                    }
                }
                positions[track] = best_k;
                signs[track] = best_sign;
            }
        }

        // Score this grid: compute reconstruction error.
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for track in 0..4usize {
            let pos = track + stride * positions[track] as usize + grid_offset;
            if pos < SUBFRAME_SIZE {
                let sgn = signs[track] as f32;
                for n in pos..SUBFRAME_SIZE {
                    syn[n] += sgn * h[n - pos];
                }
            }
        }
        let mut err = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            let e = target[n] - syn[n];
            err += e * e;
        }
        if err < best_err {
            best_err = err;
            best_grid = grid;
            best_positions = positions;
            best_signs = signs;
        }
    }
    (best_positions, best_signs, best_grid)
}

/// Compute d[n] = <target, h_n> for n in 0..SUBFRAME_SIZE.
fn compute_correlations(target: &[f32; SUBFRAME_SIZE], h: &[f32]) -> [f32; SUBFRAME_SIZE] {
    let mut d = [0.0f32; SUBFRAME_SIZE];
    for i in 0..SUBFRAME_SIZE {
        let mut acc = 0.0f32;
        // h_i[n] = h[n - i] for n >= i
        for n in i..SUBFRAME_SIZE {
            acc += target[n] * h[n - i];
        }
        d[i] = acc;
    }
    d
}

fn autocorr_at(h: &[f32], i: usize) -> f32 {
    // sum_{n=i..SUBFRAME_SIZE} h[n-i]^2 = sum_{m=0..SUBFRAME_SIZE-i} h[m]^2
    let end = SUBFRAME_SIZE.saturating_sub(i);
    let mut acc = 0.0f32;
    for m in 0..end.min(h.len()) {
        acc += h[m] * h[m];
    }
    acc
}

fn pack_fcb_bits(positions: &[u32; 4], signs: [i32; 4]) -> u32 {
    // 4 x 3-bit positions (low 12 bits) + 4 x 1-bit signs (high 4 bits).
    let mut v = 0u32;
    for i in 0..4 {
        v |= (positions[i] & 0x7) << (i * 3);
    }
    let mut sb = 0u32;
    for i in 0..4 {
        if signs[i] < 0 {
            sb |= 1 << i;
        }
    }
    v | (sb << 12)
}

pub(crate) fn unpack_fcb_bits(v: u32) -> ([u32; 4], [i32; 4]) {
    let mut positions = [0u32; 4];
    let mut signs = [1i32; 4];
    for i in 0..4 {
        positions[i] = (v >> (i * 3)) & 0x7;
        let sb = (v >> (12 + i)) & 0x1;
        signs[i] = if sb == 1 { -1 } else { 1 };
    }
    (positions, signs)
}

/// Place 4 pulses at positions specified by tracks + grid bit. Must
/// mirror the layout used by [`acelp_4pulse_search`].
pub(crate) fn place_pulses(
    positions: &[u32; 4],
    signs: [i32; 4],
    grid: u8,
    out: &mut [f32; SUBFRAME_SIZE],
) {
    let stride: usize = 8;
    let grid_offset = grid as usize * 4;
    out.fill(0.0);
    for track in 0..4usize {
        let k = positions[track] as usize;
        let pos = track + stride * k + grid_offset;
        if pos < SUBFRAME_SIZE {
            out[pos] = signs[track] as f32;
        }
    }
}

// ---------- MP-MLQ (6.3 kbit/s) pulse search ----------
//
// 3-bit "slot" index per pulse + 1-bit sign = 4 bits per pulse. Pulses
// share a track layout inspired by the ACELP one: track `t ∈ 0..n_pulses`
// at stride `MPMLQ_STRIDE = 8`, with the 1-bit grid adding a 4-sample
// shift so the union of both grids spans all 60 subframe samples.
//
// For n_pulses = 6, track 0 covers positions 0,8,16,24,32,40,48,56 (k=0..7);
// track 5 covers 5,13,21,29,37,45,53 (k=0..6, k=7 lands at 61 which is
// out of range and skipped). Per-track greedy pick followed by two
// passes of coordinate-descent refinement.

/// Compute the absolute position of an MP-MLQ pulse on `track` with slot
/// `k` and `grid` offset, using stride = `n_pulses` so each track's 8
/// slots interleave with the other tracks' and the combined pulse set
/// hits `n_pulses × 8` distinct positions — 48 for 6 pulses, 40 for 5.
/// The grid bit shifts all positions by 1 (odd/even grid).
fn mpmlq_pos_of(track: usize, k: u32, grid: u8, n_pulses: usize) -> usize {
    track + n_pulses * k as usize + grid as usize
}

/// MP-MLQ multipulse search. Returns `(positions, signs, grid)` with:
///
/// - `positions[t]` — 3-bit slot index on track `t`,
/// - `signs[t]` — `+1` or `-1`,
/// - `grid` — the shared 0/1 grid offset for this subframe.
///
/// After the per-track greedy pick, two coordinate-descent passes
/// re-optimise each pulse given the rest fixed, mirroring the ACELP
/// refinement.
fn mpmlq_pulse_search(
    target: &[f32; SUBFRAME_SIZE],
    h: &[f32],
    n_pulses: usize,
) -> ([u32; MPMLQ_PULSES_ODD], [i32; MPMLQ_PULSES_ODD], u8) {
    debug_assert!(n_pulses <= MPMLQ_PULSES_ODD);
    let d = compute_correlations(target, h);

    let mut best_grid = 0u8;
    let mut best_err = f32::INFINITY;
    let mut best_positions = [0u32; MPMLQ_PULSES_ODD];
    let mut best_signs = [1i32; MPMLQ_PULSES_ODD];

    for grid in 0..2u8 {
        let mut positions = [0u32; MPMLQ_PULSES_ODD];
        let mut signs = [1i32; MPMLQ_PULSES_ODD];

        // Pass 1: per-track greedy pick.
        for track in 0..n_pulses {
            let mut best_score = 0.0f32;
            let mut best_k = 0u32;
            let mut best_sign = 1i32;
            for k in 0..8u32 {
                let pos = mpmlq_pos_of(track, k, grid, n_pulses);
                if pos >= SUBFRAME_SIZE {
                    continue;
                }
                let ap = autocorr_at(h, pos);
                if ap < 1e-8 {
                    continue;
                }
                let dv = d[pos];
                let score = dv * dv / ap;
                if score > best_score {
                    best_score = score;
                    best_k = k;
                    best_sign = if dv >= 0.0 { 1 } else { -1 };
                }
            }
            positions[track] = best_k;
            signs[track] = best_sign;
        }

        // Score: reconstruction error against target. (Unlike ACELP, the
        // MP-MLQ pulse layout uses stride = n_pulses which puts pulses
        // at adjacent positions; a coordinate-descent refinement on that
        // layout tends to flip signs in partial-cancellation patterns
        // that the greedy pass already found a better configuration for,
        // so we skip it here.)
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for track in 0..n_pulses {
            let pos = mpmlq_pos_of(track, positions[track], grid, n_pulses);
            if pos < SUBFRAME_SIZE {
                let sgn = signs[track] as f32;
                for n in pos..SUBFRAME_SIZE {
                    syn[n] += sgn * h[n - pos];
                }
            }
        }
        let mut err = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            let e = target[n] - syn[n];
            err += e * e;
        }
        if err < best_err {
            best_err = err;
            best_grid = grid;
            best_positions = positions;
            best_signs = signs;
        }
    }

    (best_positions, best_signs, best_grid)
}

/// Place MP-MLQ pulses in the subframe buffer using the track layout used
/// by [`mpmlq_pulse_search`] (track `t ∈ 0..n_pulses`, stride
/// [`MPMLQ_STRIDE`]).
pub(crate) fn mpmlq_place_pulses(
    positions: &[u32; MPMLQ_PULSES_ODD],
    signs: &[i32; MPMLQ_PULSES_ODD],
    n_pulses: usize,
    grid: u8,
    out: &mut [f32; SUBFRAME_SIZE],
) {
    out.fill(0.0);
    for t in 0..n_pulses {
        let k = positions[t];
        let pos = mpmlq_pos_of(t, k, grid, n_pulses);
        if pos < SUBFRAME_SIZE {
            out[pos] = signs[t] as f32;
        }
    }
}

/// Pack `n_pulses` MP-MLQ pulses into the low `n_pulses * 4` bits of the
/// output: `[pos0_3 | sign0_1 | pos1_3 | sign1_1 | ...]`. The caller is
/// responsible for budgeting the correct total bit count (24 bits for 6
/// pulses, 20 bits for 5 pulses).
fn pack_mpmlq_pulses(pulses: &MpMlqPulses) -> u32 {
    let mut v = 0u32;
    for t in 0..pulses.n_pulses as usize {
        let pos = pulses.positions[t] & 0x7;
        let sign_bit = if pulses.signs[t] < 0 { 1u32 } else { 0 };
        let slot = (pos << 1) | sign_bit; // 4 bits per pulse
        v |= slot << (t * 4);
    }
    v
}

/// Inverse of [`pack_mpmlq_pulses`]. Produces a populated [`MpMlqPulses`]
/// with `n_pulses` entries.
pub(crate) fn unpack_mpmlq_pulses(v: u32, n_pulses: usize) -> MpMlqPulses {
    let mut out = MpMlqPulses {
        n_pulses: n_pulses as u8,
        ..MpMlqPulses::default()
    };
    for t in 0..n_pulses {
        let slot = (v >> (t * 4)) & 0xF;
        let pos = (slot >> 1) & 0x7;
        let sign_bit = slot & 0x1;
        out.positions[t] = pos;
        out.signs[t] = if sign_bit == 1 { -1 } else { 1 };
    }
    out
}

// ---------- gain quantisation ----------
//
// 12-bit joint ACB/FCB gain index laid out LSB→MSB as:
//
//   bits 0..3  (4 bits): ACB gain — 16 uniform levels in [0.0, 1.25]
//   bits 4..10 (7 bits): FCB gain magnitude on log2 scale, ~0.5 dB step
//   bit  11    (1 bit):  FCB gain sign (1 = negative)
//
// The FCB log2 range spans 12 octaves (roughly 1/1024 .. 16 relative to
// pulse amplitude ±1), with 128 levels giving ~0.80 dB resolution. Using
// a tight range here is the biggest single gain-VQ improvement vs the
// original 19-octave / 256-level mapping (which had ~1.8 dB resolution
// across a much wider range than real encoded frames use).

const ACB_GAIN_BITS: u32 = 4;
const ACB_GAIN_LEVELS: u32 = 1 << ACB_GAIN_BITS; // 16
const ACB_GAIN_MAX: f32 = 1.25;
const FCB_GAIN_BITS: u32 = 7;
const FCB_GAIN_LEVELS: u32 = 1 << FCB_GAIN_BITS; // 128
const FCB_GAIN_LOG2_MIN: f32 = -10.0; // 2^-10 ≈ 1e-3
const FCB_GAIN_LOG2_MAX: f32 = 2.0; //   2^+2  = 4.0
const FCB_SIGN_SHIFT: u32 = ACB_GAIN_BITS + FCB_GAIN_BITS;

fn quantise_gain(g_adapt: f32, g_fixed: f32) -> u32 {
    let acb_idx = quantise_scalar(
        g_adapt.clamp(0.0, ACB_GAIN_MAX),
        0.0,
        ACB_GAIN_MAX,
        ACB_GAIN_LEVELS,
    );
    let sign = if g_fixed < 0.0 { 1u32 } else { 0 };
    let mag = g_fixed.abs().max(2.0f32.powf(FCB_GAIN_LOG2_MIN));
    let log2_mag = mag.log2();
    let fcb_idx = quantise_scalar(
        log2_mag,
        FCB_GAIN_LOG2_MIN,
        FCB_GAIN_LOG2_MAX,
        FCB_GAIN_LEVELS,
    );
    acb_idx | (fcb_idx << ACB_GAIN_BITS) | (sign << FCB_SIGN_SHIFT)
}

/// Refine the joint (ACB, FCB) gain pair by scanning a small
/// neighbourhood of the initial indices and picking the codeword that
/// minimises `sum((target - g_acb * adapt_filt - g_fcb * fcb_filt)^2)`.
///
/// Returns the 12-bit `gain_idx` packing `acb_idx | (fcb_idx <<
/// ACB_GAIN_BITS) | (sign_bit << FCB_SIGN_SHIFT)`.
fn refine_gain_pair(
    acb_idx0: u32,
    fcb_idx0: u32,
    sign_bit: u32,
    adapt_filt: &[f32; SUBFRAME_SIZE],
    fcb_filt: &[f32; SUBFRAME_SIZE],
    target: &[f32; SUBFRAME_SIZE],
) -> u32 {
    // Narrow neighbourhood around the initial quantisation: scanning
    // too far in ACB breaks the pulse-search assumption that the FCB
    // residual is `target - g_acb_q * adapt_filt`.
    let acb_lo = acb_idx0.saturating_sub(1);
    let acb_hi = (acb_idx0 + 1).min(ACB_GAIN_LEVELS - 1);
    let fcb_lo = fcb_idx0.saturating_sub(4);
    let fcb_hi = (fcb_idx0 + 4).min(FCB_GAIN_LEVELS - 1);
    let sign_f = if sign_bit == 1 { -1.0f32 } else { 1.0 };
    let mut best_err = f32::INFINITY;
    let mut best_gi = acb_idx0 | (fcb_idx0 << ACB_GAIN_BITS) | (sign_bit << FCB_SIGN_SHIFT);
    for acb in acb_lo..=acb_hi {
        let g_a = dequantise_scalar(acb, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);
        for fcb in fcb_lo..=fcb_hi {
            let log2_mag =
                dequantise_scalar(fcb, FCB_GAIN_LOG2_MIN, FCB_GAIN_LOG2_MAX, FCB_GAIN_LEVELS);
            let g_f = sign_f * 2.0f32.powf(log2_mag);
            let mut err = 0.0f32;
            for n in 0..SUBFRAME_SIZE {
                let e = target[n] - g_a * adapt_filt[n] - g_f * fcb_filt[n];
                err += e * e;
            }
            if err < best_err {
                best_err = err;
                best_gi = acb | (fcb << ACB_GAIN_BITS) | (sign_bit << FCB_SIGN_SHIFT);
            }
        }
    }
    best_gi
}

pub(crate) fn dequantise_gain(idx: u32) -> (f32, f32) {
    let acb_idx = idx & (ACB_GAIN_LEVELS - 1);
    let fcb_idx = (idx >> ACB_GAIN_BITS) & (FCB_GAIN_LEVELS - 1);
    let sign = (idx >> FCB_SIGN_SHIFT) & 0x1;
    let g_adapt = dequantise_scalar(acb_idx, 0.0, ACB_GAIN_MAX, ACB_GAIN_LEVELS);
    let log2_mag = dequantise_scalar(
        fcb_idx,
        FCB_GAIN_LOG2_MIN,
        FCB_GAIN_LOG2_MAX,
        FCB_GAIN_LEVELS,
    );
    let mag = 2.0f32.powf(log2_mag);
    let g_fixed = if sign == 1 { -mag } else { mag };
    (g_adapt, g_fixed)
}

// ---------- filtering helpers ----------

/// Impulse response of the 1/A_weighted(z) filter, length `n`.
fn impulse_response(a_weighted: &[f32; LPC_ORDER + 1], n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n];
    let mut mem = [0.0f32; LPC_ORDER];
    for i in 0..n {
        let e = if i == 0 { 1.0 } else { 0.0 };
        let mut s = e;
        for k in 0..LPC_ORDER {
            s -= a_weighted[k + 1] * mem[k];
        }
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = s;
        h[i] = s;
    }
    h
}

/// Causal convolution `y = x * h` truncated to length of x.
fn conv_causal(x: &[f32; SUBFRAME_SIZE], h: &[f32]) -> [f32; SUBFRAME_SIZE] {
    let mut y = [0.0f32; SUBFRAME_SIZE];
    for n in 0..SUBFRAME_SIZE {
        let mut acc = 0.0f32;
        for k in 0..=n {
            if k < h.len() {
                acc += x[n - k] * h[k];
            }
        }
        y[n] = acc;
    }
    y
}

fn lsq_gain(pred: &[f32; SUBFRAME_SIZE], target: &[f32]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 1e-6f32;
    for n in 0..SUBFRAME_SIZE {
        num += pred[n] * target[n];
        den += pred[n] * pred[n];
    }
    num / den
}

/// Advance the 1/A(z) synthesis filter memory with `exc` so cross-subframe
/// state stays in sync with what the decoder will render.
fn advance_syn_mem(
    a: &[f32; LPC_ORDER + 1],
    exc: &[f32; SUBFRAME_SIZE],
    mem: &mut [f32; LPC_ORDER],
) {
    for i in 0..SUBFRAME_SIZE {
        let mut s = exc[i];
        for k in 0..LPC_ORDER {
            s -= a[k + 1] * mem[k];
        }
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = s;
    }
}

/// Zero-input response of the 1/A(z) synthesis filter over `n` samples
/// starting from the given filter memory `mem` (input = zero).
fn zero_input_response(a: &[f32; LPC_ORDER + 1], mem: &[f32; LPC_ORDER], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let mut m = *mem;
    for i in 0..n {
        let mut s = 0.0f32;
        for k in 0..LPC_ORDER {
            s -= a[k + 1] * m[k];
        }
        for k in (1..LPC_ORDER).rev() {
            m[k] = m[k - 1];
        }
        m[0] = s;
        out[i] = s;
    }
    out
}

/// Open-loop adaptive-codebook lag search. Given the current synthesis
/// target (= input signal minus zero-input response) and the synthesis
/// filter impulse response `h`, pick the integer lag `L ∈ [PITCH_MIN,
/// PITCH_MAX]` whose ACB prediction convolved with `h` most closely
/// matches the target in the least-squares sense (maximises `<target,
/// h*acb>^2 / ||h*acb||^2`).
fn open_loop_acb_lag(target: &[f32; SUBFRAME_SIZE], history: &[f32], h: &[f32]) -> i32 {
    let mut best_score = -f32::INFINITY;
    let mut best_lag = PITCH_MIN as i32;
    let mut cand = [0.0f32; SUBFRAME_SIZE];
    for lag in PITCH_MIN..=PITCH_MAX {
        copy_adaptive(history, lag as i32, &mut cand);
        let filtered = conv_causal(&cand, h);
        let mut num = 0.0f32;
        let mut den = 1e-6f32;
        for n in 0..SUBFRAME_SIZE {
            num += target[n] * filtered[n];
            den += filtered[n] * filtered[n];
        }
        if den < 1e-6 {
            continue;
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_lag = lag as i32;
        }
    }
    best_lag
}

// ---------- bit packing ----------

/// Bit writer that appends bits in LSB-first order within each byte,
/// matching [`BitReader`]'s consumption order.
struct LsbBitWriter {
    data: Vec<u8>,
    byte_pos: usize,
    bit_pos: u32,
}

impl LsbBitWriter {
    fn with_len(n: usize) -> Self {
        Self {
            data: vec![0; n],
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn write(&mut self, mut value: u32, n: u32) {
        let mut remaining = n;
        while remaining > 0 {
            let take = (8 - self.bit_pos).min(remaining);
            let chunk = (value & ((1u32 << take) - 1)) as u8;
            self.data[self.byte_pos] |= chunk << self.bit_pos;
            self.bit_pos += take;
            value >>= take;
            remaining -= take;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
    }
}

fn pack_acelp_frame(f: &FrameFields) -> Vec<u8> {
    let mut w = LsbBitWriter::with_len(ACELP_PAYLOAD_BYTES);
    // Field 0: RATE = 01.
    w.write(0b01, 2);
    // LSP.
    w.write(f.lsp_idx[0], 8);
    w.write(f.lsp_idx[1], 8);
    w.write(f.lsp_idx[2], 8);
    // ACL.
    w.write(f.acl[0] as u32 & 0x7F, 7);
    w.write(f.acl[1] as u32 & 0x3, 2);
    w.write(f.acl[2] as u32 & 0x7F, 7);
    w.write(f.acl[3] as u32 & 0x3, 2);
    // GAIN.
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(f.gain[s] & 0xFFF, 12);
    }
    // GRID.
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(f.grid[s] as u32, 1);
    }
    // FCB.
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(f.fcb[s] & 0xFFFF, 16);
    }
    w.data
}

/// Pack an MP-MLQ frame (6.3 kbit/s) into a 24-byte payload.
///
/// Layout matches [`MPMLQ_FIELDS`]: 2-bit RATE=00 + 3×8-bit LSP + 7+2+7+2
/// lag bits + 4×12-bit GAIN + 4 grid bits + {24,20,24,20}-bit MP pulses +
/// 8-bit zero padding = 192 bits = 24 bytes.
fn pack_mpmlq_frame(f: &MpMlqFrameFields) -> Vec<u8> {
    let mut w = LsbBitWriter::with_len(MPMLQ_PAYLOAD_BYTES);
    // RATE = 00 (MP-MLQ discriminator).
    w.write(0b00, 2);
    // LSP.
    w.write(f.lsp_idx[0], 8);
    w.write(f.lsp_idx[1], 8);
    w.write(f.lsp_idx[2], 8);
    // ACL (same widths as ACELP).
    w.write(f.acl[0] as u32 & 0x7F, 7);
    w.write(f.acl[1] as u32 & 0x3, 2);
    w.write(f.acl[2] as u32 & 0x7F, 7);
    w.write(f.acl[3] as u32 & 0x3, 2);
    // GAIN (4 × 12 bits).
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(f.gain[s] & 0xFFF, 12);
    }
    // GRID (4 × 1 bit).
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(f.grid[s] as u32, 1);
    }
    // MP pulses per subframe: 6 × 4 bits (odd) or 5 × 4 bits (even).
    for s in 0..SUBFRAMES_PER_FRAME {
        let n = f.mp[s].n_pulses as u32;
        let bits = n * (MPMLQ_POS_BITS + MPMLQ_SIGN_BITS);
        let packed = pack_mpmlq_pulses(&f.mp[s]);
        w.write(packed, bits);
    }
    // RSVD (8 bits of zero padding) to hit 24 bytes exactly.
    w.write(0, 8);
    w.data
}

// ---------- decoder ----------
//
// The decoder is a stateful synthesiser that mirrors the encoder's
// analysis-by-synthesis path. All of the per-frame state (previous LSP,
// excitation history, LPC filter memory) persists across packets so that
// a sequence of frames reconstructs without the per-frame transients that
// a stateless decoder would introduce at every 30 ms boundary.

/// Persistent synthesis state shared by both the stateful [`SynthesisState::decode_acelp`]
/// and [`SynthesisState::decode_mpmlq`] entry points and by the framework-facing
/// [`crate::G7231Decoder`]. The encoder holds one of these too so that its
/// analysis-by-synthesis loop sees the exact signal the decoder will.
pub struct SynthesisState {
    prev_lsp: [f32; LPC_ORDER],
    exc_history: [f32; PITCH_MAX + SUBFRAME_SIZE],
    syn_mem: [f32; LPC_ORDER],
}

impl SynthesisState {
    pub fn new() -> Self {
        let mut prev_lsp = [0.0f32; LPC_ORDER];
        let step = std::f32::consts::PI / (LPC_ORDER as f32 + 1.0);
        for k in 0..LPC_ORDER {
            prev_lsp[k] = ((k as f32 + 1.0) * step).cos();
        }
        Self {
            prev_lsp,
            exc_history: [0.0; PITCH_MAX + SUBFRAME_SIZE],
            syn_mem: [0.0; LPC_ORDER],
        }
    }

    /// Reset to the silent-LSP boot state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Core synthesis kernel: given the already-decoded per-subframe lags,
    /// excitation pulses, grids, and joint gains, render 240 samples into
    /// `pcm` while advancing `self`.
    fn synthesise(
        &mut self,
        lsp_q: &[f32; LPC_ORDER],
        lags: &[i32; SUBFRAMES_PER_FRAME],
        grid: &[u8; SUBFRAMES_PER_FRAME],
        gain: &[u32; SUBFRAMES_PER_FRAME],
        pulses_per_subframe: &[[f32; SUBFRAME_SIZE]; SUBFRAMES_PER_FRAME],
        pcm: &mut [f32; FRAME_SIZE_SAMPLES],
    ) {
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, &self.prev_lsp, lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);

            let mut adaptive = [0.0f32; SUBFRAME_SIZE];
            copy_adaptive(&self.exc_history, lags[s], &mut adaptive);

            let (g_adapt, g_fixed) = dequantise_gain(gain[s]);
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            let pulses = &pulses_per_subframe[s];
            for n in 0..SUBFRAME_SIZE {
                exc[n] = g_adapt * adaptive[n] + g_fixed * pulses[n];
            }
            let _ = grid[s]; // grid is encoded in `pulses_per_subframe`.

            // LPC synthesis: 1/A(z), writing into pcm while advancing syn_mem.
            for i in 0..SUBFRAME_SIZE {
                let mut y = exc[i];
                for k in 0..LPC_ORDER {
                    y -= a_sub[k + 1] * self.syn_mem[k];
                }
                for k in (1..LPC_ORDER).rev() {
                    self.syn_mem[k] = self.syn_mem[k - 1];
                }
                self.syn_mem[0] = y;
                pcm[s * SUBFRAME_SIZE + i] = y;
            }

            // Advance excitation history.
            self.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.exc_history.len() - SUBFRAME_SIZE;
            self.exc_history[tail..].copy_from_slice(&exc);
        }
        self.prev_lsp = *lsp_q;
    }

    /// Decode one ACELP (5.3 kbit/s) frame into 240 PCM samples.
    pub fn decode_acelp(&mut self, payload: &[u8]) -> Result<[i16; FRAME_SIZE_SAMPLES]> {
        if payload.len() < ACELP_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "G.723.1 decoder: ACELP payload smaller than 20 bytes",
            ));
        }
        let mut br = BitReader::new(&payload[..ACELP_PAYLOAD_BYTES]);
        let rate = br.read_u32(2)?;
        if rate != 0b01 {
            return Err(Error::invalid(format!(
                "G.723.1 decoder: expected RATE=01 (ACELP), got {rate:02b}"
            )));
        }
        let lsp_idx = [br.read_u32(8)?, br.read_u32(8)?, br.read_u32(8)?];
        let lsp_q = dequantise_lsp(&lsp_idx);
        let acl0 = br.read_u32(7)?;
        let acl1 = br.read_u32(2)?;
        let acl2 = br.read_u32(7)?;
        let acl3 = br.read_u32(2)?;
        let mut gain = [0u32; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            gain[s] = br.read_u32(12)?;
        }
        let mut grid = [0u8; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            grid[s] = br.read_u32(1)? as u8;
        }
        let mut fcb = [0u32; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            fcb[s] = br.read_u32(16)?;
        }

        let lag0 = decode_abs_lag(acl0);
        let lag1 = decode_delta_lag(acl1, lag0);
        let lag2 = decode_abs_lag(acl2);
        let lag3 = decode_delta_lag(acl3, lag2);
        let lags = [lag0, lag1, lag2, lag3];

        let mut pulses_per_subframe = [[0.0f32; SUBFRAME_SIZE]; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            let (positions, signs) = unpack_fcb_bits(fcb[s]);
            place_pulses(&positions, signs, grid[s], &mut pulses_per_subframe[s]);
        }

        let mut pcm_f = [0.0f32; FRAME_SIZE_SAMPLES];
        self.synthesise(
            &lsp_q,
            &lags,
            &grid,
            &gain,
            &pulses_per_subframe,
            &mut pcm_f,
        );
        Ok(to_i16_frame(&pcm_f))
    }

    /// Decode one MP-MLQ (6.3 kbit/s) frame into 240 PCM samples.
    pub fn decode_mpmlq(&mut self, payload: &[u8]) -> Result<[i16; FRAME_SIZE_SAMPLES]> {
        if payload.len() < MPMLQ_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "G.723.1 decoder: MP-MLQ payload smaller than 24 bytes",
            ));
        }
        let mut br = BitReader::new(&payload[..MPMLQ_PAYLOAD_BYTES]);
        let rate = br.read_u32(2)?;
        if rate != 0b00 {
            return Err(Error::invalid(format!(
                "G.723.1 decoder: expected RATE=00 (MP-MLQ), got {rate:02b}"
            )));
        }
        let lsp_idx = [br.read_u32(8)?, br.read_u32(8)?, br.read_u32(8)?];
        let lsp_q = dequantise_lsp(&lsp_idx);
        let acl0 = br.read_u32(7)?;
        let acl1 = br.read_u32(2)?;
        let acl2 = br.read_u32(7)?;
        let acl3 = br.read_u32(2)?;
        let mut gain = [0u32; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            gain[s] = br.read_u32(12)?;
        }
        let mut grid = [0u8; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            grid[s] = br.read_u32(1)? as u8;
        }
        let mut pulses_per_subframe = [[0.0f32; SUBFRAME_SIZE]; SUBFRAMES_PER_FRAME];
        for s in 0..SUBFRAMES_PER_FRAME {
            let n = if s % 2 == 0 {
                MPMLQ_PULSES_ODD
            } else {
                MPMLQ_PULSES_EVEN
            };
            let bits = (n as u32) * (MPMLQ_POS_BITS + MPMLQ_SIGN_BITS);
            let v = br.read_u32(bits)?;
            let mp = unpack_mpmlq_pulses(v, n);
            mpmlq_place_pulses(
                &mp.positions,
                &mp.signs,
                n,
                grid[s],
                &mut pulses_per_subframe[s],
            );
        }
        let _rsvd = br.read_u32(8)?;

        let lag0 = decode_abs_lag(acl0);
        let lag1 = decode_delta_lag(acl1, lag0);
        let lag2 = decode_abs_lag(acl2);
        let lag3 = decode_delta_lag(acl3, lag2);
        let lags = [lag0, lag1, lag2, lag3];

        let mut pcm_f = [0.0f32; FRAME_SIZE_SAMPLES];
        self.synthesise(
            &lsp_q,
            &lags,
            &grid,
            &gain,
            &pulses_per_subframe,
            &mut pcm_f,
        );
        Ok(to_i16_frame(&pcm_f))
    }
}

impl Default for SynthesisState {
    fn default() -> Self {
        Self::new()
    }
}

fn to_i16_frame(pcm: &[f32; FRAME_SIZE_SAMPLES]) -> [i16; FRAME_SIZE_SAMPLES] {
    let mut out = [0i16; FRAME_SIZE_SAMPLES];
    for (i, &v) in pcm.iter().enumerate() {
        let s = (v * 32_767.0).clamp(-32_768.0, 32_767.0);
        out[i] = s as i16;
    }
    out
}

/// Convenience stateless wrapper around [`SynthesisState::decode_acelp`] — each
/// call allocates a fresh decoder state, so concatenating the output of
/// multiple calls introduces transient artefacts at every 30 ms boundary.
/// Callers chasing high SNR across a multi-frame stream should instantiate
/// [`SynthesisState`] once and call [`SynthesisState::decode_acelp`] per
/// frame.
pub fn decode_acelp_local(payload: &[u8]) -> Result<Vec<i16>> {
    let mut st = SynthesisState::new();
    Ok(st.decode_acelp(payload)?.to_vec())
}

/// Convenience stateless wrapper around [`SynthesisState::decode_mpmlq`].
/// See [`decode_acelp_local`] for the caveat about decoder state across
/// frames.
pub fn decode_mpmlq_local(payload: &[u8]) -> Result<Vec<i16>> {
    let mut st = SynthesisState::new();
    Ok(st.decode_mpmlq(payload)?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, Frame, SampleFormat, TimeBase};

    fn params(bit_rate: Option<u64>) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.sample_rate = Some(SAMPLE_RATE_HZ);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = bit_rate;
        p
    }

    fn audio_frame(samples: &[i16]) -> Frame {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE_HZ,
            samples: samples.len() as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, SAMPLE_RATE_HZ as i64),
            data: vec![bytes],
        })
    }

    fn sine_mixture(frames: usize) -> Vec<i16> {
        let n = frames * FRAME_SIZE_SAMPLES;
        let mut out = Vec::with_capacity(n);
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE_HZ as f32;
            let v = (two_pi * 220.0 * t).sin() * 0.45
                + (two_pi * 660.0 * t).sin() * 0.25
                + (two_pi * 1100.0 * t).sin() * 0.15;
            out.push((v * 20_000.0) as i16);
        }
        out
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut p = params(None);
        p.sample_rate = Some(16_000);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn rejects_stereo() {
        let mut p = params(None);
        p.channels = Some(2);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn accepts_6300_bitrate_request() {
        // MP-MLQ path is now implemented.
        assert!(make_encoder(&params(Some(6300))).is_ok());
    }

    #[test]
    fn rejects_invalid_bitrate_request() {
        // Bit rates outside the two codec modes stay Unsupported.
        let result = make_encoder(&params(Some(8000)));
        let err = match result {
            Ok(_) => panic!("expected Unsupported, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn accepts_5300_bitrate_request() {
        assert!(make_encoder(&params(Some(5300))).is_ok());
    }

    #[test]
    fn default_bitrate_is_mpmlq() {
        // No bit_rate hint defaults to 6.3 kbit/s MP-MLQ.
        let enc = make_encoder(&params(None)).unwrap();
        assert_eq!(enc.output_params().bit_rate, Some(6_300));
    }

    #[test]
    fn silence_encodes_to_20_byte_acelp_packet() {
        let mut enc = make_encoder(&params(Some(5300))).unwrap();
        let pcm = vec![0i16; FRAME_SIZE_SAMPLES];
        enc.send_frame(&audio_frame(&pcm)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.data.len(), ACELP_PAYLOAD_BYTES);
        assert_eq!(pkt.data[0] & 0b11, 0b01, "discriminator must be 01");
        assert_eq!(pkt.duration, Some(FRAME_SIZE_SAMPLES as i64));
    }

    #[test]
    fn silence_encodes_to_24_byte_mpmlq_packet() {
        let mut enc = make_encoder(&params(Some(6300))).unwrap();
        let pcm = vec![0i16; FRAME_SIZE_SAMPLES];
        enc.send_frame(&audio_frame(&pcm)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.data.len(), MPMLQ_PAYLOAD_BYTES);
        assert_eq!(pkt.data[0] & 0b11, 0b00, "discriminator must be 00");
        assert_eq!(pkt.duration, Some(FRAME_SIZE_SAMPLES as i64));
    }

    #[test]
    fn scaffold_decoder_accepts_acelp_encoder_output() {
        let mut enc = make_encoder(&params(Some(5300))).unwrap();
        let pcm = sine_mixture(2);
        enc.send_frame(&audio_frame(&pcm)).unwrap();

        let mut reg = oxideav_codec::CodecRegistry::new();
        crate::register(&mut reg);
        let mut dec = reg
            .make_decoder(&params(None))
            .expect("decoder factory must exist");

        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).unwrap();
            let f = dec.receive_frame().unwrap();
            // Scaffold decoder emits silence; just assert it produces a
            // well-shaped audio frame of the right size.
            match f {
                Frame::Audio(af) => {
                    assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
                    assert_eq!(af.sample_rate, SAMPLE_RATE_HZ);
                    assert_eq!(af.channels, 1);
                }
                _ => panic!("expected audio frame"),
            }
        }
    }

    #[test]
    fn scaffold_decoder_accepts_mpmlq_encoder_output() {
        let mut enc = make_encoder(&params(Some(6300))).unwrap();
        let pcm = sine_mixture(2);
        enc.send_frame(&audio_frame(&pcm)).unwrap();

        let mut reg = oxideav_codec::CodecRegistry::new();
        crate::register(&mut reg);
        let mut dec = reg
            .make_decoder(&params(None))
            .expect("decoder factory must exist");

        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).unwrap();
            let f = dec.receive_frame().unwrap();
            match f {
                Frame::Audio(af) => {
                    assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
                    assert_eq!(af.sample_rate, SAMPLE_RATE_HZ);
                    assert_eq!(af.channels, 1);
                }
                _ => panic!("expected audio frame"),
            }
        }
    }

    #[test]
    fn roundtrip_sine_has_nonzero_energy_via_local_decoder() {
        // Encode a sum-of-sines signal, decode via the encoder's own
        // reference inverse (`decode_acelp_local`), and assert that the
        // output has finite samples and non-zero energy. The framework's
        // scaffold decoder always emits silence, so a full spec-compliant
        // round-trip PSNR check is not yet meaningful — see the module
        // docstring for the full caveat.
        const FRAMES: usize = 8;
        let input = sine_mixture(FRAMES);
        let mut enc = make_encoder(&params(Some(5300))).unwrap();
        enc.send_frame(&audio_frame(&input)).unwrap();
        enc.flush().unwrap();

        let mut dec = SynthesisState::new();
        let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SIZE_SAMPLES);
        let mut n_packets = 0;
        while let Ok(pkt) = enc.receive_packet() {
            n_packets += 1;
            let frame_pcm = dec.decode_acelp(&pkt.data).unwrap();
            assert_eq!(frame_pcm.len(), FRAME_SIZE_SAMPLES);
            for &s in &frame_pcm {
                assert!((s as i32).abs() <= i16::MAX as i32 + 1);
            }
            decoded.extend_from_slice(&frame_pcm);
        }
        assert_eq!(n_packets, FRAMES);

        // All samples are finite (trivially — they're i16). Check energy.
        let energy: f64 = decoded
            .iter()
            .map(|&s| {
                let x = s as f64;
                x * x
            })
            .sum();
        assert!(
            energy > 0.0,
            "decoded signal has zero energy; encoder produced silence"
        );

        // PSNR-ish sanity: reconstructed signal energy is at least 1% of
        // the input signal energy. Exact speech-codec SNR (10–15 dB) is
        // not achievable with the simplified codebooks here, but some
        // non-trivial reconstruction IS expected.
        let input_energy: f64 = input
            .iter()
            .map(|&s| {
                let x = s as f64;
                x * x
            })
            .sum();
        assert!(
            energy >= 0.01 * input_energy,
            "decoded energy {:.3e} is too small vs input {:.3e}",
            energy,
            input_energy
        );
    }

    #[test]
    fn mpmlq_roundtrip_sine_has_nonzero_energy_via_local_decoder() {
        // Parallel to the ACELP round-trip test, for the 6.3 kbit/s MP-MLQ
        // path. Encode a sum-of-sines signal at 6.3 kbit/s, decode via
        // `decode_mpmlq_local`, assert non-trivial reconstructed energy
        // (>= 1% of input energy, matching the ACELP bar).
        const FRAMES: usize = 8;
        let input = sine_mixture(FRAMES);
        let mut enc = make_encoder(&params(Some(6300))).unwrap();
        enc.send_frame(&audio_frame(&input)).unwrap();
        enc.flush().unwrap();

        let mut dec = SynthesisState::new();
        let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SIZE_SAMPLES);
        let mut n_packets = 0;
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), MPMLQ_PAYLOAD_BYTES);
            assert_eq!(pkt.data[0] & 0b11, 0b00);
            n_packets += 1;
            let frame_pcm = dec.decode_mpmlq(&pkt.data).unwrap();
            assert_eq!(frame_pcm.len(), FRAME_SIZE_SAMPLES);
            for &s in &frame_pcm {
                assert!((s as i32).abs() <= i16::MAX as i32 + 1);
            }
            decoded.extend_from_slice(&frame_pcm);
        }
        assert_eq!(n_packets, FRAMES);

        let energy: f64 = decoded.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0, "MP-MLQ decoded signal has zero energy");

        let input_energy: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(
            energy >= 0.01 * input_energy,
            "MP-MLQ decoded energy {:.3e} is too small vs input {:.3e}",
            energy,
            input_energy
        );
    }

    /// Voiced test source: 150 Hz fundamental with three harmonics, peaking
    /// at ~20 000 on i16, reasonably representative of the low-frequency
    /// voiced speech the codec is tuned for.
    fn voiced_signal(frames: usize) -> Vec<i16> {
        let n = frames * FRAME_SIZE_SAMPLES;
        let mut out = Vec::with_capacity(n);
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE_HZ as f32;
            let v = (two_pi * 150.0 * t).sin() * 0.50
                + (two_pi * 300.0 * t).sin() * 0.25
                + (two_pi * 450.0 * t).sin() * 0.15
                + (two_pi * 900.0 * t).sin() * 0.08;
            out.push((v * 20_000.0) as i16);
        }
        out
    }

    #[test]
    fn silence_encodes_to_near_silence() {
        // Regression for the `lsp_to_lpc` p/2 buffer-truncation bug. Two
        // 30 ms frames of zero PCM should decode to near-zero output —
        // with the old 6th-order truncated filter the silent LSP had
        // |h_peak| > 1e19 and the decoder saturated at ±32768. A stable
        // 10th-order LPC keeps the reconstruction bounded by quantisation
        // noise (~50 LSBs in practice).
        let mut enc = make_encoder(&params(Some(6300))).unwrap();
        let pcm = vec![0i16; FRAME_SIZE_SAMPLES * 2];
        enc.send_frame(&audio_frame(&pcm)).unwrap();
        enc.flush().unwrap();
        let mut dec = SynthesisState::new();
        while let Ok(pkt) = enc.receive_packet() {
            let out = dec.decode_mpmlq(&pkt.data).unwrap();
            let max = out.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
            assert!(
                max < 1000,
                "silence decoded to max |s|={max}, expected <1000"
            );
        }
    }

    #[test]
    fn acelp_roundtrip_voiced_psnr_floor() {
        // ACELP (5.3 kbit/s) equivalent of `mpmlq_roundtrip_voiced_psnr_floor`.
        const FRAMES: usize = 16;
        let input = voiced_signal(FRAMES);
        let mut enc = make_encoder(&params(Some(5300))).unwrap();
        enc.send_frame(&audio_frame(&input)).unwrap();
        enc.flush().unwrap();

        let mut dec = SynthesisState::new();
        let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SIZE_SAMPLES);
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), ACELP_PAYLOAD_BYTES);
            assert_eq!(pkt.data[0] & 0b11, 0b01, "discriminator must be 01");
            decoded.extend_from_slice(&dec.decode_acelp(&pkt.data).unwrap());
        }
        assert_eq!(decoded.len(), input.len());

        let n = input.len();
        let mut mse = 0.0f64;
        for i in 0..n {
            let e = decoded[i] as f64 - input[i] as f64;
            mse += e * e;
        }
        mse /= n as f64;
        let peak = 32_767.0f64;
        let psnr = 10.0 * (peak * peak / mse).log10();
        let mut sig_e = 0.0f64;
        for &s in &input {
            sig_e += (s as f64).powi(2);
        }
        sig_e /= n as f64;
        let snr = 10.0 * (sig_e / mse.max(1e-10)).log10();
        eprintln!("acelp_roundtrip_voiced: PSNR = {psnr:.2} dB, SNR = {snr:.2} dB");
        assert!(
            psnr >= 15.0,
            "ACELP voiced-signal PSNR = {psnr:.2} dB, expected >= 15 dB"
        );
    }

    #[test]
    fn mpmlq_roundtrip_voiced_psnr_floor() {
        // Full 6.3 kbit/s MP-MLQ encode -> stateful decode -> PSNR probe on
        // a voiced 150 Hz signal. The encoder runs analysis-by-synthesis
        // against a shadow decoder state (see `AnalysisState::decoder`)
        // and the decoder here holds live state across frames, so the
        // result is steady-state PSNR without the per-packet cold-start
        // transients the earlier stateless helper introduced.
        const FRAMES: usize = 16;
        let input = voiced_signal(FRAMES);
        let mut enc = make_encoder(&params(Some(6300))).unwrap();
        enc.send_frame(&audio_frame(&input)).unwrap();
        enc.flush().unwrap();

        let mut dec = SynthesisState::new();
        let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SIZE_SAMPLES);
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), MPMLQ_PAYLOAD_BYTES);
            assert_eq!(pkt.data[0] & 0b11, 0b00, "discriminator must be 00");
            decoded.extend_from_slice(&dec.decode_mpmlq(&pkt.data).unwrap());
        }
        assert_eq!(decoded.len(), input.len());

        // PSNR against PEAK = 32 767 (i16 full-scale).
        let n = input.len();
        let mut mse = 0.0f64;
        for i in 0..n {
            let e = decoded[i] as f64 - input[i] as f64;
            mse += e * e;
        }
        mse /= n as f64;
        let peak = 32_767.0f64;
        let psnr = 10.0 * (peak * peak / mse).log10();

        // Documented floor for the simplified codebooks. Observed PSNR is
        // ~6.5 dB on this signal; require at least 0 dB so the test fails
        // loudly only if the pipeline stops producing any signal at all.
        // Compute signal-energy SNR too.
        let mut sig_e = 0.0f64;
        for &s in &input {
            sig_e += (s as f64).powi(2);
        }
        sig_e /= n as f64;
        let snr = 10.0 * (sig_e / mse.max(1e-10)).log10();
        eprintln!("mpmlq_roundtrip_voiced: SNR = {snr:.2} dB");
        assert!(
            psnr >= 15.0,
            "MP-MLQ voiced-signal PSNR = {psnr:.2} dB, expected >= 15 dB"
        );
        assert!(psnr.is_finite(), "PSNR must be finite (MSE was {mse})");
        // Emit the measured value so `cargo test -- --nocapture` surfaces
        // the ~6 dB we see today and flags regressions if it drops.
        eprintln!("mpmlq_roundtrip_voiced_psnr_floor: PSNR = {psnr:.2} dB");
    }

    #[test]
    fn mpmlq_pulse_pack_round_trip() {
        // Verify pack/unpack of MP-MLQ pulses is an identity for both
        // 5-pulse and 6-pulse layouts.
        for n in [MPMLQ_PULSES_EVEN, MPMLQ_PULSES_ODD] {
            let mut p = MpMlqPulses {
                n_pulses: n as u8,
                ..MpMlqPulses::default()
            };
            for t in 0..n {
                p.positions[t] = (t as u32 * 3 + 1) & 0x7;
                p.signs[t] = if t % 2 == 0 { 1 } else { -1 };
            }
            let packed = pack_mpmlq_pulses(&p);
            let unpacked = unpack_mpmlq_pulses(packed, n);
            for t in 0..n {
                assert_eq!(unpacked.positions[t], p.positions[t]);
                assert_eq!(unpacked.signs[t], p.signs[t]);
            }
        }
    }

    #[test]
    fn multiple_frames_produce_rising_pts() {
        let mut enc = make_encoder(&params(Some(5300))).unwrap();
        let pcm = sine_mixture(4);
        enc.send_frame(&audio_frame(&pcm)).unwrap();
        enc.flush().unwrap();
        let mut last_pts = -1i64;
        while let Ok(pkt) = enc.receive_packet() {
            let pts = pkt.pts.expect("pts");
            assert!(pts > last_pts);
            last_pts = pts;
        }
    }

    #[test]
    fn lsp_quantisation_round_trips_to_valid_vector() {
        // Verify LSPs stay strictly ordered after encode / decode.
        let lsp = [
            0.95f32, 0.80, 0.55, 0.30, 0.05, -0.15, -0.40, -0.60, -0.80, -0.95,
        ];
        let (idx, q) = quantise_lsp(&lsp);
        assert!(idx.iter().all(|&i| i < 256));
        for k in 1..LPC_ORDER {
            assert!(q[k] < q[k - 1], "LSPs must be strictly decreasing");
        }
    }

    #[test]
    fn gain_quantisation_round_trip_preserves_sign() {
        let idx = quantise_gain(0.5, -2.5);
        let (g_a, g_f) = dequantise_gain(idx);
        assert!((g_a - 0.48).abs() < 0.2); // 3-bit quantiser has ~0.16 step
        assert!(g_f < 0.0);
    }
}
