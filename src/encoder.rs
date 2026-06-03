//! ITU-T G.723.1 encoder — ACELP (5.3 kbit/s) and MP-MLQ (6.3 kbit/s) paths.
//!
//! # Scope
//!
//! This module implements **both** rates of G.723.1:
//!
//! - **5.3 kbit/s ACELP** — 4 pulses per subframe on stride-8 tracks
//!   (T0..T3 each with 8 positions, a 1-bit grid shifting all pulses by
//!   +4 so the union of grids covers every sample); 20-byte payload,
//!   discriminator `01`.
//! - **6.3 kbit/s MP-MLQ** — 6 pulses on odd subframes (0, 2) and
//!   5 pulses on even subframes (1, 3); 24-byte payload, discriminator `00`.
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
//!  PCM s16 → LPC analysis (autocorrelation + Levinson + lag window)
//!          → LSP conversion (Chebyshev root-finding) + factorial
//!            scalar split VQ (24 bits across three 8-bit splits)
//!          → 4× subframe loop (ACB lookup against SynthesisState):
//!                - zero-input response of 1/A_q(z) to form ZIR-free target
//!                - open-loop lag search on the ZIR-free target
//!                - ACB gain quantised (4 bits, 0.0..1.25)
//!                - rate-specific FCB search against the ACB residual
//!                    · ACELP:  4-pulse stride-8 tracks + grid, coord
//!                              descent refinement
//!                    · MP-MLQ: 6/5-pulse greedy search on stride-N tracks
//!                - joint gain-pair refinement around the initial
//!                  quantisation (27-pair neighbourhood scan)
//!          → canonical SynthesisState::synthesise() commits decoder
//!            state so encoder + decoder stay in lockstep
//!          → bit-pack 158 bits (ACELP, 20 B, rate=01)
//!               or 192 bits (MP-MLQ, 24 B, rate=00)
//! ```
//!
//! # Not bit-compatible with ITU-T reference tables
//!
//! The LSP split VQ (a factorial scalar product code), joint gain
//! codebook (4-bit ACB + 7-bit FCB magnitude on a log2 scale + 1-bit
//! sign), and fixed-codebook pulse track layout here are a clean-room,
//! pure-Rust design. They are internally consistent and give a solid
//! round-trip PSNR (see the README and integration tests) but are not
//! bit-compatible with ITU-T Tables 5 / 7 / 9, so a bitstream produced
//! here does not decode to high-quality speech on an external,
//! spec-table G.723.1 reference decoder.

use std::collections::VecDeque;

#[cfg(test)]
use oxideav_core::AudioFrame;
use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::BitReader;
use crate::tables::{
    FRAME_SIZE_SAMPLES, HIGH_RATE_BYTES, LOW_RATE_BYTES, LPC_ORDER,
    LSP_STABILITY_DELTA_MIN_ERASURE_HZ, LSP_STABILITY_DELTA_MIN_HZ, LSP_STABILITY_MAX_ITERATIONS,
    PITCH_MAX, PITCH_MIN, POSTFILTER_GAMMA1, POSTFILTER_GAMMA2, POSTFILTER_LTP_GAMMA_HIGH,
    POSTFILTER_LTP_GAMMA_LOW, POSTFILTER_LTP_PRED_GAIN_DB_MIN, POSTFILTER_LTP_SEARCH_RADIUS,
    POSTFILTER_TILT, SAMPLE_RATE_HZ, SUBFRAMES_PER_FRAME, SUBFRAME_SIZE,
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

/// Operating rate of a decoded frame. Threaded into the post-filter
/// chain so the pitch (long-term) post-filter can pick the
/// rate-specific LTP weighting γ_ltp (G.723.1 §3.6: 0.1875 for the high
/// rate, 0.25 for the low rate).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Rate {
    /// 5.3 kbit/s ACELP. γ_ltp = 0.25.
    Low,
    /// 6.3 kbit/s MP-MLQ. γ_ltp = 0.1875.
    High,
}

impl Rate {
    /// Long-term post-filter weighting γ_ltp for this rate (G.723.1
    /// §3.6 eq. 42).
    fn ltp_gamma(self) -> f32 {
        match self {
            Rate::High => POSTFILTER_LTP_GAMMA_HIGH,
            Rate::Low => POSTFILTER_LTP_GAMMA_LOW,
        }
    }
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
        // Stream-level shape (mono / 8 kHz / S16) used to be sniffed
        // off the AudioFrame; with the slim those fields live on the
        // upstream stream's `CodecParameters` and are guaranteed by
        // the registry / pipeline that constructed this encoder
        // against `make_params()`. We trust the caller — a
        // mismatched input would have surfaced at the
        // pipeline-build pixel-format / sample-format auto-insert
        // pass and never reach this `send_frame`.
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

/// Spec-shape LSP stability procedure (G.723.1 §3.1 / 2.6, eq. 6–7.3).
///
/// G.723.1 stores decoded LSPs in this crate as **cosines** `p̃_j` of
/// normalised angular frequencies `ω_j = 2π f_j / fs`. The spec's
/// stability condition is `f_{j+1} − f_j ≥ Δ_min` in *frequency* (Hz);
/// because `cos(ω)` is strictly monotone-decreasing on `[0, π]`, the
/// equivalent test in our representation is
/// `ω_{j+1} − ω_j ≥ Δω_min` with
/// `Δω_min = 2π · Δ_min_hz / SAMPLE_RATE_HZ` rad.
///
/// Procedure per §2.6:
///
/// 1. Convert cosines → angular frequencies via `acos`.
/// 2. Find the first out-of-order pair `(j, j+1)` with `ω_{j+1} − ω_j < Δω_min`.
/// 3. Spread the pair around its midpoint by `±Δω_min/2`:
///    `ω_j ← (ω_j + ω_{j+1})/2 − Δω_min/2`,
///    `ω_{j+1} ← (ω_j + ω_{j+1})/2 + Δω_min/2`.
/// 4. Iterate up to [`LSP_STABILITY_MAX_ITERATIONS`] passes. If the
///    vector still has an out-of-order pair after the cap, the caller is
///    expected to fall back to the previous good LSP (handled by
///    `dequantise_lsp`'s post-call clamp).
///
/// The first and last frequencies are also clamped into `(0, π)` so the
/// outer LSP roots stay strictly inside the unit circle when the LPC
/// coefficients are reconstructed.
///
/// Returns `(stabilised_cosines, converged)`. The `converged` flag is
/// `false` only if the cap was hit with at least one pair still violating
/// the constraint.
pub(crate) fn enforce_lsp_stability(
    lsp_cos: &[f32; LPC_ORDER],
    delta_min_hz: f32,
) -> ([f32; LPC_ORDER], bool) {
    // Convert to angular frequency. The clamp guards against any
    // accumulated numerical drift past ±1 from a previous step.
    let mut omega = [0.0f32; LPC_ORDER];
    for i in 0..LPC_ORDER {
        omega[i] = lsp_cos[i].clamp(-1.0, 1.0).acos();
    }
    // Δ_min in normalised angular frequency: 2π · f / fs.
    let delta_min_rad = std::f32::consts::TAU * delta_min_hz / crate::tables::SAMPLE_RATE_HZ as f32;
    let half = 0.5 * delta_min_rad;
    // Floating-point tolerance for the `≥ Δ_min` check. After spreading,
    // `(mid+half) − (mid−half)` may round to slightly less than
    // `delta_min_rad` (by one f32 ulp ≈ 1e-9 rad at this magnitude); the
    // tolerance keeps the procedure from oscillating on a freshly-fixed
    // pair that satisfies the spec within rounding error.
    let tol = delta_min_rad * 1.0e-5;
    let mut converged = false;
    for _iter in 0..LSP_STABILITY_MAX_ITERATIONS {
        let mut violated = false;
        for j in 0..LPC_ORDER - 1 {
            if omega[j + 1] - omega[j] < delta_min_rad - tol {
                let mid = 0.5 * (omega[j] + omega[j + 1]);
                omega[j] = mid - half;
                omega[j + 1] = mid + half;
                violated = true;
            }
        }
        if !violated {
            converged = true;
            break;
        }
    }
    // Outer-root clamp: keep ω_0 > 0 and ω_{p-1} < π so the LSP-derived
    // LPC roots stay strictly inside the unit circle (cosine domain |p̃| < 1).
    let margin = half.max(1.0e-3);
    if omega[0] < margin {
        omega[0] = margin;
    }
    if omega[LPC_ORDER - 1] > std::f32::consts::PI - margin {
        omega[LPC_ORDER - 1] = std::f32::consts::PI - margin;
    }
    let mut out = [0.0f32; LPC_ORDER];
    for i in 0..LPC_ORDER {
        out[i] = omega[i].cos();
    }
    (out, converged)
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
    // Apply the spec's §3.1 / 2.6 stability procedure with Δ_min = 31.25 Hz
    // (eq. 6–7.3). Spreading out-of-order pairs around their midpoint
    // converges quickly on typical decoded LSPs because the factorial
    // scalar VQ already produces near-monotone output; the iterative cap
    // (10) is the spec-mandated safety net.
    let (stabilised, _converged) = enforce_lsp_stability(&out, LSP_STABILITY_DELTA_MIN_HZ);
    stabilised
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
///
/// The post-filter fields (`pf_*`) are updated only on the decoder entry
/// points and left untouched by the bare [`SynthesisState::synthesise`]
/// kernel, so the encoder's shadow-decoder pass stays on the pre-post-filter
/// signal path (what the encoder's analysis-by-synthesis loop needs to see).
pub struct SynthesisState {
    prev_lsp: [f32; LPC_ORDER],
    exc_history: [f32; PITCH_MAX + SUBFRAME_SIZE],
    syn_mem: [f32; LPC_ORDER],
    // Post-filter state ---------------------------------------------------
    /// Pre-post-filter synthesis history. Feeds the pitch post-filter's
    /// long-term predictor when the current subframe's lag reaches back
    /// beyond the subframe boundary.
    pf_syn_hist: [f32; PITCH_MAX + SUBFRAME_SIZE],
    /// Numerator memory for A(z/γ₁) of the formant post-filter.
    pf_num_mem: [f32; LPC_ORDER],
    /// Denominator memory for 1/A(z/γ₂).
    pf_den_mem: [f32; LPC_ORDER],
    /// First-order tilt compensation one-sample memory.
    pf_tilt_prev: f32,
    /// Smoothed AGC gain, preserving synthesis-signal energy across the
    /// post-filter chain.
    pf_agc_gain: f32,
    // Frame-erasure / SID concealment state -------------------------------
    /// Last decoded pitch lag — extrapolated during erasures.
    pf_last_lag: i32,
    /// Last decoded (g_adapt, g_fixed) — attenuated during erasures.
    pf_last_gain_adapt: f32,
    pf_last_gain_fixed: f32,
    /// Number of consecutive erased frames (0 = good frame most recent).
    pf_erased_run: u32,
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
            pf_syn_hist: [0.0; PITCH_MAX + SUBFRAME_SIZE],
            pf_num_mem: [0.0; LPC_ORDER],
            pf_den_mem: [0.0; LPC_ORDER],
            pf_tilt_prev: 0.0,
            pf_agc_gain: 1.0,
            pf_last_lag: 60,
            pf_last_gain_adapt: 0.0,
            pf_last_gain_fixed: 0.0,
            pf_erased_run: 0,
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

    /// Record the last frame's lag and gains for frame-erasure
    /// extrapolation. Called by the decoder entry points *after* they have
    /// decoded one good frame, so the next erased frame can repeat them.
    fn record_last_frame(
        &mut self,
        lags: &[i32; SUBFRAMES_PER_FRAME],
        gain_idx: &[u32; SUBFRAMES_PER_FRAME],
    ) {
        self.pf_last_lag = lags[SUBFRAMES_PER_FRAME - 1];
        let (g_a, g_f) = dequantise_gain(gain_idx[SUBFRAMES_PER_FRAME - 1]);
        self.pf_last_gain_adapt = g_a;
        self.pf_last_gain_fixed = g_f;
        self.pf_erased_run = 0;
    }

    /// Long-term (pitch) post-filter per G.723.1 §3.6 (eq. 42–47):
    /// search forward `M_f` ∈ `[lag − 3, lag + 3]` and backward
    /// `M_b` ∈ `[lag − 3, lag + 3]` for the maximum positive
    /// cross-correlation against the synthesis-domain signal, pick
    /// `(w_f, w_b) ∈ {(0, 0), (0, 1), (1, 0)}` based on which side has
    /// the larger prediction gain, and apply the energy-normalised LTP
    /// post-filter with rate-specific weighting `γ_ltp`. Subframes whose
    /// best pitch prediction gain is below 1.25 dB pass through
    /// unchanged (eq. 45–46 gate).
    fn ltp_post_filter_subframe(
        &self,
        syn: &[f32; SUBFRAME_SIZE],
        lag: i32,
        rate: Rate,
    ) -> [f32; SUBFRAME_SIZE] {
        // Centre the search at the decoded lag, clamped to the legal
        // PITCH_MIN..=PITCH_MAX range so we never read past the end of
        // the synthesis-domain history buffer.
        let lag_c = lag.clamp(PITCH_MIN as i32, PITCH_MAX as i32);
        let m_lo = (lag_c - POSTFILTER_LTP_SEARCH_RADIUS).max(PITCH_MIN as i32);
        let m_hi = (lag_c + POSTFILTER_LTP_SEARCH_RADIUS).min(PITCH_MAX as i32);

        // Subframe energy `T_en` (eq. 45 denominator term).
        let mut t_en = 0.0f32;
        for &v in syn.iter() {
            t_en += v * v;
        }
        // If the subframe is essentially silent, the postfilter has
        // nothing to enhance and the prediction gain is meaningless.
        if t_en < 1e-12 {
            return *syn;
        }

        let mut out = *syn;

        // Forward lookup `x[n + M_f]`: for the synthesis-domain signal
        // we read inside the current subframe — when `n + M_f >= 60`
        // we must clamp to keep the lookup in-bounds; per the spec the
        // forward window naturally stays inside the frame (Σ_n is over
        // the subframe, with M_f >= PITCH_MIN >= 18 the read at the
        // tail can extend past 60 — we fold by skipping out-of-range
        // contributions, which conservatively shrinks the correlation
        // window rather than wrapping the signal).
        let (mf_best, cf, df) = self.ltp_search_forward(syn, m_lo, m_hi);

        // Backward lookup `x[n − M_b]`: when `n − M_b < 0` the read
        // falls into the saved `pf_syn_hist` (the most recent
        // PITCH_MAX + SUBFRAME_SIZE pre-postfilter samples).
        let (mb_best, cb, db) = self.ltp_search_backward(syn, m_lo, m_hi);

        // Per-side prediction gains in linear domain:
        //   G_side = C² / (D · T_en)   (eq. 45 / eq. 46 ratio)
        // converted to dB as `−10·log10(1 − G_side)`. We keep things
        // in the linear ratio domain for the gate (and just compare
        // gains numerically); a side with C ≤ 0 contributes 0.
        let gain_f = if cf > 0.0 && df > 0.0 {
            cf * cf / (df * t_en)
        } else {
            0.0
        };
        let gain_b = if cb > 0.0 && db > 0.0 {
            cb * cb / (db * t_en)
        } else {
            0.0
        };

        // Best per-subframe prediction gain. The gate is in dB:
        //   −10·log10(1 − G) ≥ POSTFILTER_LTP_PRED_GAIN_DB_MIN
        //   ⇔ G ≥ 1 − 10^(−POSTFILTER_LTP_PRED_GAIN_DB_MIN / 10)
        let gain_best = gain_f.max(gain_b);
        let gate_linear = 1.0 - 10.0_f32.powf(-POSTFILTER_LTP_PRED_GAIN_DB_MIN / 10.0);
        if gain_best < gate_linear {
            return out; // postfilter bypassed for this subframe.
        }

        // Pick the (w_f, w_b) winner: whichever side has the larger
        // prediction gain, weight = 1 on that side and 0 on the other
        // (the spec's `{(0,0), (0,1), (1,0)}` constraint).
        let pick_forward = gain_f >= gain_b;
        let (c_chosen, d_chosen);
        let mut contrib = [0.0f32; SUBFRAME_SIZE];
        if pick_forward {
            c_chosen = cf;
            d_chosen = df;
            let m = mf_best as usize;
            for n in 0..SUBFRAME_SIZE {
                let j = n + m;
                if j < SUBFRAME_SIZE {
                    contrib[n] = syn[j];
                }
                // j outside the subframe → 0 (window-shrinkage; matches
                // the truncated correlation that produced cf/df).
            }
        } else {
            c_chosen = cb;
            d_chosen = db;
            let m = mb_best as usize;
            let hist = &self.pf_syn_hist;
            let hlen = hist.len();
            for n in 0..SUBFRAME_SIZE {
                let val = if n >= m {
                    syn[n - m]
                } else {
                    let k = m - n;
                    if k <= hlen {
                        hist[hlen - k]
                    } else {
                        0.0
                    }
                };
                contrib[n] = val;
            }
        }

        // Per-side LTP gain `g_side = C / D`, clamped to [0, 1] per spec.
        let g_side = (c_chosen / d_chosen.max(1e-12)).clamp(0.0, 1.0);
        let gamma_ltp = rate.ltp_gamma();

        // Output energy normalisation `g_p` (eq. 47): preserves the
        // subframe energy through the LTP comb-filter. With a single
        // active side the denominator simplifies to
        //   T_en + 2·γ_ltp·g_side·C + γ_ltp²·g_side²·D.
        let num_energy = t_en;
        let mut den_energy = t_en
            + 2.0 * gamma_ltp * g_side * c_chosen
            + gamma_ltp * gamma_ltp * g_side * g_side * d_chosen;
        if den_energy < 1e-12 {
            den_energy = 1e-12;
        }
        let g_p = (num_energy / den_energy).sqrt().min(1.0);

        for n in 0..SUBFRAME_SIZE {
            out[n] = g_p * (syn[n] + gamma_ltp * g_side * contrib[n]);
        }
        out
    }

    /// Maximise the forward cross-correlation `C_f = Σ_n syn[n]·syn[n + M_f]`
    /// over `M_f ∈ [m_lo, m_hi]`, returning `(M_f*, C_f*, D_f*)` where
    /// `D_f* = Σ_n syn[n + M_f*]²` is the matched energy for the
    /// winning lag. Out-of-range reads (`n + M_f ≥ SUBFRAME_SIZE`)
    /// contribute zero — they would otherwise need the next subframe's
    /// synthesis (not available at this point in the pipeline).
    fn ltp_search_forward(
        &self,
        syn: &[f32; SUBFRAME_SIZE],
        m_lo: i32,
        m_hi: i32,
    ) -> (i32, f32, f32) {
        let mut best_metric = -f32::INFINITY;
        let mut best = (m_lo, 0.0f32, 0.0f32);
        for m in m_lo..=m_hi {
            let mu = m as usize;
            if mu >= SUBFRAME_SIZE {
                continue;
            }
            let mut c = 0.0f32;
            let mut d = 0.0f32;
            for n in 0..(SUBFRAME_SIZE - mu) {
                let p = syn[n + mu];
                c += syn[n] * p;
                d += p * p;
            }
            let metric = if c > 0.0 && d > 1e-12 { c * c / d } else { 0.0 };
            if metric > best_metric {
                best_metric = metric;
                best = (m, c, d);
            }
        }
        best
    }

    /// Maximise the backward cross-correlation `C_b = Σ_n syn[n]·syn[n − M_b]`
    /// over `M_b ∈ [m_lo, m_hi]`, returning `(M_b*, C_b*, D_b*)`. The
    /// backward reach extends into `pf_syn_hist`, which carries the
    /// most recent PITCH_MAX + SUBFRAME_SIZE pre-postfilter synthesis
    /// samples.
    fn ltp_search_backward(
        &self,
        syn: &[f32; SUBFRAME_SIZE],
        m_lo: i32,
        m_hi: i32,
    ) -> (i32, f32, f32) {
        let hist = &self.pf_syn_hist;
        let hlen = hist.len();
        let mut best_metric = -f32::INFINITY;
        let mut best = (m_lo, 0.0f32, 0.0f32);
        for m in m_lo..=m_hi {
            let mu = m as usize;
            let mut c = 0.0f32;
            let mut d = 0.0f32;
            for n in 0..SUBFRAME_SIZE {
                let past = if n >= mu {
                    syn[n - mu]
                } else {
                    let k = mu - n;
                    if k <= hlen {
                        hist[hlen - k]
                    } else {
                        0.0
                    }
                };
                c += syn[n] * past;
                d += past * past;
            }
            let metric = if c > 0.0 && d > 1e-12 { c * c / d } else { 0.0 };
            if metric > best_metric {
                best_metric = metric;
                best = (m, c, d);
            }
        }
        best
    }

    /// Apply the G.723.1 post-filter chain to one subframe:
    /// pitch-enhancement (§3.6) → formant A(z/γ₁)/A(z/γ₂) (§3.8) →
    /// first-order tilt compensation → smoothed automatic-gain-control,
    /// updating the post-filter memories in place. `syn` is the 60-sample
    /// synthesis output of the current subframe; `lag` is the decoded
    /// pitch lag used as the centre of the LTP search; `rate` picks the
    /// rate-specific γ_ltp weighting in §3.6; `a_sub` are the
    /// unquantised-polynomial LPC coefficients for the subframe.
    ///
    /// The post-filter runs only on the decoder path; it is deliberately
    /// separate from [`SynthesisState::synthesise`] so the encoder's shadow
    /// synth (which needs the raw excitation-domain signal for closed-loop
    /// analysis) never sees a post-filtered intermediate.
    fn post_filter_subframe(
        &mut self,
        a_sub: &[f32; LPC_ORDER + 1],
        syn: &[f32; SUBFRAME_SIZE],
        lag: i32,
        rate: Rate,
        out: &mut [f32; SUBFRAME_SIZE],
    ) {
        // ---- 1. Long-term (pitch) post-filter per G.723.1 §3.6, eq. 42:
        //
        //   ppf[n] = g_p · { x[n] + γ_ltp · ( w_f · g_f · x[n + M_f]
        //                                   + w_b · g_b · x[n − M_b] ) }
        //
        // with `(w_f, w_b)` constrained to one of `{(0,0), (0,1), (1,0)}`,
        // `M_f` / `M_b` searched in `[L − 3, L + 3]` for the max forward /
        // backward cross-correlation (eq. 43.1–43.2), and the postfilter
        // bypassed for this subframe if the pitch-prediction gain falls
        // below 1.25 dB (eq. 45–46 gate).  γ_ltp differs by rate per §3.6.
        //
        // We run this on the synthesis-domain signal (the same signal the
        // surrounding ARMA formant postfilter sees), using the dedicated
        // pre-postfilter history `pf_syn_hist` so the back reference is
        // spectrally consistent.  The spec equation talks about the
        // excitation `e[n]`; on a quasi-stationary subframe the LTP
        // structure carries over to the synthesis signal because the LPC
        // synthesis filter does not change the periodicity.  The 1.25 dB
        // prediction-gain gate then naturally suppresses subframes where
        // the synthesis-domain LTP shape diverges from the excitation's.
        let after_pitch = self.ltp_post_filter_subframe(syn, lag, rate);

        // ---- 2. Formant post-filter A(z/γ₁) / A(z/γ₂). γ₁ < γ₂ widens
        // the formant bandwidth on the numerator and narrows it on the
        // denominator, emphasising the spectral peaks that carry speech
        // formants without shifting their centre frequency.
        let a_num = bandwidth_expand(a_sub, POSTFILTER_GAMMA1);
        let a_den = bandwidth_expand(a_sub, POSTFILTER_GAMMA2);
        let mut after_formant = [0.0f32; SUBFRAME_SIZE];
        for n in 0..SUBFRAME_SIZE {
            let x = after_pitch[n];
            // y[n] = x[n] + Σ a_num[k] · x_hist[k] - Σ a_den[k] · y_hist[k]
            let mut y = x;
            for k in 0..LPC_ORDER {
                y += a_num[k + 1] * self.pf_num_mem[k];
            }
            for k in 0..LPC_ORDER {
                y -= a_den[k + 1] * self.pf_den_mem[k];
            }
            for k in (1..LPC_ORDER).rev() {
                self.pf_num_mem[k] = self.pf_num_mem[k - 1];
                self.pf_den_mem[k] = self.pf_den_mem[k - 1];
            }
            self.pf_num_mem[0] = x;
            self.pf_den_mem[0] = y;
            after_formant[n] = y;
        }

        // ---- 3. First-order tilt compensation: y[n] = x[n] − μ · x[n-1].
        // Flattens the slight low-frequency boost introduced by the formant
        // stage so the post-filter output has the same spectral tilt as the
        // synthesis input.
        let mut after_tilt = [0.0f32; SUBFRAME_SIZE];
        let mut prev = self.pf_tilt_prev;
        for n in 0..SUBFRAME_SIZE {
            let x = after_formant[n];
            after_tilt[n] = x - POSTFILTER_TILT * prev;
            prev = x;
        }
        self.pf_tilt_prev = prev;

        // ---- 4. Smoothed AGC: target `post-filter output energy` =
        // `synthesis input energy` so the chain doesn't pump the level.
        // The per-subframe gain is smoothed exponentially across samples
        // (α = 0.85) to avoid audible gain steps at subframe boundaries.
        let mut e_in = 1e-6f32;
        let mut e_out = 1e-6f32;
        for n in 0..SUBFRAME_SIZE {
            e_in += syn[n] * syn[n];
            e_out += after_tilt[n] * after_tilt[n];
        }
        let target_gain = (e_in / e_out).sqrt().clamp(0.0, 4.0);
        let alpha = 0.85f32;
        for n in 0..SUBFRAME_SIZE {
            self.pf_agc_gain = alpha * self.pf_agc_gain + (1.0 - alpha) * target_gain;
            out[n] = after_tilt[n] * self.pf_agc_gain;
        }

        // Advance the post-filter synthesis-domain history with this
        // subframe's *pre*-post-filter samples.
        self.pf_syn_hist.rotate_left(SUBFRAME_SIZE);
        let tail = self.pf_syn_hist.len() - SUBFRAME_SIZE;
        self.pf_syn_hist[tail..].copy_from_slice(syn);
    }

    /// Run the post-filter across a full frame. `pcm` is the synthesis-
    /// filter output in `[-1, 1]`-normalised f32. `lsp_q`/`lags` match the
    /// decoded frame fields so per-subframe formant filters have the right
    /// LPC coefficients. `rate` selects the rate-specific LTP weighting in
    /// the pitch postfilter (§3.6).
    ///
    /// G.723.1 §3.6 specifies that the pitch postfilter uses `L_0` (the
    /// absolute lag of subframe 0) for subframes 0,1 and `L_2` (subframe
    /// 2's absolute lag) for subframes 2,3 — not the per-subframe
    /// delta-decoded lags. We respect that here.
    fn apply_post_filter(
        &mut self,
        lsp_q: &[f32; LPC_ORDER],
        lags: &[i32; SUBFRAMES_PER_FRAME],
        rate: Rate,
        pcm: &mut [f32; FRAME_SIZE_SAMPLES],
    ) {
        // The synthesise() pass already advanced self.prev_lsp to lsp_q;
        // for per-subframe LPC we must re-interpolate against what *was*
        // the previous LSP, so we reconstruct it from self.prev_lsp (which
        // is the current frame's quantised LSP at this point — same for
        // all subframes).  We pass `lsp_q` as both prev and cur so the
        // interpolation degenerates to the per-subframe current value;
        // this is a deliberate simplification — the post-filter only needs
        // LPC coefficients that are close enough to the synthesis filter
        // in the same subframe, not the precise interpolated curve.
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, lsp_q, lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);
            // Reference lag for the LTP postfilter: L_0 covers subframes
            // 0,1; L_2 covers subframes 2,3 (G.723.1 §3.6 prose).
            let ref_lag = if s < 2 { lags[0] } else { lags[2] };
            let start = s * SUBFRAME_SIZE;
            let end = start + SUBFRAME_SIZE;
            let mut syn = [0.0f32; SUBFRAME_SIZE];
            syn.copy_from_slice(&pcm[start..end]);
            let mut post = [0.0f32; SUBFRAME_SIZE];
            self.post_filter_subframe(&a_sub, &syn, ref_lag, rate, &mut post);
            pcm[start..end].copy_from_slice(&post);
        }
    }

    /// Concealment path for SID / erased packets. Extrapolates the last
    /// lag, attenuates the gains, seeds a pseudo-random innovation, runs
    /// the usual synthesis + post-filter pipeline, and leaves attenuated
    /// gains in `pf_last_gain_*` so repeated erasures decay smoothly.
    ///
    /// Returns 240 concealed S16 samples.
    pub fn decode_erased(&mut self) -> [i16; FRAME_SIZE_SAMPLES] {
        self.pf_erased_run = self.pf_erased_run.saturating_add(1);
        // Attenuation schedule: halve per frame for the first few erasures,
        // then mute. Anything louder would make repeated packet loss
        // audible as "buzzing".
        let atten = match self.pf_erased_run {
            1 => 0.7f32,
            2 => 0.5,
            3 => 0.35,
            4 => 0.2,
            _ => 0.0,
        };
        let g_adapt = self.pf_last_gain_adapt * atten;
        let g_fixed = self.pf_last_gain_fixed * atten;
        // Erasure-variant LSP stability check (G.723.1 §3.10.1): re-apply
        // the §2.6 ordering procedure to the extrapolated LSP with the
        // wider Δ_min = 62.5 Hz so the relaxed constraint allows pairs that
        // drifted slightly out of order during repeated erasures to be
        // pulled back without destroying the previous-frame envelope.
        let (lsp_q, _converged) =
            enforce_lsp_stability(&self.prev_lsp, LSP_STABILITY_DELTA_MIN_ERASURE_HZ);
        let lag = self.pf_last_lag.clamp(PITCH_MIN as i32, PITCH_MAX as i32);

        // Pseudo-random innovation (deterministic LCG) so concealment is
        // reproducible and doesn't introduce a tonal artefact.
        let mut lcg = 0xDEADBEEFu32.wrapping_add(self.pf_erased_run.wrapping_mul(0x9E37_79B9));
        let mut next_rand = || -> f32 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0
        };

        let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
        // Run four subframes manually so we bypass the usual gain-dequant
        // step (we already know g_adapt / g_fixed in fp form).
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, &self.prev_lsp, &lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);

            let mut adaptive = [0.0f32; SUBFRAME_SIZE];
            copy_adaptive(&self.exc_history, lag, &mut adaptive);
            let mut pulses = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                pulses[n] = next_rand();
            }
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                exc[n] = g_adapt * adaptive[n] + g_fixed * pulses[n];
            }

            // 1/A(z) synthesis.
            let mut syn = [0.0f32; SUBFRAME_SIZE];
            for i in 0..SUBFRAME_SIZE {
                let mut y = exc[i];
                for k in 0..LPC_ORDER {
                    y -= a_sub[k + 1] * self.syn_mem[k];
                }
                for k in (1..LPC_ORDER).rev() {
                    self.syn_mem[k] = self.syn_mem[k - 1];
                }
                self.syn_mem[0] = y;
                syn[i] = y;
            }
            // Post-filter. The erasure path has no rate signal, so we
            // default to the high-rate γ_ltp (the more conservative
            // value: 0.1875 vs 0.25) so concealment stays gentler than
            // either rate's normal pitch postfilter would be.
            let mut post = [0.0f32; SUBFRAME_SIZE];
            self.post_filter_subframe(&a_sub, &syn, lag, Rate::High, &mut post);
            let start = s * SUBFRAME_SIZE;
            pcm[start..start + SUBFRAME_SIZE].copy_from_slice(&post);

            // Advance excitation history with the concealed excitation.
            self.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.exc_history.len() - SUBFRAME_SIZE;
            self.exc_history[tail..].copy_from_slice(&exc);
        }

        // Decay gains in place for the next erasure in a run.
        self.pf_last_gain_adapt = g_adapt;
        self.pf_last_gain_fixed = g_fixed;

        to_i16_frame(&pcm)
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
        // Post-filter chain: pitch + formant + tilt + AGC. Updates the
        // `pf_*` post-filter memories; leaves `synthesise`'s state (exc
        // history / syn_mem / prev_lsp) alone so the encoder's shadow
        // synth sees the unmodified signal.  ACELP uses γ_ltp = 0.25 in
        // the pitch postfilter (G.723.1 §3.6).
        self.apply_post_filter(&lsp_q, &lags, Rate::Low, &mut pcm_f);
        self.record_last_frame(&lags, &gain);
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
        // Same post-filter pipeline as ACELP (pitch + formant + tilt +
        // AGC).  MP-MLQ uses γ_ltp = 0.1875 in the pitch postfilter
        // (G.723.1 §3.6).
        self.apply_post_filter(&lsp_q, &lags, Rate::High, &mut pcm_f);
        self.record_last_frame(&lags, &gain);
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
    use oxideav_core::{CodecId, CodecParameters, Frame, SampleFormat};

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
            samples: samples.len() as u32,
            pts: Some(0),
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

        let mut reg = oxideav_core::CodecRegistry::new();
        crate::register_codecs(&mut reg);
        let mut dec = reg
            .first_decoder(&params(None))
            .expect("decoder factory must exist");

        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).unwrap();
            let f = dec.receive_frame().unwrap();
            // Scaffold decoder emits silence; just assert it produces a
            // well-shaped audio frame of the right size.
            match f {
                Frame::Audio(af) => {
                    // Stream-level shape (sample_rate / channels) used
                    // to live on each frame — moved to the stream's
                    // CodecParameters with the slim. The per-frame
                    // assertion is now just the sample count.
                    assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
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

        let mut reg = oxideav_core::CodecRegistry::new();
        crate::register_codecs(&mut reg);
        let mut dec = reg
            .first_decoder(&params(None))
            .expect("decoder factory must exist");

        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).unwrap();
            let f = dec.receive_frame().unwrap();
            match f {
                Frame::Audio(af) => {
                    // Stream-level shape (sample_rate / channels) used
                    // to live on each frame — moved to the stream's
                    // CodecParameters with the slim. The per-frame
                    // assertion is now just the sample count.
                    assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
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

    /// Post-filter AGC must preserve energy: decoded PCM energy for a
    /// voiced input should be within a small factor of the pre-post-filter
    /// synthesis energy. We measure this indirectly by asserting the
    /// post-filter doesn't cause the decoded SNR floor to regress (covered
    /// by `roundtrip_two_seconds_voiced_psnr_both_rates` in the
    /// integration tests) and here check that the AGC state starts at
    /// unity gain on a fresh state.
    #[test]
    fn post_filter_state_starts_at_unity_agc() {
        let st = SynthesisState::new();
        assert_eq!(st.pf_agc_gain, 1.0);
        assert_eq!(st.pf_erased_run, 0);
    }

    /// Erased-frame concealment: a SID / Untransmitted frame must produce
    /// a full 240-sample frame that decays with run length. The first
    /// erasure keeps the gain close to the last good frame's; by the 5th
    /// erasure the gain is fully muted.
    #[test]
    fn decode_erased_produces_decaying_output() {
        let mut st = SynthesisState::new();
        st.pf_last_gain_adapt = 0.5;
        st.pf_last_gain_fixed = 0.2;
        // Seed the excitation history so there's something to propagate.
        for i in 0..st.exc_history.len() {
            st.exc_history[i] = ((i as f32 * 0.17).sin()) * 0.1;
        }
        let e1 = st.decode_erased();
        assert_eq!(e1.len(), FRAME_SIZE_SAMPLES);
        assert_eq!(st.pf_erased_run, 1);

        // Run a handful more to confirm repeated erasure decays and never
        // panics. (Samples are i16, trivially finite.)
        for _ in 0..10 {
            let ek = st.decode_erased();
            let _ = ek;
        }
        assert_eq!(st.pf_last_gain_adapt, 0.0);
        assert_eq!(st.pf_last_gain_fixed, 0.0);
    }

    /// Rate ↔ γ_ltp mapping must match the published §3.6 constants.
    #[test]
    fn rate_ltp_gamma_matches_spec() {
        assert!((Rate::High.ltp_gamma() - 0.1875).abs() < 1e-6);
        assert!((Rate::Low.ltp_gamma() - 0.25).abs() < 1e-6);
    }

    /// Silent subframe: the pitch-postfilter helper must short-circuit
    /// to the input (g_p would otherwise divide by ~0 energy).
    #[test]
    fn ltp_postfilter_passes_silence_through_unchanged() {
        let st = SynthesisState::new();
        let syn = [0.0f32; SUBFRAME_SIZE];
        let out = st.ltp_post_filter_subframe(&syn, 40, Rate::High);
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Pure-white (broadband uncorrelated) input has no LTP structure,
    /// so the spec's 1.25 dB prediction-gain gate must bypass the LTP
    /// postfilter and pass the signal through unchanged.  We build a
    /// reproducible "white" sequence with a small LCG and confirm the
    /// output equals the input bit-for-bit.
    #[test]
    fn ltp_postfilter_gates_off_on_white_signal() {
        let mut st = SynthesisState::new();
        let mut lcg: u32 = 0x1234_5678;
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for s in syn.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = ((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0;
        }
        // Empty history so the backward search starts from "silence".
        for h in st.pf_syn_hist.iter_mut() {
            *h = 0.0;
        }
        let out = st.ltp_post_filter_subframe(&syn, 40, Rate::High);
        // Predominantly bypass — within a few percent or below the gate.
        let mut max_delta = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            let d = (out[n] - syn[n]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        // White input shouldn't trigger the LTP postfilter at all → out == in.
        assert!(
            max_delta < 1e-6,
            "white signal should bypass LTP postfilter, max_delta = {max_delta}"
        );
    }

    /// Strongly periodic input: a periodic input with a slow amplitude
    /// modulation triggers the LTP postfilter (predictability sails
    /// above the 1.25 dB gate) and the output stays energy-preserving
    /// (g_p ≤ 1 ⇒ peak does not grow).  An idealised noise-free
    /// constant-amplitude sinusoid would have g_p · (1 + γ_ltp · 1) ≡ 1
    /// (mathematical identity), so we add an envelope to break that
    /// degeneracy and see the LTP comb-filter actually act.
    #[test]
    fn ltp_postfilter_engages_on_periodic_signal() {
        let mut st = SynthesisState::new();
        let period: i32 = 40;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        // Two-frame history with the same period but a slowly
        // increasing envelope so the back-reference amplitude is
        // smaller than the current subframe — this breaks the
        // (g_p · (1 + γ) = 1) degeneracy.
        let total_len = st.pf_syn_hist.len() + SUBFRAME_SIZE;
        let env = |i: usize| -> f32 { 0.2 + 0.6 * (i as f32) / (total_len as f32) };
        for (i, h) in st.pf_syn_hist.iter_mut().enumerate() {
            let phase = two_pi * (i as f32) / period as f32;
            *h = phase.sin() * env(i);
        }
        let start_idx = st.pf_syn_hist.len();
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for (n, s) in syn.iter_mut().enumerate() {
            let i = start_idx + n;
            let phase = two_pi * (i as f32) / period as f32;
            *s = phase.sin() * env(i);
        }
        let in_e: f32 = syn.iter().map(|v| v * v).sum();
        let out_high = st.ltp_post_filter_subframe(&syn, period, Rate::High);
        let out_high_e: f32 = out_high.iter().map(|v| v * v).sum();
        // g_p ≤ 1 normalises *total energy*, not per-sample peak — the
        // LTP comb-filter can locally push one sample up while pulling
        // another down. Check the energy constraint that g_p actually
        // enforces (eq. 47), with a small float epsilon.
        assert!(
            out_high_e <= in_e * 1.001,
            "energy-preserving g_p was violated: in {in_e} → out {out_high_e}"
        );
        // The output should not be identical to the input — the LTP
        // postfilter actually engaged.
        let mut max_delta = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            let d = (out_high[n] - syn[n]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        assert!(
            max_delta > 1e-3,
            "LTP postfilter did not engage on periodic input (max_delta = {max_delta})"
        );

        // Low-rate γ_ltp = 0.25 weighs the LTP contribution more
        // heavily than high-rate γ_ltp = 0.1875, so the low-rate
        // output should deviate more from the input than the high-rate
        // output on the same input — confirms the rate threading.
        let mut st2 = SynthesisState::new();
        for (i, h) in st2.pf_syn_hist.iter_mut().enumerate() {
            let phase = two_pi * (i as f32) / period as f32;
            *h = phase.sin() * env(i);
        }
        let out_low = st2.ltp_post_filter_subframe(&syn, period, Rate::Low);
        let mut delta_low = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            delta_low += (out_low[n] - syn[n]).powi(2);
        }
        let mut delta_high = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            delta_high += (out_high[n] - syn[n]).powi(2);
        }
        assert!(
            delta_low > delta_high,
            "low-rate γ_ltp = 0.25 must move the signal more than high-rate γ_ltp = 0.1875 \
             (delta_low = {delta_low}, delta_high = {delta_high})"
        );
    }

    /// Forward-lag search must lock onto the actual peak of a periodic
    /// signal embedded in the subframe.
    #[test]
    fn ltp_forward_search_locks_on_period() {
        let st = SynthesisState::new();
        let period: i32 = 40;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for (n, s) in syn.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / period as f32).sin();
        }
        // Search a window straddling the true period.
        let (best_m, c, d) = st.ltp_search_forward(&syn, 37, 43);
        assert_eq!(best_m, 40, "forward search should pick the exact period 40");
        assert!(c > 0.0);
        assert!(d > 0.0);
    }

    /// Backward-lag search uses pf_syn_hist when n − M_b is negative;
    /// when the history holds a sinusoid at the same period, the
    /// search should still find that period.
    #[test]
    fn ltp_backward_search_uses_history() {
        let mut st = SynthesisState::new();
        let period: i32 = 36;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (i, h) in st.pf_syn_hist.iter_mut().enumerate() {
            *h = (two_pi * (i as f32) / period as f32).sin();
        }
        let start_phase = st.pf_syn_hist.len() as f32;
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for (n, s) in syn.iter_mut().enumerate() {
            *s = (two_pi * (start_phase + n as f32) / period as f32).sin();
        }
        let (best_m, c, _) = st.ltp_search_backward(&syn, 33, 39);
        assert_eq!(
            best_m, 36,
            "backward search should pick the exact period 36"
        );
        assert!(c > 0.0);
    }

    /// `apply_post_filter` must respect §3.6 reference-lag rule: L_0
    /// drives subframes 0 + 1 and L_2 drives subframes 2 + 3 (never
    /// the per-subframe delta-decoded lags).  We exercise this
    /// indirectly by confirming that the chain runs without panicking
    /// on a four-subframe input whose lag pattern would otherwise be
    /// ambiguous.
    #[test]
    fn apply_post_filter_does_not_panic_on_typical_lags() {
        let mut st = SynthesisState::new();
        let lsp_q = st.prev_lsp;
        let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (n, s) in pcm.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / 50.0).sin() * 0.3;
        }
        let lags = [50, 51, 48, 49];
        st.apply_post_filter(&lsp_q, &lags, Rate::High, &mut pcm);
        for v in pcm.iter() {
            assert!(v.is_finite());
        }
    }

    // -- §3.1 / 2.6 LSP stability procedure (eq. 6–7.3) -----------------

    fn lsp_from_omegas(om: [f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
        let mut out = [0.0f32; LPC_ORDER];
        for i in 0..LPC_ORDER {
            out[i] = om[i].cos();
        }
        out
    }

    fn omega_min_gap_hz(lsp: &[f32; LPC_ORDER]) -> f32 {
        let mut min = f32::INFINITY;
        for j in 0..LPC_ORDER - 1 {
            let g = lsp[j].clamp(-1.0, 1.0).acos() - lsp[j + 1].clamp(-1.0, 1.0).acos();
            // Cosine is monotone-decreasing, so the well-ordered case has
            // acos(p̃_j) < acos(p̃_{j+1}), i.e. `g` is negative; the
            // *frequency gap* is then -g · fs / (2π).
            let hz = -g * SAMPLE_RATE_HZ as f32 / std::f32::consts::TAU;
            if hz < min {
                min = hz;
            }
        }
        min
    }

    #[test]
    fn enforce_lsp_stability_preserves_already_stable_vector() {
        // Construct an LSP whose angular frequencies are spaced 200 Hz
        // apart — well above the spec's 31.25 Hz floor. The procedure
        // must not perturb it (modulo the outer-root clamp).
        let mut omegas = [0.0f32; LPC_ORDER];
        let two_pi_per_fs = std::f32::consts::TAU / SAMPLE_RATE_HZ as f32;
        for i in 0..LPC_ORDER {
            omegas[i] = (300.0 + 200.0 * i as f32) * two_pi_per_fs;
        }
        let lsp_in = lsp_from_omegas(omegas);
        let (lsp_out, converged) = enforce_lsp_stability(&lsp_in, LSP_STABILITY_DELTA_MIN_HZ);
        assert!(converged, "already-stable vector must converge in pass 1");
        for i in 0..LPC_ORDER {
            assert!(
                (lsp_in[i] - lsp_out[i]).abs() < 1.0e-4,
                "dim {i}: stable input must be left alone (in {:.6}, out {:.6})",
                lsp_in[i],
                lsp_out[i],
            );
        }
    }

    #[test]
    fn enforce_lsp_stability_spreads_out_of_order_pair_around_midpoint() {
        // Inject a single out-of-order pair (dims 3 and 4 swapped) and
        // confirm the procedure repairs ordering with a Δ_min-wide gap.
        let two_pi_per_fs = std::f32::consts::TAU / SAMPLE_RATE_HZ as f32;
        let mut omegas = [0.0f32; LPC_ORDER];
        for i in 0..LPC_ORDER {
            omegas[i] = (300.0 + 250.0 * i as f32) * two_pi_per_fs;
        }
        omegas.swap(3, 4); // inject one frequency-domain inversion
        let lsp_in = lsp_from_omegas(omegas);
        let (lsp_out, converged) = enforce_lsp_stability(&lsp_in, LSP_STABILITY_DELTA_MIN_HZ);
        assert!(converged, "single inversion must converge inside cap");
        let min_gap_hz = omega_min_gap_hz(&lsp_out);
        // Allow a small tolerance for f32 round-trip through acos/cos.
        assert!(
            min_gap_hz >= LSP_STABILITY_DELTA_MIN_HZ - 0.5,
            "min frequency gap {:.3} Hz must be ≥ Δ_min ({} Hz)",
            min_gap_hz,
            LSP_STABILITY_DELTA_MIN_HZ,
        );
    }

    #[test]
    fn enforce_lsp_stability_erasure_uses_wider_delta_min() {
        // The same input that the 31.25 Hz path leaves untouched (gaps
        // are 200 Hz) should be widened by the erasure path's 62.5 Hz
        // floor only if any gap falls below 62.5 — but with 200 Hz
        // spacing it doesn't, so the erasure path is also a no-op.
        // Constructing a deliberately tight LSP shows the floor difference.
        let two_pi_per_fs = std::f32::consts::TAU / SAMPLE_RATE_HZ as f32;
        let mut omegas = [0.0f32; LPC_ORDER];
        for i in 0..LPC_ORDER {
            // 50 Hz spacing: above normal floor (31.25), below erasure (62.5).
            omegas[i] = (300.0 + 50.0 * i as f32) * two_pi_per_fs;
        }
        let lsp_in = lsp_from_omegas(omegas);
        let (lsp_normal, _) = enforce_lsp_stability(&lsp_in, LSP_STABILITY_DELTA_MIN_HZ);
        let (lsp_erased, _) = enforce_lsp_stability(&lsp_in, LSP_STABILITY_DELTA_MIN_ERASURE_HZ);
        let gap_normal = omega_min_gap_hz(&lsp_normal);
        let gap_erased = omega_min_gap_hz(&lsp_erased);
        // Normal path: 50 Hz spacing is already above its 31.25 Hz floor,
        // so the procedure is a no-op and the minimum gap stays at 50.
        assert!(
            gap_normal >= LSP_STABILITY_DELTA_MIN_HZ - 0.5,
            "normal path must hit ≥ 31.25 Hz; got {gap_normal:.3}"
        );
        // Erasure path: 50 Hz is below the 62.5 Hz floor, so the procedure
        // must widen at least one pair. The spec's iterative spread does
        // not guarantee every pair reaches `Δ_min` on a global-cascade
        // input within the 10-iteration cap, but the minimum gap must
        // strictly exceed the normal-path leave-alone gap, proving the
        // erasure variant engaged.
        assert!(
            gap_erased > gap_normal,
            "erasure-variant gap ({gap_erased:.3}) must exceed normal gap \
             ({gap_normal:.3}) when the input violates the wider floor"
        );
        // And it must move toward Δ_min_erasure even if it doesn't get
        // all the way there in 10 iterations.
        assert!(
            gap_erased >= LSP_STABILITY_DELTA_MIN_HZ,
            "erasure-variant gap ({gap_erased:.3}) must still respect the \
             normal floor at minimum"
        );
    }

    #[test]
    fn enforce_lsp_stability_converges_for_typical_decoded_lsp() {
        // The factorial scalar VQ in `dequantise_lsp` produces nearly-
        // monotone outputs for every reachable index triple; the
        // stability procedure should converge for all of them. Sample a
        // grid of indices and verify convergence + monotonicity in
        // angular-frequency space.
        let probes: &[[u32; 3]] = &[
            [0, 0, 0],
            [0xFF, 0xFF, 0xFF],
            [0x55, 0xAA, 0x33],
            [0x12, 0x34, 0x56],
            [0x80, 0x40, 0x20],
        ];
        for idx in probes {
            let lsp_q = dequantise_lsp(idx);
            // Monotone-decreasing in cosine domain ⇔ monotone-increasing
            // in angular-frequency domain.
            for j in 0..LPC_ORDER - 1 {
                assert!(
                    lsp_q[j] > lsp_q[j + 1],
                    "idx {idx:?}: cosine LSP must be strictly decreasing \
                     ({} -> {} at dim {j})",
                    lsp_q[j],
                    lsp_q[j + 1],
                );
            }
            let gap = omega_min_gap_hz(&lsp_q);
            assert!(
                gap >= LSP_STABILITY_DELTA_MIN_HZ - 0.5,
                "idx {idx:?}: dequantise_lsp must hit ≥ 31.25 Hz floor; got {gap:.3}"
            );
        }
    }

    #[test]
    fn enforce_lsp_stability_handles_severely_degenerate_input() {
        // All-equal LSPs are the worst case for §2.6: every pair needs
        // spreading. Confirm the iterative procedure does not blow up
        // and respects the outer-root clamp (|cos ω| < 1).
        let lsp_in = [0.5f32; LPC_ORDER];
        let (lsp_out, _converged) = enforce_lsp_stability(&lsp_in, LSP_STABILITY_DELTA_MIN_HZ);
        for &v in &lsp_out {
            assert!(v.abs() < 1.0, "outer-root clamp violated: {v}");
            assert!(v.is_finite());
        }
        // Convergence not guaranteed for this pathological input (the
        // procedure may hit the iteration cap), but the post-procedure
        // vector must still be stable enough for `lsp_to_lpc` to produce
        // a finite filter.
        let a = lsp_to_lpc(&lsp_out);
        for v in &a {
            assert!(v.is_finite(), "LPC coefficient must be finite");
        }
    }
}
