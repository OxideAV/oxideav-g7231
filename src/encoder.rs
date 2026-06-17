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
    ERASURE_ATTENUATION_DB_PER_FRAME, ERASURE_CLASSIFIER_HISTORY_LEN,
    ERASURE_CLASSIFIER_LAG_RADIUS, ERASURE_MUTE_AFTER_FRAMES, ERASURE_VOICED_THRESHOLD_DB,
    FRAME_SIZE_SAMPLES, HIGH_RATE_BYTES, LOW_RATE_BYTES, LPC_ORDER, LSP_PREDICTOR_BE,
    LSP_STABILITY_DELTA_MIN_ERASURE_HZ, LSP_STABILITY_DELTA_MIN_HZ, LSP_STABILITY_MAX_ITERATIONS,
    PITCH_MAX, PITCH_MIN, POSTFILTER_AGC_ALPHA, POSTFILTER_AGC_INIT_GAIN,
    POSTFILTER_LTP_GAMMA_HIGH, POSTFILTER_LTP_GAMMA_LOW, POSTFILTER_LTP_PRED_GAIN_DB_MIN,
    POSTFILTER_LTP_SEARCH_RADIUS, POSTFILTER_TILT_BASE, POSTFILTER_TILT_SMOOTH_ALPHA,
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

/// Formant-postfilter bandwidth expansion using the spec's *exact*
/// Q15-quantised weighting tables instead of a recomputed floating-point
/// `gamma^i`.
///
/// G.723.1 §3.8 (eq. 49.1–49.3) forms the ARMA formant postfilter
/// `A(z/γ₁) / A(z/γ₂)` with γ₁ = 0.65 (zeros) and γ₂ = 0.75 (poles). The
/// reference codec does **not** evaluate `γ^i` afresh at run time; it
/// scales each LPC coefficient by a precomputed weight `PostFiltZeroTable`
/// / `PostFiltPoleTable` (§2.18) carried in Q15. Those weights are the
/// fixed-point powers `round(γ^(i+1) · 2¹⁵)` for `i = 0..9`. Threading the
/// table through verbatim applies all ten weights from a single Q15
/// constant instead of a repeatedly-multiplied float `gamma^i`, which
/// accumulates rounding error tap-by-tap across the order-10 filter.
///
/// `weights_q15[k]` multiplies `a[k + 1]` (the order-`k+1` LPC tap); the
/// `a[0] = 1` gain tap is left untouched. This is the spec-exact source of
/// the formant postfilter coefficients, replacing the float `gamma^i`
/// path so the weighting bit-matches the ITU table.
fn postfilter_expand(
    a: &[f32; LPC_ORDER + 1],
    weights_q15: &[i16; LPC_ORDER],
) -> [f32; LPC_ORDER + 1] {
    let mut out = *a;
    for k in 0..LPC_ORDER {
        out[k + 1] = a[k + 1] * (weights_q15[k] as f32 / 32768.0);
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

/// Erasure-concealment LSP extrapolation toward the long-term DC vector
/// (G.723.1 §3.10.1).
///
/// With the decoded residual `ẽ_n` forced to zero, the concealed LSP is
/// `p̃_n = b_e · (p̃_{n-1} − p_DC) + p_DC`, where `b_e = 23/32`
/// ([`LSP_PREDICTOR_BE`]) and `p_DC` is the long-term DC vector. The
/// predictor is defined on LSP *angular frequencies* (`ω`), so the stored
/// cosine-domain `prev_lsp_cos` is mapped to `ω = acos(cos ω)`, the leak is
/// applied component-wise against the DC vector's angular frequencies, and
/// the result is mapped back to the cosine domain. Each erased frame pulls
/// every LSP frequency a fraction `1 − b_e = 9/32` of the way toward its DC
/// value, so a sustained erasure run relaxes the spectral envelope toward
/// the long-term mean instead of freezing the last good envelope.
///
/// The returned vector is *not* stability-checked here; the caller runs the
/// §3.10.1 wider-`Δ_min` ordering procedure on it.
pub(crate) fn extrapolate_lsp_toward_dc(
    prev_lsp_cos: &[f32; LPC_ORDER],
    b_e: f32,
) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let omega_prev = prev_lsp_cos[i].clamp(-1.0, 1.0).acos();
        // DC vector in the same angular-frequency domain.
        let omega_dc = crate::tables::lsp_dc_omega(i);
        let omega = b_e * (omega_prev - omega_dc) + omega_dc;
        out[i] = omega.clamp(0.0, std::f32::consts::PI).cos();
    }
    out
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

/// Sample position of the ACELP pulse on `track` (0..=3) at 3-bit slot
/// `k` (0..=7), with the global 1-bit `grid` shift applied. Returns
/// `None` for the Table-1 "(60)" / "(62)" entries (track 2 / 3, `k = 7`
/// on the even grid) that fall at or beyond the 60-sample subframe — i.e.
/// the pulse is absent.
///
/// This is the exact geometry of ITU-T G.723.1 §2.16 Table 1 (ACELP
/// excitation codebook): the four tracks have even bases 0, 2, 4, 6 and
/// stride 8, and "the positions of all pulses can be simultaneously
/// shifted by one (to occupy odd positions)" via the grid bit. The
/// canonical lookup lives in [`crate::spec_tables::acelp_track_position`];
/// this thin wrapper adapts the encoder/decoder's `usize`/`u8` indices to
/// the typed accessor.
fn acelp_pos_of(track: usize, k: u32, grid: u8) -> Option<usize> {
    let t = crate::spec_tables::AcelpTrack::ALL[track];
    crate::spec_tables::acelp_track_position(t, k as usize, grid != 0)
}

/// Four-pulse ACELP fixed-codebook search. Each of the 4 pulses lives on
/// its own track with stride-8 positions (8 candidate slots per track);
/// the grid bit shifts the whole pulse set by +1 so both even and odd
/// sample positions are reachable — the §2.16 Table 1 structure.
///
/// Track layout (grid 0, even positions):
///
/// ```text
///   T0: 0,  8, 16, 24, 32, 40, 48, 56
///   T1: 2, 10, 18, 26, 34, 42, 50, 58
///   T2: 4, 12, 20, 28, 36, 44, 52, (60)
///   T3: 6, 14, 22, 30, 38, 46, 54, (62)
/// ```
///
/// Grid 1 shifts each position by +1 (odd positions). Slots whose
/// position lands at or beyond 60 — track 2 / 3 at `k = 7` on the even
/// grid — encode an *absent* pulse per the Table 1 note. The 3-bit
/// position code + 1-bit sign per track, plus the 1-bit grid per
/// subframe, give the 17-bit algebraic codebook; the search scans
/// 2 × 4 × 8 = 64 candidates.
///
/// After the per-track greedy pick, the algorithm does two passes of
/// coordinate-descent refinement: for each pulse in turn it re-optimises
/// its (position, sign) given the other three fixed — so pulses that
/// were sub-optimal because of correlation with another pulse on the
/// grid get adjusted.
fn acelp_4pulse_search(target: &[f32; SUBFRAME_SIZE], h: &[f32]) -> ([u32; 4], [i32; 4], u8) {
    let d = compute_correlations(target, h);
    let positions_per_track: usize = 8;

    let mut best_grid = 0u8;
    let mut best_err = f32::INFINITY;
    let mut best_positions = [0u32; 4];
    let mut best_signs = [1i32; 4];

    for grid in 0..2u8 {
        // Pass 1: per-track greedy pick (initial solution).
        let mut positions = [0u32; 4];
        let mut signs = [1i32; 4];
        for track in 0..4usize {
            let mut best_gain2 = 0.0f32;
            let mut best_k = 0u32;
            let mut best_sign = 1i32;
            for k in 0..positions_per_track {
                let pos = match acelp_pos_of(track, k as u32, grid) {
                    Some(p) => p,
                    None => continue,
                };
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
                    if let Some(pos) = acelp_pos_of(t2, positions[t2], grid) {
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
                    let pos = match acelp_pos_of(track, k as u32, grid) {
                        Some(p) => p,
                        None => continue,
                    };
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
            if let Some(pos) = acelp_pos_of(track, positions[track], grid) {
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
/// mirror the §2.16 Table 1 layout used by [`acelp_4pulse_search`] (even
/// bases 0, 2, 4, 6; stride 8; the grid bit is the global +1 odd shift).
/// A 3-bit slot whose position lands at or beyond the subframe boundary
/// (the Table 1 "(60)" / "(62)" entries) places no pulse — i.e. an
/// absent pulse.
pub(crate) fn place_pulses(
    positions: &[u32; 4],
    signs: [i32; 4],
    grid: u8,
    out: &mut [f32; SUBFRAME_SIZE],
) {
    out.fill(0.0);
    for track in 0..4usize {
        if let Some(pos) = acelp_pos_of(track, positions[track], grid) {
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
    /// Inter-subframe-smoothed first-order normalised autocorrelation
    /// `k1` of the synthesis input, driving the §3.8 tilt-compensation
    /// coefficient `μ = POSTFILTER_TILT_BASE · k1` (eq. 49.2).
    /// `(1 − POSTFILTER_TILT_SMOOTH_ALPHA)·k1[prev] +
    ///  POSTFILTER_TILT_SMOOTH_ALPHA·k`, where `k = r(1)/r(0)` is recomputed
    /// per subframe.
    pf_tilt_k1: f32,
    /// Smoothed AGC gain `g[n]` per G.723.1 §3.9 eq. 51, persisting across
    /// subframes (and frames). Initialised to `POSTFILTER_AGC_INIT_GAIN` at
    /// cold start per §3.11.
    pf_agc_gain: f32,
    // Frame-erasure / SID concealment state -------------------------------
    /// Last decoded pitch lag — extrapolated during erasures.
    pf_last_lag: i32,
    /// Last decoded (g_adapt, g_fixed) — attenuated during erasures.
    pf_last_gain_adapt: f32,
    pf_last_gain_fixed: f32,
    /// Number of consecutive erased frames (0 = good frame most recent).
    pf_erased_run: u32,
    /// Saved `L_2` (last good frame's third-subframe lag) — feeds the
    /// G.723.1 §3.10.2 voiced/unvoiced classifier and the voiced-path
    /// periodic-excitation regenerator.
    pf_last_lag2: i32,
    /// Average of the last good frame's subframe-2 / subframe-3 fixed-
    /// codebook gains — drives the unvoiced concealment branch of
    /// §3.10.2 ("the saved average of subframe-2/3 gain indices").
    pf_last_gain_unvoiced: f32,
    /// Trailing 120 samples of post-filtered decoder output. The §3.10.2
    /// classifier cross-correlates this with itself shifted by `L_2 ± 3`
    /// to decide voiced vs unvoiced and to refine the pitch period used
    /// for the voiced-path periodic regenerator.
    pf_pcm_hist: [f32; ERASURE_CLASSIFIER_HISTORY_LEN],
}

/// Whole-frame forward context for the §3.6 pitch postfilter (trace §8):
/// the forward cross-correlation `x[n + M_f]` reaches into the next
/// subframe, so the postfilter needs the raw (pre-postfilter) synthesis
/// samples of the whole frame plus the current subframe's start offset.
#[derive(Copy, Clone)]
struct ForwardCtx<'a> {
    /// Raw (pre-postfilter) synthesis samples of the whole 240-sample frame.
    raw_frame: &'a [f32; FRAME_SIZE_SAMPLES],
    /// Frame-relative start index of the subframe being post-filtered.
    sf_start: usize,
}

impl SynthesisState {
    pub fn new() -> Self {
        // §3.11: every static decoder variable is zeroed *except* the
        // previous LSP vector, which initialises to the long-term DC vector
        // p_DC (the spec's predictor reference, not an evenly-spaced
        // placeholder). Stored in the synthesiser's cosine domain.
        let prev_lsp = crate::tables::lsp_dc_cosines();
        Self {
            prev_lsp,
            exc_history: [0.0; PITCH_MAX + SUBFRAME_SIZE],
            syn_mem: [0.0; LPC_ORDER],
            pf_syn_hist: [0.0; PITCH_MAX + SUBFRAME_SIZE],
            pf_num_mem: [0.0; LPC_ORDER],
            pf_den_mem: [0.0; LPC_ORDER],
            pf_tilt_prev: 0.0,
            pf_tilt_k1: 0.0,
            pf_agc_gain: POSTFILTER_AGC_INIT_GAIN,
            pf_last_lag: 60,
            pf_last_gain_adapt: 0.0,
            pf_last_gain_fixed: 0.0,
            pf_erased_run: 0,
            pf_last_lag2: 60,
            pf_last_gain_unvoiced: 0.0,
            pf_pcm_hist: [0.0; ERASURE_CLASSIFIER_HISTORY_LEN],
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
    ///
    /// In addition to the last-subframe state used by the legacy
    /// concealment path, this also captures:
    ///
    /// - `L_2` (third-subframe lag): drives the G.723.1 §3.10.2 voiced /
    ///   unvoiced classifier's cross-correlation window `L_2 ± 3` and
    ///   the voiced-path periodic regenerator's pitch period.
    /// - The average fixed-codebook gain across subframes 2 and 3:
    ///   spec-mandated drive level for the unvoiced concealment branch
    ///   ("the saved average of subframe-2/3 gain indices is used").
    fn record_last_frame(
        &mut self,
        lags: &[i32; SUBFRAMES_PER_FRAME],
        gain_idx: &[u32; SUBFRAMES_PER_FRAME],
    ) {
        self.pf_last_lag = lags[SUBFRAMES_PER_FRAME - 1];
        let (g_a, g_f) = dequantise_gain(gain_idx[SUBFRAMES_PER_FRAME - 1]);
        self.pf_last_gain_adapt = g_a;
        self.pf_last_gain_fixed = g_f;
        // G.723.1 §3.10.2 classifier inputs.
        self.pf_last_lag2 = lags[2];
        let (_, g_f2) = dequantise_gain(gain_idx[2]);
        let (_, g_f3) = dequantise_gain(gain_idx[3]);
        self.pf_last_gain_unvoiced = 0.5 * (g_f2 + g_f3);
        self.pf_erased_run = 0;
    }

    /// Update the trailing-PCM classifier history with the last
    /// `ERASURE_CLASSIFIER_HISTORY_LEN` samples of a freshly synthesised
    /// (post-filtered) frame. Called by the decoder entry points after
    /// `apply_post_filter` so the §3.10.2 classifier sees the same
    /// signal a downstream listener would.
    fn record_pcm_history(&mut self, pcm_f: &[f32; FRAME_SIZE_SAMPLES]) {
        let tail = FRAME_SIZE_SAMPLES - ERASURE_CLASSIFIER_HISTORY_LEN;
        self.pf_pcm_hist
            .copy_from_slice(&pcm_f[tail..tail + ERASURE_CLASSIFIER_HISTORY_LEN]);
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
        fwd: ForwardCtx<'_>,
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

        // Forward lookup `x[n + M_f]`: per G.723.1 §3.6 / trace §8 the
        // pitch postfilter is defined over the whole-frame synthesis
        // signal, so when `n + M_f >= 60` the read continues into the next
        // subframe of the raw (pre-postfilter) frame `raw_frame` rather
        // than being truncated. Only when the forward reach exceeds the
        // frame end (sample 239) does the contribution drop to zero.
        let (mf_best, cf, df) = self.ltp_search_forward(syn, fwd, m_lo, m_hi);

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
                let g = fwd.sf_start + n + m; // global frame index of x[n + M_f]
                if g < FRAME_SIZE_SAMPLES {
                    contrib[n] = fwd.raw_frame[g];
                }
                // Forward reach past the frame end (sample 239) → 0;
                // matches the correlation that produced cf/df.
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
        fwd: ForwardCtx<'_>,
        m_lo: i32,
        m_hi: i32,
    ) -> (i32, f32, f32) {
        let mut best_metric = -f32::INFINITY;
        let mut best = (m_lo, 0.0f32, 0.0f32);
        for m in m_lo..=m_hi {
            let mu = m as usize;
            let mut c = 0.0f32;
            let mut d = 0.0f32;
            for n in 0..SUBFRAME_SIZE {
                // x[n + M_f] over the whole-frame synthesis signal
                // (§3.6 / trace §8): reads into the next subframe of the
                // raw pre-postfilter frame; past the frame end → 0.
                let g = fwd.sf_start + n + mu;
                let p = if g < FRAME_SIZE_SAMPLES {
                    fwd.raw_frame[g]
                } else {
                    0.0
                };
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
        fwd: ForwardCtx<'_>,
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
        let after_pitch = self.ltp_post_filter_subframe(syn, fwd, lag, rate);

        // ---- 2. Formant post-filter A(z/γ₁) / A(z/γ₂). γ₁ < γ₂ widens
        // the formant bandwidth on the numerator and narrows it on the
        // denominator, emphasising the spectral peaks that carry speech
        // formants without shifting their centre frequency.
        // Use the spec's exact Q15-quantised §2.18 PostFilt weighting
        // tables (the fixed-point γ₁ = 0.65 / γ₂ = 0.75 powers) rather than
        // recomputing γ^i in float, so the formant postfilter coefficients
        // match the ITU reference weighting bit-for-bit.
        let a_num = postfilter_expand(a_sub, &crate::spec_tables::POSTFILTER_ZERO_Q15);
        let a_den = postfilter_expand(a_sub, &crate::spec_tables::POSTFILTER_POLE_Q15);
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

        // ---- 3. First-order tilt compensation per G.723.1 §3.8, eq. 49.2:
        //
        //   y[n] = x[n] − μ · x[n − 1],   μ = POSTFILTER_TILT_BASE · k1
        //
        // where `k1` is the inter-subframe-smoothed first-order normalised
        // autocorrelation `r(1)/r(0)` of the synthesis input `sy[n]`:
        //
        //   k1[s] = (1 − α_tilt) · k1[s − 1] + α_tilt · k,
        //   k = Σ sy[n]·sy[n − 1] / Σ sy[n]² ,   α_tilt = 1/4.
        //
        // Replaces the previous fixed-`μ = 0.25` shortcut so the tilt term
        // tracks the input's spectral tilt subframe-by-subframe instead of
        // applying a constant high-frequency cut.
        let mut r0 = 0.0f32;
        let mut r1 = 0.0f32;
        for n in 1..SUBFRAME_SIZE {
            r0 += syn[n] * syn[n];
            r1 += syn[n] * syn[n - 1];
        }
        // r0 picks up syn[0]² too — the missing term in the loop above.
        r0 += syn[0] * syn[0];
        let k = if r0 > 0.0 {
            (r1 / r0).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        self.pf_tilt_k1 = (1.0 - POSTFILTER_TILT_SMOOTH_ALPHA) * self.pf_tilt_k1
            + POSTFILTER_TILT_SMOOTH_ALPHA * k;
        let mu = POSTFILTER_TILT_BASE * self.pf_tilt_k1;
        let mut after_tilt = [0.0f32; SUBFRAME_SIZE];
        let mut prev = self.pf_tilt_prev;
        for n in 0..SUBFRAME_SIZE {
            let x = after_formant[n];
            after_tilt[n] = x - mu * prev;
            prev = x;
        }
        self.pf_tilt_prev = prev;

        // ---- 4. Adaptive gain scaling per G.723.1 §3.9, eq. 50–52:
        //
        //   g_s = sqrt( Σ sy²[n] / Σ pf²[n] ),    g_s = 1 if denominator is 0
        //   g[n] = (1 − α) · g[n − 1] + α · g_s,   α = 1/16
        //   q[n] = pf[n] · g[n] · (1 + α)
        //
        // `g_s` is constant over the subframe but the leaky-integrator
        // update of `g[n]` runs per sample so the gain transition between
        // subframes is smooth; the `(1 + α)` boost on the output undoes the
        // average attenuation introduced by the smoothing filter.
        // Replaces the previous α = 0.85 per-sample chase + `(e_in/e_out)`
        // target shortcut so the AGC follows the spec's leaky-integrator
        // shape exactly.
        let mut e_in = 0.0f32;
        let mut e_out = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            e_in += syn[n] * syn[n];
            e_out += after_tilt[n] * after_tilt[n];
        }
        let g_s = if e_out > 0.0 {
            (e_in / e_out).sqrt()
        } else {
            1.0
        };
        let alpha = POSTFILTER_AGC_ALPHA;
        let scale = 1.0 + alpha;
        for n in 0..SUBFRAME_SIZE {
            self.pf_agc_gain = (1.0 - alpha) * self.pf_agc_gain + alpha * g_s;
            out[n] = after_tilt[n] * self.pf_agc_gain * scale;
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
    /// LPC coefficients. `prev_lsp` is the *previous frame's* decoded LSP
    /// vector, captured before [`SynthesisState::synthesise`] advanced
    /// `self.prev_lsp` to `lsp_q`. `rate` selects the rate-specific LTP
    /// weighting in the pitch postfilter (§3.6).
    ///
    /// G.723.1 §3.6 specifies that the pitch postfilter uses `L_0` (the
    /// absolute lag of subframe 0) for subframes 0,1 and `L_2` (subframe
    /// 2's absolute lag) for subframes 2,3 — not the per-subframe
    /// delta-decoded lags. We respect that here.
    fn apply_post_filter(
        &mut self,
        prev_lsp: &[f32; LPC_ORDER],
        lsp_q: &[f32; LPC_ORDER],
        lags: &[i32; SUBFRAMES_PER_FRAME],
        rate: Rate,
        pcm: &mut [f32; FRAME_SIZE_SAMPLES],
    ) {
        // The formant postfilter A(z/γ₁)/A(z/γ₂) (§3.8) operates on the
        // same per-subframe quantised synthesis filter Ã_i(z) the LPC
        // synthesis stage used.  Those coefficients come from the §3.3 /
        // §2.7 (eq. 8) per-subframe LSP interpolation between the previous
        // frame's decoded LSP and the current frame's, with weights
        // (0.75/0.25), (0.5/0.5), (0.25/0.75), (0/1) for subframes 0..3 —
        // *not* a frame-constant LSP.  `synthesise()` already advanced
        // `self.prev_lsp` to `lsp_q`, so the caller passes the captured
        // pre-synthesis previous LSP here and we reproduce the identical
        // interpolation curve, keeping the postfilter's formant filter
        // matched to the synthesis filter subframe-for-subframe.
        // G.723.1 §3.6 / trace §8: the pitch postfilter is defined over the
        // *whole-frame* synthesis signal `{sy[n]}_{0..239}` — the forward
        // cross-correlation `x[n + M_f]` (M_f ≈ L_i + 3) for samples near a
        // subframe's tail reaches into the *next* subframe. Snapshot the
        // raw (pre-postfilter) synthesis frame up front so each subframe's
        // forward LTP search can read across the subframe boundary instead
        // of truncating the correlation window at sample 60.
        let raw_frame = *pcm;
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, prev_lsp, lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);
            // Reference lag for the LTP postfilter: L_0 covers subframes
            // 0,1; L_2 covers subframes 2,3 (G.723.1 §3.6 prose).
            let ref_lag = if s < 2 { lags[0] } else { lags[2] };
            let start = s * SUBFRAME_SIZE;
            let end = start + SUBFRAME_SIZE;
            let mut syn = [0.0f32; SUBFRAME_SIZE];
            syn.copy_from_slice(&pcm[start..end]);
            let mut post = [0.0f32; SUBFRAME_SIZE];
            let fwd = ForwardCtx {
                raw_frame: &raw_frame,
                sf_start: start,
            };
            self.post_filter_subframe(&a_sub, &syn, fwd, ref_lag, rate, &mut post);
            pcm[start..end].copy_from_slice(&post);
        }
    }

    /// Concealment path for SID / erased packets — G.723.1 §3.10.
    ///
    /// Implements the spec's two-stage interpolation:
    ///
    /// 1. **LSP interpolation** (§3.10.1): reuse the previous decoded
    ///    LSP vector, re-applying the §2.6 ordering procedure with the
    ///    relaxed `Δ_min = 62.5 Hz` so extrapolation drift can be pulled
    ///    back without destroying the envelope.
    /// 2. **Residual interpolation** (§3.10.2): a voiced/unvoiced
    ///    classifier cross-correlates the saved trailing 120 samples of
    ///    post-filtered output with itself shifted by `L_2 ± 3`. The
    ///    prediction gain (in dB) decides the branch:
    ///    - prediction gain `> 0.58 dB` ⇒ voiced: regenerate a periodic
    ///      excitation at the classifier's pitch period from the saved
    ///      excitation history.
    ///    - prediction gain `≤ 0.58 dB` ⇒ unvoiced: regenerate a uniform
    ///      pseudo-random excitation scaled by the saved average gain
    ///      across subframes 2 and 3 (`pf_last_gain_unvoiced`).
    ///
    /// Sustained erasure attenuates the regenerated vector by an extra
    /// `2.5 dB` per consecutive interpolated frame and mutes completely
    /// after `3` interpolated frames (`ERASURE_MUTE_AFTER_FRAMES`).
    ///
    /// Returns 240 concealed S16 samples.
    pub fn decode_erased(&mut self) -> [i16; FRAME_SIZE_SAMPLES] {
        self.pf_erased_run = self.pf_erased_run.saturating_add(1);

        // §3.10.2 attenuation: 2.5 dB per consecutive erased frame, mute
        // completely after `ERASURE_MUTE_AFTER_FRAMES` (3) frames.
        let atten = if self.pf_erased_run > ERASURE_MUTE_AFTER_FRAMES {
            0.0
        } else {
            let db = ERASURE_ATTENUATION_DB_PER_FRAME * self.pf_erased_run as f32;
            10f32.powf(-db / 20.0)
        };

        // §3.10.1: erasure LSP interpolation. The decoded residual ẽ_n is
        // forced to zero (step 1) and the predicted vector uses the
        // erasure predictor b_e = 23/32 (step 2), giving
        //   p̃_n = ẽ_n + p̄_n + p_DC = b_e · (p̃_{n-1} − p_DC) + p_DC.
        // The predictor operates on LSP *angular frequencies*, so convert
        // the stored cosine-domain previous LSP and the DC vector to ω,
        // leak ω toward the DC frequencies at rate 1 − b_e per erased
        // frame, then convert back. The wider Δ_min = 62.5 Hz stability
        // procedure (step 3) re-orders the extrapolated vector.
        let lsp_extrap = extrapolate_lsp_toward_dc(&self.prev_lsp, LSP_PREDICTOR_BE);
        let (lsp_q, _converged) =
            enforce_lsp_stability(&lsp_extrap, LSP_STABILITY_DELTA_MIN_ERASURE_HZ);

        // §3.10.2 voiced/unvoiced classifier: cross-correlate the saved
        // post-filtered PCM history with itself shifted by `L_2 ± 3` and
        // take the largest prediction gain.
        let (voiced, classifier_lag) = self.classify_erasure_voicing();

        // Pseudo-random innovation generator for the unvoiced branch.
        // Deterministic LCG so concealment is reproducible.
        let mut lcg = 0xDEADBEEFu32.wrapping_add(self.pf_erased_run.wrapping_mul(0x9E37_79B9));
        let mut next_rand = || -> f32 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0
        };

        // Scaled drive level. The voiced branch reuses the saved
        // last-subframe (g_adapt) since the excitation is already
        // shaped through the adaptive codebook; the unvoiced branch
        // uses the saved average of subframes 2 and 3 fixed gains per
        // §3.10.2.
        let g_adapt = self.pf_last_gain_adapt * atten;
        let g_fixed_unvoiced = self.pf_last_gain_unvoiced * atten;
        // Voiced uses the classifier-estimated pitch; unvoiced has no
        // periodic structure, so fall back to the last good lag to keep
        // the adaptive-codebook lookup well-defined (the contribution
        // multiplies to zero anyway when the classifier reports unvoiced
        // and the unvoiced branch suppresses `g_adapt`).
        let lag = if voiced {
            classifier_lag
        } else {
            self.pf_last_lag
        }
        .clamp(PITCH_MIN as i32, PITCH_MAX as i32);

        let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, &self.prev_lsp, &lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);

            let mut adaptive = [0.0f32; SUBFRAME_SIZE];
            copy_adaptive(&self.exc_history, lag, &mut adaptive);

            // §3.10.2 branch.
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            if voiced {
                // Voiced: periodic excitation at the classifier's pitch.
                // The adaptive codebook tap already replays the periodic
                // structure, so suppress the fixed-codebook innovation
                // (clause text: "periodic excitation at the classifier's
                // pitch period").
                for (slot, a) in exc.iter_mut().zip(adaptive.iter()) {
                    *slot = g_adapt * *a;
                }
            } else {
                // Unvoiced: uniform random, scaled by the saved average
                // fixed-codebook gain. The adaptive contribution is
                // dropped — an unvoiced frame has no pitch structure to
                // extend.
                let _ = adaptive;
                for slot in exc.iter_mut() {
                    *slot = g_fixed_unvoiced * next_rand();
                }
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
            //
            // Concealment synthesises and post-filters subframe-by-subframe
            // in one pass, so the whole-frame forward LTP context isn't
            // available here — we present the current subframe at frame
            // offset 0 with no successor samples, so the §3.6 forward reach
            // past sample 60 contributes zero (the original window-shrinkage
            // behaviour, kept deliberately gentle on the concealed signal).
            let mut raw_sf = [0.0f32; FRAME_SIZE_SAMPLES];
            raw_sf[..SUBFRAME_SIZE].copy_from_slice(&syn);
            let mut post = [0.0f32; SUBFRAME_SIZE];
            let fwd = ForwardCtx {
                raw_frame: &raw_sf,
                sf_start: 0,
            };
            self.post_filter_subframe(&a_sub, &syn, fwd, lag, Rate::High, &mut post);
            let start = s * SUBFRAME_SIZE;
            pcm[start..start + SUBFRAME_SIZE].copy_from_slice(&post);

            // Advance excitation history with the concealed excitation.
            self.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.exc_history.len() - SUBFRAME_SIZE;
            self.exc_history[tail..].copy_from_slice(&exc);
        }

        // Persist the extrapolated LSP as the previous-frame vector so a
        // sustained erasure run keeps leaking toward the DC vector frame
        // after frame (§3.10.1: p̃_{n-1} is the previous *decoded* LSP, so
        // each concealed frame feeds the next), and so a good frame that
        // ends the run interpolates from the concealed envelope rather than
        // the stale pre-erasure one.
        self.prev_lsp = lsp_q;

        // Update classifier history with the concealed PCM so a
        // subsequent erasure in the same run sees a fresh tail.
        self.record_pcm_history(&pcm);

        to_i16_frame(&pcm)
    }

    /// G.723.1 §3.10.2 voiced/unvoiced classifier.
    ///
    /// Cross-correlates the saved post-filtered PCM history with itself
    /// shifted by `L_2 ± ERASURE_CLASSIFIER_LAG_RADIUS` and returns
    /// `(voiced, best_lag)`:
    ///
    /// - `voiced = true` if the best-lag prediction gain (in dB) exceeds
    ///   `ERASURE_VOICED_THRESHOLD_DB` (0.58 dB).
    /// - `best_lag` is the lag in `L_2 ± 3` maximising the prediction
    ///   gain — only meaningful when `voiced` is `true`; for unvoiced it
    ///   still returns the maximising lag but callers should fall back
    ///   to `pf_last_lag`.
    fn classify_erasure_voicing(&self) -> (bool, i32) {
        let hist = &self.pf_pcm_hist;
        let n = hist.len();

        // Total energy of the trailing window.
        let mut energy: f32 = 0.0;
        for &v in hist.iter() {
            energy += v * v;
        }
        if energy <= 0.0 {
            return (false, self.pf_last_lag2);
        }

        let centre = self.pf_last_lag2;
        let radius = ERASURE_CLASSIFIER_LAG_RADIUS;
        let mut best_lag = centre;
        let mut best_gain_db = f32::NEG_INFINITY;
        for d in -radius..=radius {
            let lag = (centre + d).clamp(PITCH_MIN as i32, PITCH_MAX as i32);
            let lag_u = lag as usize;
            if lag_u >= n {
                continue;
            }
            // Forward auto-correlation:
            //   C = Σ_{k=lag..n} hist[k] · hist[k - lag]
            //   E = Σ_{k=lag..n} hist[k - lag]^2
            // Prediction gain (per the §3.6 / §3.10.2 prose):
            //   −10·log10(1 − C² / (E · T_en))
            // where `T_en` is the energy of the analysis segment.
            let mut c: f32 = 0.0;
            let mut e_lag: f32 = 0.0;
            let mut t_en: f32 = 0.0;
            for k in lag_u..n {
                let cur = hist[k];
                let prev = hist[k - lag_u];
                c += cur * prev;
                e_lag += prev * prev;
                t_en += cur * cur;
            }
            if e_lag <= 0.0 || t_en <= 0.0 {
                continue;
            }
            let ratio = (c * c) / (e_lag * t_en);
            // ratio is bounded in [0, 1] by Cauchy–Schwarz; clamp for
            // floating-point slop so the log is well-defined.
            let one_minus = (1.0 - ratio).clamp(1.0e-30, 1.0);
            let gain_db = -10.0 * one_minus.log10();
            if gain_db > best_gain_db {
                best_gain_db = gain_db;
                best_lag = lag;
            }
        }

        (best_gain_db > ERASURE_VOICED_THRESHOLD_DB, best_lag)
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

        // Capture the previous frame's LSP before synthesise() advances
        // self.prev_lsp to lsp_q, so the postfilter can reproduce the same
        // §2.7 per-subframe interpolated formant filter the synthesis used.
        let prev_lsp_snapshot = self.prev_lsp;
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
        self.apply_post_filter(&prev_lsp_snapshot, &lsp_q, &lags, Rate::Low, &mut pcm_f);
        self.record_last_frame(&lags, &gain);
        self.record_pcm_history(&pcm_f);
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

        // Capture the previous frame's LSP before synthesise() advances
        // self.prev_lsp to lsp_q (see decode_acelp for the rationale).
        let prev_lsp_snapshot = self.prev_lsp;
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
        self.apply_post_filter(&prev_lsp_snapshot, &lsp_q, &lags, Rate::High, &mut pcm_f);
        self.record_last_frame(&lags, &gain);
        self.record_pcm_history(&pcm_f);
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

    /// Test helper: run the post-filter on a single subframe with no
    /// whole-frame forward LTP context (the subframe sits at frame offset 0
    /// with no successor samples, so the §3.6 forward reach contributes
    /// zero — exactly the per-subframe behaviour these tilt / AGC / LTP
    /// unit tests intend to exercise).
    fn pf_sf(
        st: &mut SynthesisState,
        a: &[f32; LPC_ORDER + 1],
        syn: &[f32; SUBFRAME_SIZE],
        lag: i32,
        rate: Rate,
        out: &mut [f32; SUBFRAME_SIZE],
    ) {
        let mut raw = [0.0f32; FRAME_SIZE_SAMPLES];
        raw[..SUBFRAME_SIZE].copy_from_slice(syn);
        let fwd = ForwardCtx {
            raw_frame: &raw,
            sf_start: 0,
        };
        st.post_filter_subframe(a, syn, fwd, lag, rate, out);
    }

    /// Test helper: run only the §3.6 pitch postfilter on a single subframe
    /// with no whole-frame forward context (forward reach past sample 60 →
    /// zero), for the LTP-specific unit tests.
    fn ltp_sf(
        st: &SynthesisState,
        syn: &[f32; SUBFRAME_SIZE],
        lag: i32,
        rate: Rate,
    ) -> [f32; SUBFRAME_SIZE] {
        let mut raw = [0.0f32; FRAME_SIZE_SAMPLES];
        raw[..SUBFRAME_SIZE].copy_from_slice(syn);
        let fwd = ForwardCtx {
            raw_frame: &raw,
            sf_start: 0,
        };
        st.ltp_post_filter_subframe(syn, fwd, lag, rate)
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

    /// The ACELP fixed-codebook pulse geometry matches §2.16 Table 1:
    /// four tracks with even bases 0, 2, 4, 6 and stride 8, with the grid
    /// bit applying the global +1 odd shift. `acelp_pos_of` and
    /// `place_pulses` must agree, and the search must never emit a slot
    /// that decodes to a different sample than it placed.
    #[test]
    fn acelp_pulse_geometry_matches_table1() {
        // Even grid (shift = 0): each track's k = 0 hits its Table 1 base.
        assert_eq!(acelp_pos_of(0, 0, 0), Some(0));
        assert_eq!(acelp_pos_of(1, 0, 0), Some(2));
        assert_eq!(acelp_pos_of(2, 0, 0), Some(4));
        assert_eq!(acelp_pos_of(3, 0, 0), Some(6));
        // Stride 8 across the slots of track 0.
        assert_eq!(acelp_pos_of(0, 7, 0), Some(56));
        assert_eq!(acelp_pos_of(1, 7, 0), Some(58));
        // Table 1 "(60)" / "(62)" — track 2 / 3 at k = 7 on the even grid
        // fall outside the 60-sample subframe → absent pulse.
        assert_eq!(acelp_pos_of(2, 7, 0), None);
        assert_eq!(acelp_pos_of(3, 7, 0), None);
        // Odd grid (shift = 1) moves the whole set up by one.
        assert_eq!(acelp_pos_of(0, 0, 1), Some(1));
        assert_eq!(acelp_pos_of(2, 6, 1), Some(53));
        // Track 2 k = 7 was 60 (absent) on the even grid; on the odd grid
        // it would be 61 → still absent.
        assert_eq!(acelp_pos_of(2, 7, 1), None);

        // place_pulses agrees with acelp_pos_of for every present slot and
        // drops the absent ones.
        let positions = [3u32, 7, 7, 2]; // T2/T3 k=7 are absent on even grid
        let signs = [1i32, -1, 1, -1];
        let mut out = [0.0f32; SUBFRAME_SIZE];
        place_pulses(&positions, signs, 0, &mut out);
        // T0 k=3 → 0 + 24 = 24 (+1); T1 k=7 → 58 (−1); T3 k=2 → 22 (−1).
        assert_eq!(out[24], 1.0);
        assert_eq!(out[58], -1.0);
        assert_eq!(out[22], -1.0);
        // The two absent pulses placed nothing — exactly three non-zero
        // samples remain.
        let nonzero = out.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 3);
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
        assert_eq!(st.pf_agc_gain, POSTFILTER_AGC_INIT_GAIN);
        assert_eq!(st.pf_agc_gain, 1.0);
        assert_eq!(st.pf_tilt_k1, 0.0);
        assert_eq!(st.pf_erased_run, 0);
    }

    /// G.723.1 §3.8 eq. 49.2 tilt-compensation coefficient
    /// `k1 = (1 − α) · k1_prev + α · r(1)/r(0)` smooths across subframes
    /// (`α = POSTFILTER_TILT_SMOOTH_ALPHA`). Driving the post-filter with a
    /// strongly auto-correlated synthesis input (low-frequency dominated)
    /// must move the smoothed `pf_tilt_k1` toward the per-subframe `k` and
    /// stay bounded inside `[−1, 1]`.
    #[test]
    fn post_filter_tilt_k1_smooths_per_subframe_per_spec() {
        let mut st = SynthesisState::new();
        // Smooth low-pass: each sample is the running mean of the previous
        // two, so r(1)/r(0) is strongly positive and close to 1.
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        let mut acc = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            // Sum-of-cosines, period ~30 samples → strong r(1).
            let t = n as f32;
            acc = (t * 0.21).cos() * 0.5 + acc * 0.5;
            syn[n] = acc * 1000.0;
        }
        let a = default_a();
        let mut out = [0.0f32; SUBFRAME_SIZE];

        // First subframe: k1 starts at 0, gets pulled toward k by α.
        pf_sf(&mut st, &a, &syn, 60, Rate::High, &mut out);
        let k1_after_1 = st.pf_tilt_k1;
        assert!(
            k1_after_1 > 0.0,
            "low-pass synthesis input should push k1 positive, got {k1_after_1}"
        );
        assert!(
            k1_after_1.abs() <= 1.0,
            "k1 must stay inside [-1, 1], got {k1_after_1}"
        );

        // Drive the same input several more times and verify k1 monotonically
        // approaches the per-subframe k (leaky integrator). We don't pin the
        // exact terminal value because k itself depends on syn's endpoints
        // and the smoothing factor — but the magnitude should keep growing
        // (or hold steady once k is reached).
        let mut last = k1_after_1;
        for _ in 0..6 {
            pf_sf(&mut st, &a, &syn, 60, Rate::High, &mut out);
            assert!(
                st.pf_tilt_k1 >= last - 1e-4,
                "leaky integrator should be non-decreasing toward k: was {last}, now {}",
                st.pf_tilt_k1
            );
            assert!(st.pf_tilt_k1.abs() <= 1.0);
            last = st.pf_tilt_k1;
        }
    }

    /// G.723.1 §3.8 eq. 49.2 tilt: per-subframe `k = r(1)/r(0)` must use
    /// the synthesis-domain signal (no smoothing over k itself; the
    /// integrator runs on `k1`). Constant zero input must therefore produce
    /// `k = 0` and leave `pf_tilt_k1` unchanged.
    #[test]
    fn post_filter_tilt_k1_zero_input_zeroes_k() {
        let mut st = SynthesisState::new();
        st.pf_tilt_k1 = 0.4; // seed nontrivial state
        let zero = [0.0f32; SUBFRAME_SIZE];
        let a = default_a();
        let mut out = [0.0f32; SUBFRAME_SIZE];
        pf_sf(&mut st, &a, &zero, 60, Rate::High, &mut out);
        // k = 0 (r0 == r1 == 0 path), so the integrator decays:
        //   k1' = (1 − α) · k1 = 0.75 · 0.4 = 0.30
        let expected = (1.0 - POSTFILTER_TILT_SMOOTH_ALPHA) * 0.4;
        assert!(
            (st.pf_tilt_k1 - expected).abs() < 1e-6,
            "expected k1' = {expected}, got {}",
            st.pf_tilt_k1
        );
    }

    /// G.723.1 §3.9 eq. 51 AGC: `g[n] = (1 − α) · g[n − 1] + α · g_s` with
    /// `α = 1/16`. When the post-filter doesn't change the energy
    /// (`g_s ≈ 1`), the smoothed gain stays at its initial unity value and
    /// the output reaches `pf[n] · 1 · (1 + α) = pf[n] · 17/16`. Driving
    /// the filter with zero input gives `pf[n] = 0` regardless, but we can
    /// verify `pf_agc_gain` does not drift away from unity when fed silence.
    #[test]
    fn post_filter_agc_holds_unity_on_silence() {
        let mut st = SynthesisState::new();
        let g0 = st.pf_agc_gain;
        let zero = [0.0f32; SUBFRAME_SIZE];
        let a = default_a();
        let mut out = [0.0f32; SUBFRAME_SIZE];
        pf_sf(&mut st, &a, &zero, 60, Rate::High, &mut out);
        // g_s degenerate-path defaults to 1 (eq. 50 "set to 1 if denominator
        // is 0"), so the leaky integrator pulls toward unity from unity:
        // g[n] stays at 1.
        for n in 0..SUBFRAME_SIZE {
            assert!(
                out[n].abs() < 1e-6,
                "silence-in → silence-out, got {} at {n}",
                out[n]
            );
        }
        assert!(
            (st.pf_agc_gain - g0).abs() < 1e-6,
            "AGC should stay at unity on silence, drifted to {}",
            st.pf_agc_gain
        );
    }

    /// G.723.1 §3.9 eq. 51 AGC: with `α = 1/16` and a single subframe at
    /// constant `g_s`, the per-sample integrator runs `g[n] = (1 − α) g[n−1]
    /// + α · g_s`. Closed form: starting from `g0`, after `N` samples,
    /// `g[N − 1] = g0 + (g_s − g0) · (1 − (1 − α)^N)`. For our
    /// `SUBFRAME_SIZE = 60` and `α = 1/16`, `(1 − 1/16)^60 ≈ 0.0205`, so
    /// `g[59] ≈ g0 + 0.9795 · (g_s − g0)`. We verify the closed-form value
    /// matches the integrator running over a unit-magnitude synthesis input
    /// after a pass-through formant + tilt (which here we approximate by
    /// reading the AGC state directly).
    #[test]
    fn post_filter_agc_leaky_integrator_matches_closed_form() {
        // Build a synthesis signal large enough that `e_in > 0`, then a
        // post-formant/tilt output of half the amplitude so `g_s ≈ 2`.
        // We can't easily decouple all four stages, so instead: drive the
        // raw AGC update for N samples by hand and check the leaky-integrator
        // closed form matches the simulated trajectory.
        let alpha = POSTFILTER_AGC_ALPHA;
        let g_s = 2.0f32;
        let mut g = 1.0f32; // start from unity (init)
        for _ in 0..SUBFRAME_SIZE {
            g = (1.0 - alpha) * g + alpha * g_s;
        }
        let one_minus_alpha_n = (1.0f32 - alpha).powi(SUBFRAME_SIZE as i32);
        let expected = 1.0 + (g_s - 1.0) * (1.0 - one_minus_alpha_n);
        assert!(
            (g - expected).abs() < 1e-5,
            "leaky-integrator simulation {g} != closed form {expected}"
        );
    }

    /// Erased-frame concealment: a SID / Untransmitted frame must produce
    /// a full 240-sample frame that decays with run length. The first
    /// erasure keeps the gain close to the last good frame's; by the 5th
    /// G.723.1 §3.10.2 attenuation schedule: the regenerated
    /// excitation is attenuated 2.5 dB per consecutive erased frame and
    /// muted after `ERASURE_MUTE_AFTER_FRAMES` (3) frames. Verifies the
    /// erased-run counter advances and that any frame past the mute
    /// threshold emits exact silence.
    #[test]
    fn decode_erased_attenuation_schedule_matches_spec() {
        let mut st = SynthesisState::new();
        st.pf_last_gain_adapt = 0.5;
        st.pf_last_gain_fixed = 0.2;
        st.pf_last_gain_unvoiced = 0.2;
        // Seed the excitation history so there's something to propagate.
        for i in 0..st.exc_history.len() {
            st.exc_history[i] = ((i as f32 * 0.17).sin()) * 0.1;
        }
        let e1 = st.decode_erased();
        assert_eq!(e1.len(), FRAME_SIZE_SAMPLES);
        assert_eq!(st.pf_erased_run, 1);

        // Run frames 2..=ERASURE_MUTE_AFTER_FRAMES and one past.
        for _ in 0..ERASURE_MUTE_AFTER_FRAMES {
            let _ = st.decode_erased();
        }
        // First frame past the mute threshold must be exact silence.
        let muted = st.decode_erased();
        assert!(
            muted.iter().all(|&s| s == 0),
            "expected silence after mute threshold"
        );
    }

    /// G.723.1 §3.11: the decoder cold-starts its previous LSP vector at the
    /// long-term DC vector p_DC (in the synthesiser cosine domain), not an
    /// evenly-spaced placeholder. The resulting cosines must equal
    /// `lsp_dc_cosines()` exactly and be a strictly-ordered LSP set
    /// (strictly-decreasing cosines / strictly-increasing frequencies,
    /// inside the open unit interval).
    #[test]
    fn cold_start_prev_lsp_is_dc_vector() {
        let st = SynthesisState::new();
        let dc = crate::tables::lsp_dc_cosines();
        assert_eq!(st.prev_lsp, dc, "cold-start prev_lsp must equal p_DC");
        for k in 0..LPC_ORDER {
            assert!(
                st.prev_lsp[k] > -1.0 && st.prev_lsp[k] < 1.0,
                "DC cosine {k} out of (-1, 1): {}",
                st.prev_lsp[k]
            );
            if k > 0 {
                assert!(
                    st.prev_lsp[k] < st.prev_lsp[k - 1],
                    "DC cosines must be strictly decreasing at {k}"
                );
            }
        }
    }

    /// G.723.1 §3.10.1: erasure LSP extrapolation leaks the previous LSP
    /// toward the DC vector at rate `1 − b_e = 9/32` per frame. Each
    /// extrapolated angular frequency must land exactly on the convex
    /// combination `b_e·ω_prev + (1 − b_e)·ω_DC`, and a previous vector that
    /// already equals the DC vector must be a fixed point.
    #[test]
    fn erasure_lsp_extrapolation_leaks_toward_dc() {
        // A prev LSP deliberately offset from DC (each ω shifted +0.2 rad,
        // re-clamped into (0, π) so it stays a valid ordered set).
        let dc = crate::tables::lsp_dc_cosines();
        let mut prev = [0.0f32; LPC_ORDER];
        for k in 0..LPC_ORDER {
            let omega_dc = (dc[k] as f32).clamp(-1.0, 1.0).acos();
            let shifted = (omega_dc + 0.2).min(std::f32::consts::PI - 1e-3);
            prev[k] = shifted.cos();
        }

        let out = extrapolate_lsp_toward_dc(&prev, LSP_PREDICTOR_BE);
        for k in 0..LPC_ORDER {
            let omega_prev = prev[k].clamp(-1.0, 1.0).acos();
            let omega_dc = crate::tables::lsp_dc_omega(k);
            let expected = (LSP_PREDICTOR_BE * (omega_prev - omega_dc) + omega_dc).cos();
            assert!(
                (out[k] - expected).abs() < 1e-5,
                "dim {k}: got {}, expected {expected}",
                out[k]
            );
            // The extrapolated frequency must sit strictly between prev and
            // DC (a true leak toward DC, never overshoot).
            let omega_out = out[k].clamp(-1.0, 1.0).acos();
            let lo = omega_prev.min(omega_dc);
            let hi = omega_prev.max(omega_dc);
            assert!(
                omega_out >= lo - 1e-4 && omega_out <= hi + 1e-4,
                "dim {k}: leaked ω {omega_out} not between {lo} and {hi}"
            );
            assert!(
                (omega_out - omega_dc).abs() <= (omega_prev - omega_dc).abs() + 1e-4,
                "dim {k}: leak moved away from DC"
            );
        }

        // Fixed point: prev == DC ⇒ output == DC.
        let dc_cos = crate::tables::lsp_dc_cosines();
        let fixed = extrapolate_lsp_toward_dc(&dc_cos, LSP_PREDICTOR_BE);
        for k in 0..LPC_ORDER {
            assert!(
                (fixed[k] - dc_cos[k]).abs() < 1e-5,
                "DC vector must be a fixed point of the erasure leak at {k}"
            );
        }
    }

    /// G.723.1 §3.10.1 across a sustained erasure run: because the concealed
    /// LSP is persisted as the previous vector, each successive erased frame
    /// pulls the spectral envelope monotonically closer to the DC vector.
    #[test]
    fn sustained_erasure_relaxes_lsp_toward_dc() {
        let omega_dc: Vec<f32> = (0..LPC_ORDER).map(crate::tables::lsp_dc_omega).collect();

        let mut st = SynthesisState::new();
        // Start the previous LSP well away from DC so there is room to leak.
        let mut prev = [0.0f32; LPC_ORDER];
        for k in 0..LPC_ORDER {
            let shifted = (omega_dc[k] + 0.25).min(std::f32::consts::PI - 1e-2);
            prev[k] = shifted.cos();
        }
        st.prev_lsp = prev;
        st.pf_last_gain_adapt = 0.3;
        st.pf_last_gain_unvoiced = 0.1;
        for i in 0..st.exc_history.len() {
            st.exc_history[i] = (i as f32 * 0.13).sin() * 0.05;
        }

        let dist = |lsp: &[f32; LPC_ORDER]| -> f32 {
            let mut acc = 0.0;
            for k in 0..LPC_ORDER {
                let w = lsp[k].clamp(-1.0, 1.0).acos();
                acc += (w - omega_dc[k]).powi(2);
            }
            acc.sqrt()
        };

        let mut last = dist(&st.prev_lsp);
        // Two leaks within the mute window (run counts 1 and 2). Each must
        // strictly reduce the distance to DC.
        for _ in 0..2 {
            let _ = st.decode_erased();
            let now = dist(&st.prev_lsp);
            assert!(
                now < last - 1e-4,
                "sustained erasure must move LSP closer to DC: {now} !< {last}"
            );
            last = now;
        }
    }

    /// G.723.1 §3.10.2 voiced/unvoiced classifier: a strongly periodic
    /// trailing window should be reported as voiced with a lag close to
    /// the seeded pitch period; a broadband-random trailing window
    /// should be reported as unvoiced.
    #[test]
    fn erasure_classifier_distinguishes_voiced_and_unvoiced() {
        // Voiced: pure 100 Hz sinusoid at 8 kHz ⇒ period ≈ 80 samples.
        let mut st = SynthesisState::new();
        st.pf_last_lag2 = 80;
        for i in 0..st.pf_pcm_hist.len() {
            let t = i as f32;
            st.pf_pcm_hist[i] = (2.0 * std::f32::consts::PI * t / 80.0).sin();
        }
        let (voiced, lag) = st.classify_erasure_voicing();
        assert!(voiced, "pure sinusoid should classify voiced");
        assert!(
            (lag - 80).abs() <= ERASURE_CLASSIFIER_LAG_RADIUS,
            "expected lag near 80, got {lag}"
        );

        // Unvoiced: deterministic LCG broadband noise.
        let mut st2 = SynthesisState::new();
        st2.pf_last_lag2 = 80;
        let mut lcg: u32 = 0x1234_5678;
        for s in st2.pf_pcm_hist.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = ((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0;
        }
        let (voiced2, _) = st2.classify_erasure_voicing();
        assert!(
            !voiced2,
            "broadband noise should classify unvoiced (gain {})",
            "n/a"
        );

        // Empty / zero history must return unvoiced without panicking.
        let st3 = SynthesisState::new();
        let (voiced3, _) = st3.classify_erasure_voicing();
        assert!(!voiced3, "silent history must classify unvoiced");
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
        let out = ltp_sf(&st, &syn, 40, Rate::High);
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
        let out = ltp_sf(&st, &syn, 40, Rate::High);
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
        let out_high = ltp_sf(&st, &syn, period, Rate::High);
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
        let out_low = ltp_sf(&st2, &syn, period, Rate::Low);
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
    /// signal, reading across the subframe boundary into the whole-frame
    /// synthesis buffer (§3.6 / trace §8) — for `M_f` near the period the
    /// forward read at the subframe tail lands in the *next* subframe.
    #[test]
    fn ltp_forward_search_locks_on_period() {
        let st = SynthesisState::new();
        let period: i32 = 40;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        // Whole-frame sinusoid so the forward reach past sample 60 reads a
        // consistent continuation rather than zeros.
        let mut raw = [0.0f32; FRAME_SIZE_SAMPLES];
        for (n, s) in raw.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / period as f32).sin();
        }
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        syn.copy_from_slice(&raw[..SUBFRAME_SIZE]);
        // Search a window straddling the true period, starting at frame
        // offset 0 (the first subframe).
        let fwd = ForwardCtx {
            raw_frame: &raw,
            sf_start: 0,
        };
        let (best_m, c, d) = st.ltp_search_forward(&syn, fwd, 37, 43);
        assert_eq!(best_m, 40, "forward search should pick the exact period 40");
        assert!(c > 0.0);
        assert!(d > 0.0);
    }

    /// The whole-frame forward reach (§3.6 / trace §8) must actually read
    /// the successor subframe: a subframe whose tail samples differ only in
    /// the *next* subframe of `raw_frame` must produce a different forward
    /// correlation than the old window-shrinkage (zero past sample 60)
    /// would. We compare a frame whose successor continues the pattern
    /// against one whose successor is zeroed; with `M_f` large enough that
    /// `n + M_f >= 60` for some `n`, the two must differ.
    #[test]
    fn ltp_forward_reach_reads_next_subframe() {
        let st = SynthesisState::new();
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let period = 40.0f32;
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for (n, s) in syn.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / period).sin();
        }

        // Continuation present in the second subframe.
        let mut raw_full = [0.0f32; FRAME_SIZE_SAMPLES];
        for (n, s) in raw_full.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / period).sin();
        }
        // Successor zeroed (old window-shrinkage equivalent).
        let mut raw_trunc = [0.0f32; FRAME_SIZE_SAMPLES];
        raw_trunc[..SUBFRAME_SIZE].copy_from_slice(&syn);

        // M_f = 38..42 forces n + M_f >= 60 for the tail samples.
        let fwd_full = ForwardCtx {
            raw_frame: &raw_full,
            sf_start: 0,
        };
        let fwd_trunc = ForwardCtx {
            raw_frame: &raw_trunc,
            sf_start: 0,
        };
        let (_, c_full, d_full) = st.ltp_search_forward(&syn, fwd_full, 38, 42);
        let (_, c_trunc, d_trunc) = st.ltp_search_forward(&syn, fwd_trunc, 38, 42);
        assert!(
            (c_full - c_trunc).abs() > 1e-3 || (d_full - d_trunc).abs() > 1e-3,
            "whole-frame forward reach must read the successor subframe \
             (c_full={c_full}, c_trunc={c_trunc}, d_full={d_full}, d_trunc={d_trunc})"
        );
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
        let prev_lsp = st.prev_lsp;
        let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (n, s) in pcm.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / 50.0).sin() * 0.3;
        }
        let lags = [50, 51, 48, 49];
        st.apply_post_filter(&prev_lsp, &lsp_q, &lags, Rate::High, &mut pcm);
        for v in pcm.iter() {
            assert!(v.is_finite());
        }
    }

    /// The formant postfilter's per-subframe LPC must come from the §2.7
    /// (eq. 8) interpolation between the *previous frame's* LSP and the
    /// current frame's, not a frame-constant LSP.  With distinct prev/cur
    /// LSP vectors the early subframes (weighted heavily toward `prev`)
    /// must produce a measurably different postfiltered signal than when
    /// `prev == cur`; the last subframe (weight 0/1 on `prev`) must be
    /// (near-)identical between the two runs, since its interpolation
    /// ignores `prev` entirely.
    #[test]
    fn post_filter_uses_interpolated_lpc_across_the_frame() {
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let make_pcm = || {
            let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
            for (n, s) in pcm.iter_mut().enumerate() {
                *s = (two_pi * (n as f32) / 50.0).sin() * 0.3;
            }
            pcm
        };
        // Current-frame LSP (decoder default) and a deliberately different
        // previous-frame LSP (omegas shifted up), both strictly ordered.
        let cur = SynthesisState::new().prev_lsp;
        let mut prev = [0.0f32; LPC_ORDER];
        let step = std::f32::consts::PI / (LPC_ORDER as f32 + 1.0);
        for k in 0..LPC_ORDER {
            // Shift each omega by +0.15 rad → a distinct but ordered LSP.
            prev[k] = ((k as f32 + 1.0) * step + 0.15).min(3.05).cos();
        }
        let lags = [50, 51, 48, 49];

        // Run A: postfilter with the true previous-frame LSP.
        let mut st_a = SynthesisState::new();
        let mut pcm_a = make_pcm();
        st_a.apply_post_filter(&prev, &cur, &lags, Rate::High, &mut pcm_a);

        // Run B: postfilter with prev == cur (the old degenerate path).
        let mut st_b = SynthesisState::new();
        let mut pcm_b = make_pcm();
        st_b.apply_post_filter(&cur, &cur, &lags, Rate::High, &mut pcm_b);

        // Subframe 0 (weight 0.75 on prev) must differ when prev != cur.
        let sub0_diff: f32 = (0..SUBFRAME_SIZE)
            .map(|n| (pcm_a[n] - pcm_b[n]).abs())
            .sum();
        assert!(
            sub0_diff > 1e-4,
            "subframe 0 must reflect the previous-frame LSP via §2.7 interpolation (diff={sub0_diff})"
        );

        // Subframe 3 (weight 0/1 on prev) ignores prev, so the two runs
        // must agree there — modulo the carried postfilter memory, which
        // we bound loosely relative to the subframe-0 divergence.
        let last = 3 * SUBFRAME_SIZE;
        let sub3_diff: f32 = (0..SUBFRAME_SIZE)
            .map(|n| (pcm_a[last + n] - pcm_b[last + n]).abs())
            .sum();
        assert!(
            sub3_diff < sub0_diff,
            "subframe 3 ignores prev (weight 0/1) so it must diverge less than subframe 0 \
             (sub3={sub3_diff}, sub0={sub0_diff})"
        );
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

    /// `postfilter_expand` must scale `a[k+1]` by the spec's Q15
    /// §2.18 PostFilt weighting tables exactly (`PostFilt[k] / 2¹⁵`),
    /// leaving the `a[0] = 1` gain tap untouched.
    #[test]
    fn postfilter_expand_uses_q15_tables_verbatim() {
        // Arbitrary non-trivial LPC vector (a[0] is the gain tap).
        let mut a = [1.0f32; LPC_ORDER + 1];
        for (k, v) in a.iter_mut().enumerate() {
            *v = 1.0 - 0.05 * (k as f32);
        }
        let zero = crate::spec_tables::POSTFILTER_ZERO_Q15;
        let pole = crate::spec_tables::POSTFILTER_POLE_Q15;

        let num = postfilter_expand(&a, &zero);
        let den = postfilter_expand(&a, &pole);

        // Gain tap is left as-is in both.
        assert_eq!(num[0], a[0]);
        assert_eq!(den[0], a[0]);
        for k in 0..LPC_ORDER {
            let want_num = a[k + 1] * (zero[k] as f32 / 32768.0);
            let want_den = a[k + 1] * (pole[k] as f32 / 32768.0);
            assert!((num[k + 1] - want_num).abs() < 1e-9);
            assert!((den[k + 1] - want_den).abs() < 1e-9);
        }
    }

    /// The spec Q15 weighting tables are the fixed-point powers
    /// `γ^(i+1)` of γ₁ = 0.65 (zeros) / γ₂ = 0.75 (poles): each entry equals
    /// `round(γ^(i+1) · 2¹⁵)` (round half away from zero, as `f64::round`
    /// does), and the sequence is strictly decreasing — the property a
    /// bandwidth-expansion weighting must satisfy. Pinning the tables to
    /// the closed-form powers guards against an accidental transcription
    /// drift while documenting that `postfilter_expand` applies the exact
    /// §2.18 weighting rather than a repeatedly-multiplied float `gamma^i`
    /// (which accumulates rounding error across the 10 taps).
    #[test]
    fn postfilter_q15_tables_are_decreasing_gamma_powers() {
        let zero = crate::spec_tables::POSTFILTER_ZERO_Q15;
        let pole = crate::spec_tables::POSTFILTER_POLE_Q15;
        for k in 1..LPC_ORDER {
            assert!(zero[k] < zero[k - 1], "zero table must be decreasing");
            assert!(pole[k] < pole[k - 1], "pole table must be decreasing");
        }
        for (k, &w) in zero.iter().enumerate() {
            let want = (0.65f64.powi(k as i32 + 1) * 32768.0).round() as i16;
            assert_eq!(w, want, "zero[{k}] must be round(0.65^(k+1)*2^15)");
        }
        for (k, &w) in pole.iter().enumerate() {
            let want = (0.75f64.powi(k as i32 + 1) * 32768.0).round() as i16;
            assert_eq!(w, want, "pole[{k}] must be round(0.75^(k+1)*2^15)");
        }
    }
}
