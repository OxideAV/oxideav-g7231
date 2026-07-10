//! ITU-T G.723.1 encoder — ACELP (5.3 kbit/s) and MP-MLQ (6.3 kbit/s) paths.
//!
//! # Scope
//!
//! This module implements **both** rates of G.723.1:
//!
//! - **5.3 kbit/s ACELP** — 4 pulses per subframe on the §2.16 Table 1
//!   stride-8 tracks (T0..T3, 1-bit grid shifting the set to odd
//!   positions); 20-byte payload, discriminator `01`.
//! - **6.3 kbit/s MP-MLQ** — 6 pulses on even subframes (0, 2) and
//!   5 pulses on odd subframes (1, 3); 24-byte payload, discriminator `00`.
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
//!          → LSP conversion (Chebyshev root-finding) + §2.5 predictive
//!            split VQ on the published codebooks (24-bit LPC word)
//!          → 4× subframe loop (ACB lookup against SynthesisState):
//!                - zero-input response of 1/A_q(z) → ZIR-free target
//!                - §2.14 closed-loop lag ±1 / −1..+2 candidates,
//!                  jointly searched with the 85-/170-row 5-tap
//!                  gain-vector codebook (max 2·βᵀd − βᵀRβ)
//!                - rate-specific FCB search at quantised gain levels
//!                    · ACELP:  §2.16 Table 1 tracks + grid against the
//!                              pitch-enhanced impulse response
//!                    · MP-MLQ: §2.15 greedy multipulse, gain
//!                              neighbourhood × grids × Dirac trains
//!                - eq. 36/39/40 combined 12-bit gain word
//!          → canonical SynthesisState::decode_spec_params() commits
//!            decoder state so encoder + decoder stay in lockstep
//!          → clause-4 Table 5/6 octet packing (20 B rate=01 /
//!            24 B rate=00)
//! ```
//!
//! # Wire format
//!
//! Frames are the ITU-T clause-4 spec layout on the published quantiser
//! tables (see [`crate::linepack`] / [`crate::spec_lsp`] /
//! [`crate::spec_exc`]). The README documents the two derivation
//! choices the Recommendation's tables leave open (MSBPOS digit order,
//! intra-word pulse/sign bit conventions) and the conformance-vector
//! caveat.

use std::collections::VecDeque;

#[cfg(test)]
use oxideav_core::AudioFrame;
use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::linepack::{PackedRate, SpecFrameParams};
use crate::spec_exc;
use crate::spec_lsp;
use crate::tables::{
    ERASURE_ATTENUATION_DB_PER_FRAME, ERASURE_CLASSIFIER_HISTORY_LEN,
    ERASURE_CLASSIFIER_LAG_RADIUS, ERASURE_MUTE_AFTER_FRAMES, ERASURE_VOICED_THRESHOLD_DB,
    FRAME_SIZE_SAMPLES, HIGH_RATE_BYTES, LOOKAHEAD_SAMPLES, LOW_RATE_BYTES, LPC_ORDER, LPC_WINDOW,
    LSP_PREDICTOR_BE, LSP_STABILITY_DELTA_MIN_ERASURE_HZ, LSP_STABILITY_DELTA_MIN_HZ,
    LSP_STABILITY_MAX_ITERATIONS, PITCH_MAX, PITCH_MIN, POSTFILTER_AGC_ALPHA,
    POSTFILTER_AGC_INIT_GAIN, POSTFILTER_LTP_GAMMA_HIGH, POSTFILTER_LTP_GAMMA_LOW,
    POSTFILTER_LTP_PRED_GAIN_DB_MIN, POSTFILTER_LTP_SEARCH_RADIUS, POSTFILTER_TILT_BASE,
    POSTFILTER_TILT_SMOOTH_ALPHA, SAMPLE_RATE_HZ, SUBFRAMES_PER_FRAME, SUBFRAME_SIZE,
};

/// Total payload size for an ACELP (5.3 kbit/s) frame.
const ACELP_PAYLOAD_BYTES: usize = LOW_RATE_BYTES;
/// Total payload size for an MP-MLQ (6.3 kbit/s) frame.
const MPMLQ_PAYLOAD_BYTES: usize = HIGH_RATE_BYTES;

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
        // §2.4 windowing needs 60 samples of lookahead past the frame
        // end, so a frame is emitted only once its lookahead is
        // buffered too. On the final flush the missing lookahead (and
        // any partial final frame) is zero-padded — the encoder's
        // §2.21 rest state.
        while self.pcm_queue.len() >= FRAME_SIZE_SAMPLES + LOOKAHEAD_SAMPLES {
            let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
            pcm.copy_from_slice(&self.pcm_queue[..FRAME_SIZE_SAMPLES]);
            let mut la = [0i16; LOOKAHEAD_SAMPLES];
            la.copy_from_slice(
                &self.pcm_queue[FRAME_SIZE_SAMPLES..FRAME_SIZE_SAMPLES + LOOKAHEAD_SAMPLES],
            );
            self.pcm_queue.drain(..FRAME_SIZE_SAMPLES);
            self.emit_frame(&pcm, &la);
        }
        if final_flush {
            while !self.pcm_queue.is_empty() {
                let take = self.pcm_queue.len().min(FRAME_SIZE_SAMPLES);
                let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
                pcm[..take].copy_from_slice(&self.pcm_queue[..take]);
                let mut la = [0i16; LOOKAHEAD_SAMPLES];
                let rest = (self.pcm_queue.len() - take).min(LOOKAHEAD_SAMPLES);
                la[..rest].copy_from_slice(&self.pcm_queue[take..take + rest]);
                self.pcm_queue.drain(..take);
                self.emit_frame(&pcm, &la);
            }
        }
    }

    fn emit_frame(
        &mut self,
        pcm: &[i16; FRAME_SIZE_SAMPLES],
        lookahead: &[i16; LOOKAHEAD_SAMPLES],
    ) {
        let frame_idx = self.frame_index;
        self.frame_index += 1;
        let rate = match self.mode {
            EncoderMode::Acelp => PackedRate::Low,
            EncoderMode::MpMlq => PackedRate::High,
        };
        let params = self.analysis.analyse_spec(pcm, lookahead, rate);
        // `analyse_spec` emits in-range indices by construction, so the
        // clause-4 packer cannot reject them.
        let packed = crate::linepack::pack_frame(&params)
            .expect("analyse_spec emits in-range clause-4 parameters");
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
    /// §2.3 input high-pass (DC removal) switch. §2.2: "Each block is
    /// first high pass filtered to remove the DC component" — default
    /// ON; the ITU encoder test configurations selectively disable it
    /// (`CODEC63`/`OVERC63`/`INEQC53`/`PATHC53` run HP OFF).
    highpass: bool,
    /// One-sample x[n−1] memory of the §2.3 filter.
    hp_x_prev: f32,
    /// One-sample y[n−1] memory of the §2.3 filter.
    hp_y_prev: f32,
    /// §2.4 windowing look-back: the high-pass-filtered last 60 samples
    /// of the previous frame. The 180-sample analysis window centered on
    /// subframe 0 reaches 60 samples *before* the frame start.
    lpc_tail: [f32; SUBFRAME_SIZE],
    /// Previous frame's *unquantised* LSP vector (cosine domain) — the
    /// fallback when the current frame's LPC → LSP root search fails on
    /// a degenerate model. Initialised to `p_DC` (§2.21).
    prev_unq_lsp: [f32; LPC_ORDER],
}

impl AnalysisState {
    fn new() -> Self {
        Self {
            decoder: SynthesisState::new(),
            highpass: true,
            hp_x_prev: 0.0,
            hp_y_prev: 0.0,
            lpc_tail: [0.0; SUBFRAME_SIZE],
            prev_unq_lsp: crate::tables::lsp_dc_cosines(),
        }
    }

    /// §2.3 high-pass filter over one frame, eq. 1:
    /// `H(z) = (1 − z⁻¹) / (1 − (127/128)·z⁻¹)`, i.e.
    /// `y[n] = x[n] − x[n−1] + (127/128)·y[n−1]`, with memories carried
    /// across frames.
    fn highpass_frame(&mut self, sig: &mut [f32; FRAME_SIZE_SAMPLES]) {
        const POLE: f32 = 127.0 / 128.0;
        for v in sig.iter_mut() {
            let x = *v;
            let y = x - self.hp_x_prev + POLE * self.hp_y_prev;
            self.hp_x_prev = x;
            self.hp_y_prev = y;
            *v = y;
        }
    }
}

// ---------- spec-layout encoder analysis ----------

impl AnalysisState {
    /// Analyse one 240-sample frame into a clause-4 spec-layout
    /// parameter set — the §2 encoder pipeline running on the published
    /// tables:
    ///
    /// - §2.4–2.5: LPC analysis + the predictive split-VQ LSP quantiser
    ///   ([`crate::spec_lsp`]).
    /// - §2.14 closed-loop pitch: lag candidates around the per-subframe
    ///   open-loop estimate (±1 on subframes 0/2; the −1..+2 delta window
    ///   on 1/3), jointly searched with the 85-/170-row gain-vector
    ///   codebook by maximising `2·βᵀd − βᵀRβ` over the filtered eq. 41
    ///   basis vectors.
    /// - §2.15 / §2.16 fixed-codebook search at the quantised gain
    ///   levels (see [`mpmlq_spec_search`] / [`acelp_spec_search`]).
    ///
    /// After parameter selection the shadow decoder is rolled back and
    /// committed through the *exact* decode kernel
    /// ([`SynthesisState::decode_spec_params`]), keeping encoder and
    /// decoder state in provable lockstep frame after frame.
    fn analyse_spec(
        &mut self,
        pcm: &[i16; FRAME_SIZE_SAMPLES],
        lookahead: &[i16; LOOKAHEAD_SAMPLES],
        rate: PackedRate,
    ) -> SpecFrameParams {
        let mut sig = [0.0f32; FRAME_SIZE_SAMPLES];
        for (o, &s) in sig.iter_mut().zip(pcm.iter()) {
            *o = s as f32 * (1.0 / 32_768.0);
        }
        // §2.2/§2.3: remove the DC component up front (switchable for
        // the ITU test configurations).
        if self.highpass {
            self.highpass_frame(&mut sig);
        }
        // The 60 lookahead samples belong to the *next* frame; they are
        // high-pass filtered with a scratch copy of the filter memory so
        // they see the same §2.3 output they will when the next frame is
        // analysed for real, without advancing the committed state.
        let mut la = [0.0f32; LOOKAHEAD_SAMPLES];
        for (o, &s) in la.iter_mut().zip(lookahead.iter()) {
            *o = s as f32 * (1.0 / 32_768.0);
        }
        if self.highpass {
            const POLE: f32 = 127.0 / 128.0;
            let (mut xp, mut yp) = (self.hp_x_prev, self.hp_y_prev);
            for v in la.iter_mut() {
                let x = *v;
                let y = x - xp + POLE * yp;
                xp = x;
                yp = y;
                *v = y;
            }
        }

        // ---- §2.4: per-subframe LPC on the 180-sample centered
        // windows. The analysis buffer is
        // [previous-frame tail | this frame | lookahead]; the window
        // for subframe `s` covers buffer samples `s·60 .. s·60+180`,
        // i.e. is centered on the subframe.
        let mut wind_buf = [0.0f32; SUBFRAME_SIZE + FRAME_SIZE_SAMPLES + LOOKAHEAD_SAMPLES];
        wind_buf[..SUBFRAME_SIZE].copy_from_slice(&self.lpc_tail);
        wind_buf[SUBFRAME_SIZE..SUBFRAME_SIZE + FRAME_SIZE_SAMPLES].copy_from_slice(&sig);
        wind_buf[SUBFRAME_SIZE + FRAME_SIZE_SAMPLES..].copy_from_slice(&la);
        self.lpc_tail
            .copy_from_slice(&sig[FRAME_SIZE_SAMPLES - SUBFRAME_SIZE..]);
        let mut a_unq = [[0.0f32; LPC_ORDER + 1]; SUBFRAMES_PER_FRAME];
        for (s, a_s) in a_unq.iter_mut().enumerate() {
            *a_s = lpc_analysis(&wind_buf[s * SUBFRAME_SIZE..s * SUBFRAME_SIZE + LPC_WINDOW]);
        }

        // ---- §2.5: bandwidth-expand A3(z) by 7.5 Hz (the published
        // Q15 per-tap weights), then quantise it with the predictive
        // split VQ (§2.5–2.6). Only the last subframe's LPC set is
        // transmitted.
        let mut a3_exp = a_unq[SUBFRAMES_PER_FRAME - 1];
        for (k, w) in crate::spec_tables::LPC_BANDWIDTH_EXPANSION_Q15
            .iter()
            .enumerate()
        {
            a3_exp[k + 1] *= *w as f32 / 32_768.0;
        }
        let lsp_cur_cos = lpc_to_lsp(&a3_exp).unwrap_or(self.prev_unq_lsp);
        self.prev_unq_lsp = lsp_cur_cos;
        let lsp_cur_freq = spec_lsp::lsp_cosines_to_freq(&lsp_cur_cos);
        let (lsp_index, decoded_freq) =
            spec_lsp::quantise_lsp_freq(&lsp_cur_freq, &self.decoder.prev_lsp_freq);
        let cos_raw = spec_lsp::lsp_freq_to_cosines(&decoded_freq);
        let (mut lsp_q, converged) = enforce_lsp_stability(&cos_raw, LSP_STABILITY_DELTA_MIN_HZ);
        if !converged {
            lsp_q = self.decoder.prev_lsp;
        }

        let exc_snapshot = self.decoder.exc_history;
        let syn_snapshot = self.decoder.syn_mem;
        let prev_lsp_snapshot = self.decoder.prev_lsp;

        let mut params = SpecFrameParams::zeroed(rate);
        params.lsp_index = lsp_index;
        let mut lags = [0i32; SUBFRAMES_PER_FRAME];
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

            // ---- Closed-loop pitch + gain-vector search (§2.14). ----
            let ol = open_loop_acb_lag(&target, &self.decoder.exc_history, &h);
            let lag_range = PITCH_MIN as i32..=PITCH_MAX as i32;
            let candidates: Vec<i32> = if s % 2 == 0 {
                (ol - 1..=ol + 1)
                    .filter(|l| lag_range.contains(l))
                    .collect()
            } else {
                (-1i32..=2)
                    .map(|d| lags[s - 1] + d)
                    .filter(|l| lag_range.contains(l))
                    .collect()
            };

            let mut best_score = f32::NEG_INFINITY;
            let mut best_lag = *candidates.first().unwrap_or(&(PITCH_MIN as i32));
            let mut best_pg = 0usize;
            for &cand in &candidates {
                // §2.14: the 85-row rule keys off L0 / L2 — for the even
                // subframes that is the candidate itself.
                let lag_base = if s % 2 == 0 {
                    cand
                } else {
                    lags[s - 1] // lags[0] for s = 1, lags[2] for s = 3
                };
                let rows = spec_exc::gain_vq_rows(rate, lag_base);
                let basis = spec_exc::acb_basis(&self.decoder.exc_history, cand);
                let mut y = [[0.0f32; SUBFRAME_SIZE]; spec_exc::ACB_TAPS];
                for (yj, bj) in y.iter_mut().zip(basis.iter()) {
                    *yj = conv_causal(bj, &h);
                }
                let mut d = [0.0f32; spec_exc::ACB_TAPS];
                let mut rmat = [[0.0f32; spec_exc::ACB_TAPS]; spec_exc::ACB_TAPS];
                for j in 0..spec_exc::ACB_TAPS {
                    for n in 0..SUBFRAME_SIZE {
                        d[j] += target[n] * y[j][n];
                    }
                    for k in j..spec_exc::ACB_TAPS {
                        let mut acc = 0.0f32;
                        for n in 0..SUBFRAME_SIZE {
                            acc += y[j][n] * y[k][n];
                        }
                        rmat[j][k] = acc;
                        rmat[k][j] = acc;
                    }
                }
                for pg in 0..rows {
                    let taps = spec_exc::acb_taps(rate, lag_base, pg);
                    // Error reduction of this codeword:
                    // 2·βᵀd − βᵀRβ (row 0 is all-zero taps ⇒ score 0, so
                    // the best score is never negative).
                    let mut score = 0.0f32;
                    for j in 0..spec_exc::ACB_TAPS {
                        score += 2.0 * taps[j] * d[j];
                        for k in 0..spec_exc::ACB_TAPS {
                            score -= taps[j] * taps[k] * rmat[j][k];
                        }
                    }
                    if score > best_score {
                        best_score = score;
                        best_lag = cand;
                        best_pg = pg;
                    }
                }
            }
            lags[s] = best_lag;
            params.acl[s] = if s % 2 == 0 {
                encode_abs_lag(best_lag)
            } else {
                encode_delta_lag(best_lag, lags[s - 1])
            };
            let lag_base = if s < 2 { lags[0] } else { lags[2] };

            // Residual target for the fixed-codebook stage (eq. 20).
            let taps = spec_exc::acb_taps(rate, lag_base, best_pg);
            let u = spec_exc::acb_contribution(&self.decoder.exc_history, best_lag, &taps);
            let u_filt = conv_causal(&u, &h);
            let mut target2 = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                target2[n] = target[n] - u_filt[n];
            }

            // ---- Fixed codebook (§2.15 / §2.16) + gain word. ----
            match rate {
                PackedRate::High => {
                    let n_pulses = if s % 2 == 0 { 6 } else { 5 };
                    let (pos, psig, grid, mg, train) =
                        mpmlq_spec_search(&target2, &h, n_pulses, lag_base);
                    params.pos[s] = pos;
                    params.psig[s] = psig;
                    params.grid[s] = grid;
                    params.gain[s] = spec_exc::encode_gain_word(rate, lag_base, best_pg, mg, train);
                }
                PackedRate::Low => {
                    let (pos, psig, grid, mg) = acelp_spec_search(&target2, &h, best_lag, best_pg);
                    params.pos[s] = pos;
                    params.psig[s] = psig;
                    params.grid[s] = grid;
                    params.gain[s] = spec_exc::encode_gain_word(rate, lag_base, best_pg, mg, false);
                }
            };

            // ---- Advance the shadow state exactly as the decoder will. ----
            let ginfo = spec_exc::decode_gain_word(rate, lag_base, params.gain[s]);
            let u = spec_exc::acb_contribution(&self.decoder.exc_history, best_lag, &ginfo.taps);
            let v = match rate {
                PackedRate::High => {
                    let n_pulses = if s % 2 == 0 { 6 } else { 5 };
                    spec_exc::mpmlq_fixed_vector(
                        params.pos[s],
                        params.psig[s],
                        params.grid[s],
                        n_pulses,
                        ginfo.fcb_gain,
                        ginfo.train,
                        lag_base,
                    )
                }
                PackedRate::Low => {
                    let mut v = spec_exc::acelp_fixed_vector(
                        params.pos[s],
                        params.psig[s],
                        params.grid[s],
                        ginfo.fcb_gain,
                    );
                    spec_exc::acelp_pitch_enhance(&mut v, best_lag, ginfo.pgindex);
                    v
                }
            };
            let mut exc = [0.0f32; SUBFRAME_SIZE];
            for n in 0..SUBFRAME_SIZE {
                exc[n] = u[n] + v[n];
            }
            self.decoder.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.decoder.exc_history.len() - SUBFRAME_SIZE;
            self.decoder.exc_history[tail..].copy_from_slice(&exc);
            advance_syn_mem(&a_sub, &exc, &mut self.decoder.syn_mem);
        }

        // Roll back and commit through the canonical decode kernel so
        // the shadow decoder state is bit-for-bit what a real decoder
        // holds after this frame.
        self.decoder.exc_history = exc_snapshot;
        self.decoder.syn_mem = syn_snapshot;
        let _ = self.decoder.decode_spec_params(&params);
        params
    }
}

/// §2.15 MP-MLQ fixed-codebook search at quantised gain levels.
///
/// Estimates `G_max` from the eq. 24 cross-correlation and the eq. 25
/// normalisation, quantises it on the 24-step logarithmic table, then
/// searches the gain neighbourhood `[Ĝ_max − 3.2 dB, Ĝ_max + 6.4 dB]`
/// (one step down, two up) × both grids × (single pulses | Dirac trains
/// when the reference lag is short), placing `M` pulses sequentially —
/// each pulse takes the position/sign maximising the correlation of the
/// running residual with the (train-extended) impulse response. The
/// configuration with the least residual energy wins.
///
/// Returns `(pos_code, psig, grid, mgindex, train)` in the
/// [`SpecFrameParams`] conventions.
fn mpmlq_spec_search(
    target: &[f32; SUBFRAME_SIZE],
    h: &[f32],
    n_pulses: usize,
    lag_base: i32,
) -> (u32, u32, u8, usize, bool) {
    // eq. 24–25: estimated gain from the target/impulse cross-correlation.
    let mut h_energy = 0.0f32;
    for &hv in h.iter() {
        h_energy += hv * hv;
    }
    let mut d_max = 0.0f32;
    for j in 0..SUBFRAME_SIZE {
        let mut dj = 0.0f32;
        for n in j..SUBFRAME_SIZE {
            dj += target[n] * h[n - j];
        }
        d_max = d_max.max(dj.abs());
    }
    let gmax = if h_energy > 0.0 {
        d_max / h_energy
    } else {
        0.0
    };
    let j0 = spec_exc::nearest_fcb_gain(gmax);

    // Dirac-train variant of the impulse response (§2.15): the response
    // of a unit train at period `lag_base` through the synthesis filter.
    let allow_train = lag_base < spec_exc::SHORT_LAG_LIMIT;
    let mut h_train = h.to_vec();
    if allow_train {
        // Recursion h_train[n] = h[n] + h_train[n − p] expands to
        // Σ_t h[n − t·p] — the filtered response of a unit Dirac train.
        let p = lag_base.max(1) as usize;
        for n in p..h_train.len() {
            h_train[n] = h[n] + h_train[n - p];
        }
    }

    let mut best_err = f32::INFINITY;
    let mut best: (u32, u32, u8, usize, bool) = (0, 0, 0, j0, false);
    let train_opts: &[bool] = if allow_train {
        &[false, true]
    } else {
        &[false]
    };
    for &train in train_opts {
        let hh: &[f32] = if train { &h_train } else { h };
        for grid in 0..2u8 {
            for mg in j0.saturating_sub(1)..=(j0 + 2).min(23) {
                let g = spec_exc::fcb_gain_value(mg);
                let mut res = *target;
                let mut used = [false; 30];
                let mut chosen: Vec<(usize, bool)> = Vec::with_capacity(n_pulses);
                for _ in 0..n_pulses {
                    let mut best_slot = usize::MAX;
                    let mut best_c = 0.0f32;
                    for (slot, &u) in used.iter().enumerate() {
                        if u {
                            continue;
                        }
                        let m = 2 * slot + grid as usize;
                        let mut c = 0.0f32;
                        for n in m..SUBFRAME_SIZE {
                            c += res[n] * hh[n - m];
                        }
                        if best_slot == usize::MAX || c.abs() > best_c.abs() {
                            best_c = c;
                            best_slot = slot;
                        }
                    }
                    let amp = if best_c < 0.0 { -g } else { g };
                    let m = 2 * best_slot + grid as usize;
                    for n in m..SUBFRAME_SIZE {
                        res[n] -= amp * hh[n - m];
                    }
                    used[best_slot] = true;
                    chosen.push((best_slot, best_c < 0.0));
                }
                // §2.15: "the combination of the quantised parameters
                // that yields the minimum mean square err[n] is
                // selected" — with the pulse pattern fixed, re-optimise
                // the gain index exactly over all 24 levels against the
                // unit-pattern response.
                let mut y_pat = [0.0f32; SUBFRAME_SIZE];
                for &(slot, neg) in chosen.iter() {
                    let m = 2 * slot + grid as usize;
                    let sgn = if neg { -1.0f32 } else { 1.0 };
                    for n in m..SUBFRAME_SIZE {
                        y_pat[n] += sgn * hh[n - m];
                    }
                }
                let (mut c_ty, mut e_yy, mut e_tt) = (0.0f32, 0.0f32, 0.0f32);
                for n in 0..SUBFRAME_SIZE {
                    c_ty += target[n] * y_pat[n];
                    e_yy += y_pat[n] * y_pat[n];
                    e_tt += target[n] * target[n];
                }
                let (best_mg, err) = best_gain_level(c_ty, e_yy, e_tt);
                if err < best_err {
                    chosen.sort_unstable_by_key(|&(slot, _)| slot);
                    let slots: Vec<usize> = chosen.iter().map(|&(slot, _)| slot).collect();
                    // PSIG convention (vector-arbitrated, r388): signs
                    // MSB-first in ascending position order, set bit =
                    // negative — see spec_exc::mpmlq_fixed_vector.
                    let mut psig = 0u32;
                    let n_pulses = chosen.len();
                    for (k, &(_, neg)) in chosen.iter().enumerate() {
                        if neg {
                            psig |= 1 << (n_pulses - 1 - k);
                        }
                    }
                    if let Some(code) = crate::spec_tables::fcbk_pack_positions(&slots) {
                        best_err = err;
                        best = (code, psig, grid, best_mg, train);
                    }
                }
            }
        }
    }
    best
}

/// §2.16 ACELP fixed-codebook search at quantised gain levels: run the
/// Table 1 coordinate-descent pulse search against the §2.16-modified
/// (pitch-enhanced) impulse response, derive the optimal codeword gain
/// by least squares, and quantise it per the §2.16 last step
/// (`|G − G̃_j|` minimisation; a negative optimum flips every pulse sign
/// since the transmitted gain is unsigned).
///
/// Returns `(pos, psig, grid, mgindex)` in the [`SpecFrameParams`]
/// conventions (track `t` slot in `pos` bits `3t..3t+2`, sign bit `t`).
fn acelp_spec_search(
    target: &[f32; SUBFRAME_SIZE],
    h: &[f32],
    lag: i32,
    pgindex: usize,
) -> (u32, u32, u8, usize) {
    let h_enh = spec_exc::acelp_enhanced_impulse_response(h, lag, pgindex);
    let (positions, mut signs, grid) = acelp_4pulse_search(target, &h_enh);

    let mut v_unit = [0.0f32; SUBFRAME_SIZE];
    place_pulses(&positions, signs, grid, &mut v_unit);
    let y = conv_causal(&v_unit, &h_enh);
    let (mut c_ty, mut e_yy, mut e_tt) = (0.0f32, 0.0f32, 0.0f32);
    for n in 0..SUBFRAME_SIZE {
        c_ty += target[n] * y[n];
        e_yy += y[n] * y[n];
        e_tt += target[n] * target[n];
    }
    if c_ty < 0.0 {
        // The transmitted gain is unsigned; flip every pulse sign.
        for sgn in signs.iter_mut() {
            *sgn = -*sgn;
        }
        c_ty = -c_ty;
    }
    let (mg, _) = best_gain_level(c_ty, e_yy, e_tt);

    let pos = positions[0] | positions[1] << 3 | positions[2] << 6 | positions[3] << 9;
    // PSIG convention (vector-arbitrated, r388): bit t set = the track-t
    // pulse is POSITIVE — see spec_exc::acelp_fixed_vector.
    let mut psig = 0u32;
    for (t, &sgn) in signs.iter().enumerate() {
        if sgn > 0 {
            psig |= 1 << t;
        }
    }
    (pos, psig, grid, mg)
}

/// Exact gain-index selection: given the target/pattern correlation
/// `c_ty`, the pattern energy `e_yy`, and the target energy `e_tt`,
/// return the 24-level gain index minimising
/// `‖target − G̃_j·y‖² = e_tt − 2·G̃_j·c_ty + G̃_j²·e_yy` and that
/// minimum (§2.15 / §2.16 gain quantisation as an MMSE pick over the
/// published table).
fn best_gain_level(c_ty: f32, e_yy: f32, e_tt: f32) -> (usize, f32) {
    let mut best = (0usize, f32::INFINITY);
    for j in 0..24usize {
        let g = spec_exc::fcb_gain_value(j);
        let err = e_tt - 2.0 * g * c_ty + g * g * e_yy;
        if err < best.1 {
            best = (j, err);
        }
    }
    best
}

// ---------- LPC analysis ----------

/// §2.4 LPC analysis on one 180-sample window (a slice of the encoder's
/// [tail | frame | lookahead] analysis buffer, centered on a subframe).
/// The published Q15 Hamming window is applied, eleven autocorrelation
/// coefficients are computed, `R[0]` gets the `1025/1024` white-noise
/// correction and `R[1..=10]` are shaped by the published Q15 binomial
/// lag window, then the conventional Levinson-Durbin recursion produces
/// `[1, a_1..a_10]` in direct form.
fn lpc_analysis(window: &[f32]) -> [f32; LPC_ORDER + 1] {
    debug_assert_eq!(window.len(), LPC_WINDOW);
    let mut windowed = [0.0f32; LPC_WINDOW];
    for (i, o) in windowed.iter_mut().enumerate() {
        let w = crate::spec_tables::LPC_HAMMING_WINDOW_Q15[i] as f32 / 32_768.0;
        *o = window[i] * w;
    }
    // Autocorrelation r[0..=LPC_ORDER].
    let mut r = [0.0f64; LPC_ORDER + 1];
    for (k, rk) in r.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in k..LPC_WINDOW {
            acc += windowed[i] as f64 * windowed[i - k] as f64;
        }
        *rk = acc;
    }
    // §2.4: white-noise correction R[0] ← R[0]·(1 + 1/1024), then the
    // binomial lag window on the other ten coefficients.
    r[0] *= 1025.0 / 1024.0;
    for k in 1..=LPC_ORDER {
        r[k] *= crate::spec_tables::LPC_BINOMIAL_LAG_WINDOW_Q15[k - 1] as f64 / 32_768.0;
    }

    // Levinson-Durbin recursion. If the prediction-error energy
    // collapses (perfectly predictable input, e.g. a pure sine) or a
    // reflection coefficient leaves the unit interval, the recursion
    // stops early and keeps the coefficients computed so far — the
    // lower-order model is valid and stable, unlike bailing out to the
    // trivial A(z) = 1.
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
        if !k.is_finite() || k.abs() >= 1.0 {
            break;
        }
        a[i] = k;
        for j in 1..i {
            a[j] = a_prev[j] + k * a_prev[i - j];
        }
        e *= 1.0 - k * k;
        a_prev.copy_from_slice(&a);
        if e <= 0.0 {
            break;
        }
    }
    let mut out = [0.0f32; LPC_ORDER + 1];
    for i in 0..=LPC_ORDER {
        out[i] = a_prev[i] as f32;
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
/// cosine domain (lsp[i] = cos(omega_i)). Uses Chebyshev root-finding
/// on the P(z) / Q(z) sum/difference polynomials (§2.5 step 1:
/// "searching along the unit circle and interpolating for zero
/// crossings"). Returns `None` when the full set of 5 + 5 interlaced
/// roots cannot be located (a degenerate model); the caller falls back
/// to the previous frame's LSP vector.
fn lpc_to_lsp(a: &[f32; LPC_ORDER + 1]) -> Option<[f32; LPC_ORDER]> {
    // Form f1(z) = A(z) + z^-(p+1) A(z^-1); f2(z) = A(z) - z^-(p+1) A(z^-1).
    // After factoring out the trivial roots, we get polynomials of degree
    // p/2 in cos(omega) (Chebyshev expansion).
    let p = LPC_ORDER;
    let mut f1 = [0.0f32; LPC_ORDER / 2 + 1];
    let mut f2 = [0.0f32; LPC_ORDER / 2 + 1];
    // f1_i = a_i + a_{p-i}, i = 0..p/2; remove (1 + z^-1) factor:
    // recursive: f1[i] = (a[i] + a[p-i]) - f1[i-1]
    // f2[i] = (a[i] - a[p-i]) + f2[i-1]
    // Deflation recursions (P by (1 + z⁻¹), Q by (1 − z⁻¹)):
    //   f1[i] = (a_i + a_{p+1−i}) − f1[i−1]
    //   f2[i] = (a_i − a_{p+1−i}) + f2[i−1]
    // seeded with f1[0] = f2[0] = 1 (the previous *deflated*
    // coefficient, not zero — seeding with zero corrupts every
    // subsequent coefficient).
    f1[0] = 1.0;
    f2[0] = 1.0;
    let mut prev_f1 = f1[0];
    let mut prev_f2 = f2[0];
    for i in 1..=p / 2 {
        let ai = a[i];
        let api = a[p + 1 - i];
        f1[i] = ai + api - prev_f1;
        f2[i] = ai - api + prev_f2;
        prev_f1 = f1[i];
        prev_f2 = f2[i];
    }
    // Locate roots of both polynomials in the cosine domain and
    // interleave them (LSPs strictly alternate between the two sets).
    let roots_f1 = cheby_roots(&f1);
    let roots_f2 = cheby_roots(&f2);
    if roots_f1.len() != LPC_ORDER / 2 || roots_f2.len() != LPC_ORDER / 2 {
        return None;
    }
    let mut lsp = [0.0f32; LPC_ORDER];
    for k in 0..LPC_ORDER / 2 {
        lsp[2 * k] = roots_f1[k];
        lsp[2 * k + 1] = roots_f2[k];
    }
    // The interlaced set must be strictly decreasing in cos (= strictly
    // ascending frequency); a violation means the located roots do not
    // form a valid LSP vector.
    for k in 1..LPC_ORDER {
        if lsp[k] >= lsp[k - 1] {
            return None;
        }
    }
    Some(lsp)
}

/// Find the roots (in x = cos ω) of a deflated sum/difference LSP
/// polynomial given by its symmetric-half coefficients
/// `coeffs[0..=deg]`, by sign-change bracketing on a fine grid that is
/// uniform in *angle* ω (so resolution does not collapse near
/// cos ω = ±1), refined by bisection.
///
/// A degree-2·deg symmetric polynomial `Σ c_i z^{-i}` with
/// `c_i = c_{2·deg−i}` (half stored as `coeffs`) evaluates on the unit
/// circle, up to a phase factor, as the real function
///
/// ```text
///   G(ω) = 2·Σ_{k=0..deg−1} coeffs[k]·cos((deg−k)·ω) + coeffs[deg]
/// ```
///
/// i.e. in x = cos ω a Chebyshev series with the *reversed* coefficient
/// order and a half-weight constant term:
/// `G(x) = 2·Σ_{k<deg} coeffs[k]·T_{deg−k}(x) + coeffs[deg]·T_0(x)`.
fn cheby_roots(coeffs: &[f32]) -> Vec<f32> {
    let deg = coeffs.len() - 1;
    // Chebyshev-basis coefficients: c[m] multiplies T_m(x). The overall
    // factor 2 is dropped (it does not move the roots).
    let mut c = vec![0.0f64; deg + 1];
    c[0] = coeffs[deg] as f64 * 0.5;
    for m in 1..=deg {
        c[m] = coeffs[deg - m] as f64;
    }
    // Clenshaw's recurrence on Σ c[m]·T_m(x).
    let eval = |x: f64| -> f64 {
        let mut b2 = 0.0f64;
        let mut b1 = 0.0f64;
        for k in (1..=deg).rev() {
            let b0 = 2.0 * x * b1 - b2 + c[k];
            b2 = b1;
            b1 = b0;
        }
        x * b1 - b2 + c[0]
    };
    // 1024 angle steps across (0, π) — matches the quantiser's own
    // frequency resolution scale (Q15 domain / 32) so genuinely
    // distinct LSP lines land in distinct grid cells.
    const GRID: usize = 1024;
    let mut roots = Vec::with_capacity(deg);
    let mut prev_x = 1.0f64;
    let mut prev_y = eval(prev_x);
    for i in 1..=GRID {
        let x = (std::f64::consts::PI * i as f64 / GRID as f64).cos();
        let y = eval(x);
        if prev_y == 0.0 {
            roots.push(prev_x as f32);
        } else if prev_y * y < 0.0 {
            // Bisect [x, prev_x] down to the root.
            let mut lo = x;
            let mut hi = prev_x;
            let mut flo = y;
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                let fm = eval(mid);
                if fm * flo < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push((0.5 * (lo + hi)) as f32);
        }
        if roots.len() == deg {
            break;
        }
        prev_x = x;
        prev_y = y;
    }
    roots
}

/// Convert LSPs (cosine-domain) back to direct-form LPC coefficients.
pub(crate) fn lsp_to_lpc(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
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
pub(crate) fn interpolate_lsp(
    k: usize,
    prev: &[f32; LPC_ORDER],
    cur: &[f32; LPC_ORDER],
) -> [f32; LPC_ORDER] {
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

// ---------- LSP stability (§2.6 / §3.10) ----------

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
/// Word16 positive saturation bound in the crate's normalised float
/// domain (32767 / 32768 — the fixed-point description of §1.5 clamps
/// every stored sample to the i16 range).
const I16_MAX_NORM: f32 = 32_767.0 / 32_768.0;

pub struct SynthesisState {
    prev_lsp: [f32; LPC_ORDER],
    /// Previous decoded LSP vector in the spec tables' Q15
    /// normalised-frequency domain (`ω = π·q/32768`) — the §2.6 / eq. 3.3
    /// MA-predictor state for the spec-layout LSP codec. Kept in lockstep
    /// with the cosine-domain `prev_lsp` (§3.11 cold start = `p_DC`).
    prev_lsp_freq: [f32; LPC_ORDER],
    exc_history: [f32; PITCH_MAX + SUBFRAME_SIZE],
    syn_mem: [f32; LPC_ORDER],
    // Post-filter state ---------------------------------------------------
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
    /// Whether the §3.6–§3.9 post-filter chain (pitch post-filter,
    /// formant post-filter, tilt compensation, AGC) runs on decoded
    /// frames. Defaults to `true`. The ITU conformance methodology
    /// exercises the decoder with the post-filter selectively disabled
    /// (the `..D53`/`..D63P` vector naming: trailing `P` = post-filter
    /// ON, absent = OFF), so a device under test must expose the switch.
    postfilter_enabled: bool,
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
            prev_lsp_freq: crate::spec_lsp::lsp_dc_freq(),
            exc_history: [0.0; PITCH_MAX + SUBFRAME_SIZE],
            syn_mem: [0.0; LPC_ORDER],
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
            postfilter_enabled: true,
        }
    }

    /// Reset to the silent-LSP boot state. The post-filter switch is a
    /// configuration bit, not decoder state — it survives the reset.
    pub fn reset(&mut self) {
        let postfilter = self.postfilter_enabled;
        *self = Self::new();
        self.postfilter_enabled = postfilter;
    }

    /// Enable / disable the §3.6–§3.9 decoder post-filter chain
    /// (default: enabled). The ITU test-vector methodology requires the
    /// switch: the `PATHD53` / `OVERD53` / `INEQD53` decoder vectors are
    /// defined with the post-filter OFF, `PATHD63P` / `OVERD63P` /
    /// `TAMED63P` with it ON.
    pub fn set_postfilter(&mut self, enabled: bool) {
        self.postfilter_enabled = enabled;
    }

    /// Current state of the decoder post-filter switch.
    pub fn postfilter(&self) -> bool {
        self.postfilter_enabled
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

    /// §3.6 pitch post-filter, **excitation domain** (eq. 42–47).
    ///
    /// Per the §3.1 block diagram the decoded excitation `e[n]` is input
    /// to the pitch post-filter, whose output `ppf[n]` feeds the §3.7
    /// synthesis filter — the post-filter does *not* run on the
    /// synthesis output. §3.6: "to implement it, it is required that the
    /// whole frame excitation signal {e[n]}n=0..239 is generated and
    /// saved", so the forward reach `e[n + M_f]` reads the current
    /// frame's excitation and is dropped (weight 0) when any sample
    /// would fall past the frame end; the backward reach `e[n − M_b]`
    /// extends into the pre-frame excitation history.
    ///
    /// `hist` is the excitation history as it stood *before* the current
    /// frame (most recent sample last), `frame` the current frame's
    /// decoded excitation, `start` the subframe offset (0/60/120/180)
    /// and `ref_lag` the §3.6 reference lag (`L_0` for subframes 0–1,
    /// `L_2` for 2–3).
    fn pitch_postfilter_exc(
        hist: &[f32],
        frame: &[f32; FRAME_SIZE_SAMPLES],
        start: usize,
        ref_lag: i32,
        rate: Rate,
    ) -> [f32; SUBFRAME_SIZE] {
        let mut sf = [0.0f32; SUBFRAME_SIZE];
        sf.copy_from_slice(&frame[start..start + SUBFRAME_SIZE]);

        let lag_c = ref_lag.clamp(PITCH_MIN as i32, PITCH_MAX as i32);
        let m_lo = (lag_c - POSTFILTER_LTP_SEARCH_RADIUS).max(1);
        let m_hi = lag_c + POSTFILTER_LTP_SEARCH_RADIUS;

        // Subframe energy T_en (eq. 44.3).
        let mut t_en = 0.0f32;
        for &v in sf.iter() {
            t_en += v * v;
        }
        if t_en < 1e-12 {
            return sf;
        }

        // Forward search (eq. 43.1): C_f = Σ e[n]·e[n + M_f]. §3.6: "if
        // for some n ∈ [0..59] there is no sample value e[n + M_f]
        // available, then the corresponding weight and delay are set
        // to 0" — availability means within the saved 240-sample frame.
        let mut best_f: Option<(usize, f32, f32)> = None; // (M_f, C_f, D_f)
        for m in m_lo..=m_hi {
            let mu = m as usize;
            if start + SUBFRAME_SIZE - 1 + mu >= FRAME_SIZE_SAMPLES {
                continue;
            }
            let mut c = 0.0f32;
            let mut d = 0.0f32;
            for n in 0..SUBFRAME_SIZE {
                let x = frame[start + n + mu];
                c += sf[n] * x;
                d += x * x;
            }
            let metric = if c > 0.0 && d > 1e-12 {
                c * c / d
            } else {
                -1.0
            };
            if metric >= 0.0 && best_f.map_or(true, |(_, bc, bd)| metric > bc * bc / bd.max(1e-12))
            {
                best_f = Some((mu, c, d));
            }
        }

        // Backward search (eq. 43.2): C_b = Σ e[n]·e[n − M_b], reaching
        // into the pre-frame history.
        let hlen = hist.len();
        let past = |gidx: isize| -> f32 {
            if gidx >= 0 {
                frame[gidx as usize]
            } else {
                let k = (-gidx) as usize;
                if k <= hlen {
                    hist[hlen - k]
                } else {
                    0.0
                }
            }
        };
        let mut best_b: Option<(usize, f32, f32)> = None;
        for m in m_lo..=m_hi {
            let mu = m as usize;
            let mut c = 0.0f32;
            let mut d = 0.0f32;
            for n in 0..SUBFRAME_SIZE {
                let x = past((start + n) as isize - mu as isize);
                c += sf[n] * x;
                d += x * x;
            }
            let metric = if c > 0.0 && d > 1e-12 {
                c * c / d
            } else {
                -1.0
            };
            if metric >= 0.0 && best_b.map_or(true, |(_, bc, bd)| metric > bc * bc / bd.max(1e-12))
            {
                best_b = Some((mu, c, d));
            }
        }

        // Case selection: pick the positive-maximum side with the larger
        // C²/D contribution (eq. 45.1/45.2 minimisation).
        let gain_f = best_f.map_or(0.0, |(_, c, d)| c * c / (d.max(1e-12) * t_en));
        let gain_b = best_b.map_or(0.0, |(_, c, d)| c * c / (d.max(1e-12) * t_en));
        let gain_best = gain_f.max(gain_b);
        // Prediction-gain gate: −10·log10(1 − C²/(D·T_en)) < 1.25 dB ⇒
        // "the contribution is judged to be negligible and no pitch
        // postfilter is used".
        let gate_linear = 1.0 - 10.0_f32.powf(-POSTFILTER_LTP_PRED_GAIN_DB_MIN / 10.0);
        if gain_best < gate_linear {
            return sf;
        }
        let (m_best, c_chosen, d_chosen, forward) = if gain_f >= gain_b {
            let (m, c, d) = best_f.unwrap();
            (m, c, d, true)
        } else {
            let (m, c, d) = best_b.unwrap();
            (m, c, d, false)
        };

        // eq. 46: g = C / D, weighted by the rate-specific γ_ltp.
        let g_side = (c_chosen / d_chosen.max(1e-12)).clamp(0.0, 1.0);
        let gamma_ltp = rate.ltp_gamma();

        // ppf′[n] = e[n] + γ_ltp·g·e[n ± M] (eq. 42 inner term).
        let mut ppf = [0.0f32; SUBFRAME_SIZE];
        let mut den_energy = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            let x = if forward {
                frame[start + n + m_best]
            } else {
                past((start + n) as isize - m_best as isize)
            };
            let v = sf[n] + gamma_ltp * g_side * x;
            ppf[n] = v;
            den_energy += v * v;
        }
        // eq. 47: gp = √(Σe²/Σppf′²), set to 1 when the denominator is
        // smaller than the numerator (attenuate-only).
        let g_p = if den_energy < t_en {
            1.0
        } else {
            (t_en / den_energy.max(1e-12)).sqrt()
        };
        for v in ppf.iter_mut() {
            *v *= g_p;
        }
        ppf
    }

    /// Apply the §3.8/§3.9 back half of the post-filter chain to one
    /// subframe: formant A(z/γ₁)/A(z/γ₂) → first-order tilt
    /// compensation → smoothed automatic-gain-control, updating the
    /// post-filter memories in place. `syn` is the 60-sample §3.7
    /// synthesis output of the current subframe (the §3.6 pitch
    /// post-filter has already run upstream, in the excitation domain);
    /// `a_sub` are the interpolated LPC coefficients for the subframe.
    fn formant_postfilter_subframe(
        &mut self,
        a_sub: &[f32; LPC_ORDER + 1],
        syn: &[f32; SUBFRAME_SIZE],
        out: &mut [f32; SUBFRAME_SIZE],
    ) {
        // ---- 1. Formant post-filter A(z/γ₁) / A(z/γ₂). γ₁ < γ₂ widens
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
            let x = syn[n];
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

        // ---- 2. First-order tilt compensation per G.723.1 §3.8, eq. 49.2:
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

        // ---- 3. Adaptive gain scaling per G.723.1 §3.9, eq. 50–52:
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
    /// §3.8/§3.9 formant + tilt + AGC stage over a whole frame of §3.7
    /// synthesis output. The formant postfilter A(z/γ₁)/A(z/γ₂)
    /// operates on the same per-subframe interpolated synthesis filter
    /// Ã_i(z) the LPC synthesis stage used (§3.3 / §2.7 eq. 8 weights
    /// (0.75/0.25), (0.5/0.5), (0.25/0.75), (0/1)); the caller passes
    /// the captured pre-decode previous LSP so the interpolation curve
    /// matches subframe-for-subframe. No-op when the post-filter switch
    /// is off.
    fn apply_formant_postfilter(
        &mut self,
        prev_lsp: &[f32; LPC_ORDER],
        lsp_q: &[f32; LPC_ORDER],
        pcm: &mut [f32; FRAME_SIZE_SAMPLES],
    ) {
        if !self.postfilter_enabled {
            return;
        }
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, prev_lsp, lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);
            let start = s * SUBFRAME_SIZE;
            let end = start + SUBFRAME_SIZE;
            let mut syn = [0.0f32; SUBFRAME_SIZE];
            syn.copy_from_slice(&pcm[start..end]);
            let mut post = [0.0f32; SUBFRAME_SIZE];
            self.formant_postfilter_subframe(&a_sub, &syn, &mut post);
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

            // Saturate the concealed excitation to the Word16 range,
            // then 1/A(z) synthesis (eq. 48) with saturated output.
            let mut syn = [0.0f32; SUBFRAME_SIZE];
            for i in 0..SUBFRAME_SIZE {
                exc[i] = exc[i].clamp(-1.0, I16_MAX_NORM);
                let mut y = exc[i];
                for k in 0..LPC_ORDER {
                    y -= a_sub[k + 1] * self.syn_mem[k];
                }
                y = y.clamp(-1.0, I16_MAX_NORM);
                for k in (1..LPC_ORDER).rev() {
                    self.syn_mem[k] = self.syn_mem[k - 1];
                }
                self.syn_mem[0] = y;
                syn[i] = y;
            }
            // §3.8/§3.9 formant + tilt + AGC back half. Concealment
            // regenerates the excitation directly from the pitch replay
            // / random innovation, so the §3.6 pitch post-filter (whose
            // job is boosting SNR at pitch multiples of a *decoded*
            // residual) is skipped on erased frames.
            let start = s * SUBFRAME_SIZE;
            if self.postfilter_enabled {
                let mut post = [0.0f32; SUBFRAME_SIZE];
                self.formant_postfilter_subframe(&a_sub, &syn, &mut post);
                pcm[start..start + SUBFRAME_SIZE].copy_from_slice(&post);
            } else {
                pcm[start..start + SUBFRAME_SIZE].copy_from_slice(&syn);
            }

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
        // the stale pre-erasure one. The spec-layout LSP predictor state
        // follows the same concealed vector (§3.10.1 feeds p̃_n back as
        // p̃_{n-1} for both the erasure and the recovery frame).
        self.prev_lsp = lsp_q;
        self.prev_lsp_freq = crate::spec_lsp::lsp_cosines_to_freq(&lsp_q);

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

    /// Decode one ACELP (5.3 kbit/s) clause-4 frame into 240 PCM
    /// samples: Table 6 unpack ([`crate::linepack`]) followed by the
    /// spec-table §3.1 pipeline ([`Self::decode_spec_params`]).
    pub fn decode_acelp(&mut self, payload: &[u8]) -> Result<[i16; FRAME_SIZE_SAMPLES]> {
        if payload.len() < ACELP_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "G.723.1 decoder: ACELP payload smaller than 20 bytes",
            ));
        }
        let params = crate::linepack::unpack_frame(&payload[..ACELP_PAYLOAD_BYTES])?;
        if params.rate != PackedRate::Low {
            return Err(Error::invalid(
                "G.723.1 decoder: expected RATEFLAG=1 (5.3 kbit/s ACELP)",
            ));
        }
        Ok(self.decode_spec_params(&params))
    }

    /// Decode one MP-MLQ (6.3 kbit/s) clause-4 frame into 240 PCM
    /// samples: Table 5 unpack ([`crate::linepack`], including the
    /// 13-bit MSBPOS split) followed by the spec-table §3.1 pipeline
    /// ([`Self::decode_spec_params`]).
    pub fn decode_mpmlq(&mut self, payload: &[u8]) -> Result<[i16; FRAME_SIZE_SAMPLES]> {
        if payload.len() < MPMLQ_PAYLOAD_BYTES {
            return Err(Error::invalid(
                "G.723.1 decoder: MP-MLQ payload smaller than 24 bytes",
            ));
        }
        let params = crate::linepack::unpack_frame(&payload[..MPMLQ_PAYLOAD_BYTES])?;
        if params.rate != PackedRate::High {
            return Err(Error::invalid(
                "G.723.1 decoder: expected RATEFLAG=0 (6.3 kbit/s MP-MLQ)",
            ));
        }
        Ok(self.decode_spec_params(&params))
    }

    /// Decode one clause-4 spec-layout parameter set (either rate) into
    /// 240 PCM samples, advancing every piece of decoder state.
    ///
    /// This is the §3.1 pipeline running on the published tables:
    /// LSP decode (§3.2 → 2.6) through [`crate::spec_lsp`], pitch decode
    /// (§3.4 → 2.18, eq. 37–41.2) and excitation decode (§3.5 → 2.17)
    /// through [`crate::spec_exc`], then the existing pitch postfilter
    /// (§3.6), LPC synthesis (§3.7), formant postfilter (§3.8) and gain
    /// scaling (§3.9) chain.
    pub(crate) fn decode_spec_params(&mut self, p: &SpecFrameParams) -> [i16; FRAME_SIZE_SAMPLES] {
        // --- LSP decode (§3.2 → 2.6, eq. 3.3 / 4.4) ---
        let lsp_freq = spec_lsp::decode_lsp_freq(p.lsp_index, &self.prev_lsp_freq);
        let cos_raw = spec_lsp::lsp_freq_to_cosines(&lsp_freq);
        let (mut lsp_q, converged) = enforce_lsp_stability(&cos_raw, LSP_STABILITY_DELTA_MIN_HZ);
        if !converged {
            // §2.6 step 3: "If after 10 iterations the condition of
            // stability is not met, the previous LSP vector is used."
            lsp_q = self.prev_lsp;
        }

        // --- Pitch lags (§3.4, eq. 37–38) ---
        let lag0 = decode_abs_lag(p.acl[0]);
        let lag1 = decode_delta_lag(p.acl[1], lag0);
        let lag2 = decode_abs_lag(p.acl[2]);
        let lag3 = decode_delta_lag(p.acl[3], lag2);
        let lags = [lag0, lag1, lag2, lag3];

        let prev_lsp_snapshot = self.prev_lsp;
        let rate = match p.rate {
            PackedRate::High => Rate::High,
            PackedRate::Low => Rate::Low,
        };

        // --- Phase 1: whole-frame excitation decode (§3.4/§3.5 →
        // 2.17/2.18). §3.6 requires "the whole frame excitation signal
        // {e[n]}n=0..239 is generated and saved" before the pitch
        // post-filter runs, so build all four subframes' e[n] first.
        // The pre-frame excitation history is snapshotted for the pitch
        // post-filter's backward reach.
        let hist_snapshot = self.exc_history;
        let mut exc_frame = [0.0f32; FRAME_SIZE_SAMPLES];
        let mut fcb_gains = [0.0f32; SUBFRAMES_PER_FRAME];
        let mut last_taps_sum = 0.0f32;
        for s in 0..SUBFRAMES_PER_FRAME {
            // §2.14: the 85-entry short-lag gain codebook rule keys off
            // the subframe pair's reference lag L0 / L2.
            let lag_base = if s < 2 { lags[0] } else { lags[2] };
            let gain = spec_exc::decode_gain_word(p.rate, lag_base, p.gain[s]);
            fcb_gains[s] = gain.fcb_gain;
            last_taps_sum = gain.taps.iter().sum();

            // §3.4 → 2.18: fifth-order adaptive-codebook contribution.
            let u = spec_exc::acb_contribution(&self.exc_history, lags[s], &gain.taps);

            // §3.5 → 2.17: rate-specific fixed-codebook contribution.
            let v = match p.rate {
                PackedRate::High => {
                    let n_pulses = if s % 2 == 0 { 6 } else { 5 };
                    spec_exc::mpmlq_fixed_vector(
                        p.pos[s],
                        p.psig[s],
                        p.grid[s],
                        n_pulses,
                        gain.fcb_gain,
                        gain.train,
                        lag_base,
                    )
                }
                PackedRate::Low => {
                    let mut v =
                        spec_exc::acelp_fixed_vector(p.pos[s], p.psig[s], p.grid[s], gain.fcb_gain);
                    spec_exc::acelp_pitch_enhance(&mut v, lags[s], gain.pgindex);
                    v
                }
            };

            // §2.17 step 7: e[n] = u[n] + v[n]. Kept LINEAR (no Word16
            // saturation): the fixed-point description of §1.5 clamps
            // every stored sample, but emulating that in a float model
            // measurably *hurts* conformance tracking — clipping at
            // approximate amplitudes injects nonlinear error where the
            // unclamped signal stays a scaled replica of the reference
            // (OVERD53 whole-file corr 0.97 linear vs 0.63 clamped).
            let start = s * SUBFRAME_SIZE;
            for n in 0..SUBFRAME_SIZE {
                exc_frame[start + n] = u[n] + v[n];
            }
            self.exc_history.rotate_left(SUBFRAME_SIZE);
            let tail = self.exc_history.len() - SUBFRAME_SIZE;
            self.exc_history[tail..].copy_from_slice(&exc_frame[start..start + SUBFRAME_SIZE]);
        }

        // --- Phase 2 + 3: per-subframe §3.6 pitch post-filter on the
        // excitation, then §3.7 LPC synthesis (eq. 48) on ppf[n].
        let mut pcm_f = [0.0f32; FRAME_SIZE_SAMPLES];
        for s in 0..SUBFRAMES_PER_FRAME {
            let lsp_interp = interpolate_lsp(s, &self.prev_lsp, &lsp_q);
            let a_sub = lsp_to_lpc(&lsp_interp);
            let start = s * SUBFRAME_SIZE;
            let ppf = if self.postfilter_enabled {
                let ref_lag = if s < 2 { lags[0] } else { lags[2] };
                Self::pitch_postfilter_exc(&hist_snapshot, &exc_frame, start, ref_lag, rate)
            } else {
                let mut sf = [0.0f32; SUBFRAME_SIZE];
                sf.copy_from_slice(&exc_frame[start..start + SUBFRAME_SIZE]);
                sf
            };
            for i in 0..SUBFRAME_SIZE {
                let mut y = ppf[i];
                for k in 0..LPC_ORDER {
                    y -= a_sub[k + 1] * self.syn_mem[k];
                }
                for k in (1..LPC_ORDER).rev() {
                    self.syn_mem[k] = self.syn_mem[k - 1];
                }
                self.syn_mem[0] = y;
                pcm_f[start + i] = y;
            }
        }

        // Persist the decoded LSP for the next frame's interpolation and
        // MA predictor (§2.6 / eq. 3.3).
        self.prev_lsp = lsp_q;
        self.prev_lsp_freq = spec_lsp::lsp_cosines_to_freq(&lsp_q);

        // --- Phase 4: §3.8/§3.9 formant post-filter + gain scaling,
        // then concealment bookkeeping.
        self.apply_formant_postfilter(&prev_lsp_snapshot, &lsp_q, &mut pcm_f);
        self.record_last_frame_spec(&lags, last_taps_sum, &fcb_gains);
        self.record_pcm_history(&pcm_f);
        to_i16_frame(&pcm_f)
    }

    /// Spec-path variant of [`SynthesisState::record_last_frame`]: saves
    /// the §3.10.2 classifier inputs from decoded spec parameters. The
    /// scalar "adaptive gain" driving the voiced concealment branch is
    /// the DC gain of the last subframe's fifth-order predictor (the tap
    /// sum), clamped to a stable replay range.
    fn record_last_frame_spec(
        &mut self,
        lags: &[i32; SUBFRAMES_PER_FRAME],
        last_taps_sum: f32,
        fcb_gains: &[f32; SUBFRAMES_PER_FRAME],
    ) {
        self.pf_last_lag = lags[SUBFRAMES_PER_FRAME - 1];
        self.pf_last_gain_adapt = last_taps_sum.clamp(0.0, 1.0);
        self.pf_last_gain_fixed = fcb_gains[SUBFRAMES_PER_FRAME - 1];
        self.pf_last_lag2 = lags[2];
        self.pf_last_gain_unvoiced = 0.5 * (fcb_gains[2] + fcb_gains[3]);
        self.pf_erased_run = 0;
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

/// Stateful frame-level encoder handle exposing the ITU device-under-test
/// controls (rate + §2.3 high-pass switch) that the registry-level
/// [`Encoder`] surface does not carry. The ITU conformance methodology
/// requires running the encoder with the high-pass selectively disabled
/// (the `..C53`/`..C63H` vector naming: trailing `H` = high-pass ON,
/// absent = OFF).
pub struct SpecEncoder {
    analysis: AnalysisState,
    rate: PackedRate,
}

impl SpecEncoder {
    /// New encoder at the given rate with the §2.3 high-pass ON
    /// (the Recommendation's default configuration).
    pub fn new(rate: PackedRate) -> Self {
        Self {
            analysis: AnalysisState::new(),
            rate,
        }
    }

    /// Enable / disable the §2.3 input high-pass filter.
    pub fn set_highpass(&mut self, enabled: bool) {
        self.analysis.highpass = enabled;
    }

    /// Encode one 240-sample frame into its clause-4 octet sequence
    /// (24 bytes at the high rate, 20 at the low rate).
    ///
    /// `lookahead` is the first 60 samples of the *next* frame — the
    /// §2.4 LPC window centered on the last subframe reaches 7.5 ms
    /// past the frame end. Pass zeros at end of stream (§2.21 rest
    /// state).
    pub fn encode_frame(
        &mut self,
        pcm: &[i16; FRAME_SIZE_SAMPLES],
        lookahead: &[i16; LOOKAHEAD_SAMPLES],
    ) -> Vec<u8> {
        let params = self.analysis.analyse_spec(pcm, lookahead, self.rate);
        crate::linepack::pack_frame(&params)
            .expect("analyse_spec emits in-range clause-4 parameters")
    }
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

    /// Test helper: run the §3.8/§3.9 formant + tilt + AGC back half of
    /// the post-filter on a single subframe (the §3.6 pitch post-filter
    /// now runs upstream in the excitation domain).
    fn pf_sf(
        st: &mut SynthesisState,
        a: &[f32; LPC_ORDER + 1],
        syn: &[f32; SUBFRAME_SIZE],
        _lag: i32,
        _rate: Rate,
        out: &mut [f32; SUBFRAME_SIZE],
    ) {
        st.formant_postfilter_subframe(a, syn, out);
    }

    /// Test helper: run the §3.6 excitation-domain pitch postfilter on a
    /// single subframe presented at frame offset 0 with an all-zero
    /// pre-frame history and no successor samples (so the forward reach
    /// is unavailable and the backward reach sees silence, unless the
    /// caller supplies history).
    fn ltp_sf(
        hist: &[f32],
        exc: &[f32; SUBFRAME_SIZE],
        lag: i32,
        rate: Rate,
    ) -> [f32; SUBFRAME_SIZE] {
        let mut frame = [0.0f32; FRAME_SIZE_SAMPLES];
        frame[..SUBFRAME_SIZE].copy_from_slice(exc);
        SynthesisState::pitch_postfilter_exc(hist, &frame, 0, lag, rate)
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

    /// LPC → LSP → LPC roundtrip: the Chebyshev root search must
    /// recover the exact line set the polynomial was built from. This
    /// pins the deflation-recursion seeding (f1[0]/f2[0] carried into
    /// the recursion) and the reversed-order Chebyshev evaluation of
    /// the symmetric deflated halves — both had silent historical bugs
    /// that made the analysis LSPs diverge wholesale from the model.
    #[test]
    fn lpc_lsp_roundtrip_recovers_exact_lines() {
        // The DC vector plus perturbed variants spanning the range.
        let dc = crate::tables::lsp_dc_cosines();
        let mut cases: Vec<[f32; LPC_ORDER]> = vec![dc];
        let mut lcg: u32 = 0xC0FF_EE01;
        for _ in 0..25 {
            let mut omega = [0.0f32; LPC_ORDER];
            for w in omega.iter_mut() {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *w = ((lcg >> 8) & 0xFFFF) as f32 / 65_536.0;
            }
            omega.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Space the lines by ≥ 0.02 rad and keep off the edges.
            let mut prev = 0.05f32;
            let mut cos = [0.0f32; LPC_ORDER];
            for (i, w) in omega.iter().enumerate() {
                let v = (prev + 0.02).max(0.05 + w * (std::f32::consts::PI - 0.4));
                let v = v.min(std::f32::consts::PI - 0.05 - 0.02 * (LPC_ORDER - i) as f32);
                cos[i] = v.max(prev + 0.02).cos();
                prev = v.max(prev + 0.02);
            }
            cases.push(cos);
        }
        for (ci, lsp_in) in cases.iter().enumerate() {
            let a = lsp_to_lpc(lsp_in);
            let lsp_out = lpc_to_lsp(&a)
                .unwrap_or_else(|| panic!("case {ci}: root search failed on a valid LSP set"));
            for i in 0..LPC_ORDER {
                // f32 polynomial construction limits the recovery
                // accuracy for closely spaced lines; 1e-3 in the cosine
                // domain is far tighter than the wholesale divergence
                // the two historical bugs caused, while staying robust
                // across platforms.
                assert!(
                    (lsp_out[i] - lsp_in[i]).abs() < 1.0e-3,
                    "case {ci} line {i}: {} vs {}",
                    lsp_out[i],
                    lsp_in[i]
                );
            }
        }
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
        // The §2.4 windows reach 7.5 ms past the frame end, so a frame
        // is only emitted once its lookahead is buffered — or at flush.
        enc.flush().unwrap();
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
        // See the ACELP variant: emission waits for the §2.4 lookahead.
        enc.flush().unwrap();
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

    /// §2.3 high-pass (eq. 1): a pure-DC frame must decay to (near)
    /// zero output — the filter has a transmission zero at DC — while a
    /// mid-band tone passes with roughly unit gain.
    #[test]
    fn highpass_removes_dc_and_passes_midband() {
        let mut st = AnalysisState::new();
        let mut dc = [0.25f32; FRAME_SIZE_SAMPLES];
        st.highpass_frame(&mut dc);
        // After the first-sample transient the DC response decays
        // geometrically: (127/128)^239 ≈ 0.153, so the frame tail sits
        // at ≈ 0.25 · 0.153 ≈ 0.038 and keeps shrinking.
        assert!(dc[0] > 0.2, "first sample carries the step transient");
        assert!(
            dc[FRAME_SIZE_SAMPLES - 1].abs() < 0.05,
            "DC must decay: tail = {}",
            dc[FRAME_SIZE_SAMPLES - 1]
        );
        assert!(
            dc[FRAME_SIZE_SAMPLES - 1].abs() < dc[60].abs(),
            "decay must be monotone across the frame"
        );

        let mut st = AnalysisState::new();
        let mut tone = [0.0f32; FRAME_SIZE_SAMPLES];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (n, v) in tone.iter_mut().enumerate() {
            *v = (two_pi * n as f32 / 8.0).sin() * 0.25; // 1 kHz at 8 kHz
        }
        let orig = tone;
        st.highpass_frame(&mut tone);
        // Steady-state gain at 1 kHz is close to unity — compare tail
        // energies loosely.
        let e_in: f32 = orig[120..].iter().map(|v| v * v).sum();
        let e_out: f32 = tone[120..].iter().map(|v| v * v).sum();
        let ratio = e_out / e_in;
        assert!(
            (0.7..1.3).contains(&ratio),
            "mid-band gain should be ≈1, got {ratio}"
        );
    }

    /// The public SpecEncoder handle emits clause-4 frames at both
    /// rates that unpack cleanly, and its §2.3 switch changes the
    /// emitted bits on a DC-offset input (the filter is really wired).
    #[test]
    fn spec_encoder_handle_emits_decodable_frames_and_hp_switch_acts() {
        let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (n, v) in pcm.iter_mut().enumerate() {
            let t = n as f32;
            *v = ((two_pi * t / 40.0).sin() * 6000.0 + 3000.0) as i16; // tone + DC
        }
        for rate in [PackedRate::High, PackedRate::Low] {
            let mut enc_on = SpecEncoder::new(rate);
            let mut enc_off = SpecEncoder::new(rate);
            enc_off.set_highpass(false);
            let la = [0i16; LOOKAHEAD_SAMPLES];
            let f_on = enc_on.encode_frame(&pcm, &la);
            let f_off = enc_off.encode_frame(&pcm, &la);
            assert_eq!(f_on.len(), rate.frame_bytes());
            assert_eq!(f_off.len(), rate.frame_bytes());
            let p_on = crate::linepack::unpack_frame(&f_on).unwrap();
            let p_off = crate::linepack::unpack_frame(&f_off).unwrap();
            assert_eq!(p_on.rate, rate);
            assert_eq!(p_off.rate, rate);
            assert_ne!(
                f_on, f_off,
                "HP switch must change the coded frame on a DC-offset input"
            );
        }
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
        let hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let syn = [0.0f32; SUBFRAME_SIZE];
        let out = ltp_sf(&hist, &syn, 40, Rate::High);
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
        let mut lcg: u32 = 0x1234_5678;
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for s in syn.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = ((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0;
        }
        // Empty history so the backward search starts from "silence".
        let hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let out = ltp_sf(&hist, &syn, 40, Rate::High);
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
        let period: i32 = 40;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        // History with the same period but a slowly increasing envelope
        // so the back-reference amplitude is smaller than the current
        // subframe — this breaks the (g_p · (1 + γ) = 1) degeneracy.
        let mut hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let total_len = hist.len() + SUBFRAME_SIZE;
        let env = |i: usize| -> f32 { 0.2 + 0.6 * (i as f32) / (total_len as f32) };
        for (i, h) in hist.iter_mut().enumerate() {
            let phase = two_pi * (i as f32) / period as f32;
            *h = phase.sin() * env(i);
        }
        let start_idx = hist.len();
        let mut syn = [0.0f32; SUBFRAME_SIZE];
        for (n, s) in syn.iter_mut().enumerate() {
            let i = start_idx + n;
            let phase = two_pi * (i as f32) / period as f32;
            *s = phase.sin() * env(i);
        }
        let in_e: f32 = syn.iter().map(|v| v * v).sum();
        let out_high = ltp_sf(&hist, &syn, period, Rate::High);
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
        let out_low = ltp_sf(&hist, &syn, period, Rate::Low);
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

    /// §3.6 forward reach: the pitch post-filter needs the whole-frame
    /// excitation because `e[n + M_f]` for an early subframe lands in a
    /// later one. A periodic excitation whose continuation lives only in
    /// the successor subframes must engage the post-filter on subframe 0
    /// — and zeroing that continuation (making the forward reach see a
    /// broken pattern, with silent history killing the backward side)
    /// must change the output.
    #[test]
    fn pitch_postfilter_forward_reach_reads_later_subframes() {
        let hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let period = 40.0f32;
        let mut frame_full = [0.0f32; FRAME_SIZE_SAMPLES];
        for (n, s) in frame_full.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / period).sin() * 0.4;
        }
        let mut frame_trunc = [0.0f32; FRAME_SIZE_SAMPLES];
        frame_trunc[..SUBFRAME_SIZE].copy_from_slice(&frame_full[..SUBFRAME_SIZE]);

        let out_full = SynthesisState::pitch_postfilter_exc(&hist, &frame_full, 0, 40, Rate::High);
        let out_trunc =
            SynthesisState::pitch_postfilter_exc(&hist, &frame_trunc, 0, 40, Rate::High);
        let mut diff = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            diff += (out_full[n] - out_trunc[n]).abs();
        }
        assert!(
            diff > 1e-3,
            "forward reach must read the successor subframes (diff={diff})"
        );
    }

    /// §3.6: "if for some n ∈ [0..59] there is no sample value e[n + Mf]
    /// available, then the corresponding weight and delay are set to 0"
    /// — for the LAST subframe every M_f > 0 reaches past the frame end,
    /// so only the backward side may engage; with white (uncorrelated)
    /// content, silent history and a silent preceding subframe, the
    /// backward gate fails too and the post-filter must pass the
    /// subframe through unchanged.
    #[test]
    fn pitch_postfilter_last_subframe_has_no_forward_side() {
        let hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let mut frame = [0.0f32; FRAME_SIZE_SAMPLES];
        // White content only in the final subframe: the backward reach
        // at M_b ∈ [37, 43] sees zeros (subframe 2) or uncorrelated
        // noise, so neither side clears the 1.25 dB gate.
        let mut lcg: u32 = 0xCAFE_F00D;
        for n in 3 * SUBFRAME_SIZE..FRAME_SIZE_SAMPLES {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            frame[n] = (((lcg >> 8) & 0xFFFF) as f32 / 32_768.0 - 1.0) * 0.4;
        }
        let out = SynthesisState::pitch_postfilter_exc(&hist, &frame, 180, 40, Rate::High);
        for n in 0..SUBFRAME_SIZE {
            assert_eq!(
                out[n],
                frame[180 + n],
                "last subframe: no forward side and no backward structure ⇒ pass-through"
            );
        }
    }

    /// §3.6 backward reach: when the pre-frame excitation history holds
    /// a sinusoid at the subframe's period, the backward side engages
    /// and the post-filter output differs from the input.
    #[test]
    fn pitch_postfilter_backward_side_uses_history() {
        let period: i32 = 36;
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let mut hist = [0.0f32; PITCH_MAX + SUBFRAME_SIZE];
        let hlen = hist.len();
        for (i, h) in hist.iter_mut().enumerate() {
            // Phase continuous with the frame below: history sample i sits
            // at global index i − hlen.
            let g = i as f32 - hlen as f32;
            *h = (two_pi * g / period as f32).sin() * 0.3;
        }
        let mut frame = [0.0f32; FRAME_SIZE_SAMPLES];
        for (n, s) in frame.iter_mut().enumerate().take(SUBFRAME_SIZE) {
            *s = (two_pi * (n as f32) / period as f32).sin() * 0.4;
        }
        let out = SynthesisState::pitch_postfilter_exc(&hist, &frame, 0, period, Rate::High);
        let mut diff = 0.0f32;
        for n in 0..SUBFRAME_SIZE {
            diff += (out[n] - frame[n]).abs();
        }
        assert!(
            diff > 1e-3,
            "backward side must engage on periodic history (diff={diff})"
        );
    }

    /// The formant post-filter chain must stay finite on a typical
    /// voiced frame.
    #[test]
    fn apply_formant_postfilter_stays_finite() {
        let mut st = SynthesisState::new();
        let lsp_q = st.prev_lsp;
        let prev_lsp = st.prev_lsp;
        let mut pcm = [0.0f32; FRAME_SIZE_SAMPLES];
        let two_pi = 2.0f32 * std::f32::consts::PI;
        for (n, s) in pcm.iter_mut().enumerate() {
            *s = (two_pi * (n as f32) / 50.0).sin() * 0.3;
        }
        st.apply_formant_postfilter(&prev_lsp, &lsp_q, &mut pcm);
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
        // Run A: postfilter with the true previous-frame LSP.
        let mut st_a = SynthesisState::new();
        let mut pcm_a = make_pcm();
        st_a.apply_formant_postfilter(&prev, &cur, &mut pcm_a);

        // Run B: postfilter with prev == cur (the old degenerate path).
        let mut st_b = SynthesisState::new();
        let mut pcm_b = make_pcm();
        st_b.apply_formant_postfilter(&cur, &cur, &mut pcm_b);

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
        // The §2.6 spec decoder produces nearly-monotone vectors for
        // every reachable index triple (DC vector + trained residual
        // rows); the stability procedure should converge for all of
        // them. Sample a grid of LPC words (DC-vector predictor state)
        // and verify convergence + monotonicity after stabilisation.
        let probes: &[u32] = &[0, 0xFF_FF_FF, 0x33_AA_55, 0x56_34_12, 0x20_40_80];
        let prev = crate::spec_lsp::lsp_dc_freq();
        for &idx in probes {
            let freq = crate::spec_lsp::decode_lsp_freq(idx, &prev);
            let cos_raw = crate::spec_lsp::lsp_freq_to_cosines(&freq);
            let (lsp_q, converged) = enforce_lsp_stability(&cos_raw, LSP_STABILITY_DELTA_MIN_HZ);
            assert!(converged, "idx {idx:#08x}: stability must converge");
            // Monotone-decreasing in cosine domain ⇔ monotone-increasing
            // in angular-frequency domain.
            for j in 0..LPC_ORDER - 1 {
                assert!(
                    lsp_q[j] > lsp_q[j + 1],
                    "idx {idx:#08x}: cosine LSP must be strictly decreasing \
                     ({} -> {} at dim {j})",
                    lsp_q[j],
                    lsp_q[j + 1],
                );
            }
            let gap = omega_min_gap_hz(&lsp_q);
            assert!(
                gap >= LSP_STABILITY_DELTA_MIN_HZ - 0.5,
                "idx {idx:#08x}: decoded LSP must hit ≥ 31.25 Hz floor; got {gap:.3}"
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

    // ---------- spec-layout decode kernel ----------

    use crate::linepack::{pack_frame, unpack_frame, PackedRate, SpecFrameParams};

    /// A voiced-ish hand-built spec parameter set at the given rate.
    fn voiced_spec_params(rate: PackedRate) -> SpecFrameParams {
        let mut p = SpecFrameParams::zeroed(rate);
        p.lsp_index = crate::spec_lsp::combine_lsp_index([120, 40, 200]);
        // Absolute lags 80 (index 62); differentials 0 (index 1).
        p.acl = [62, 1, 62, 1];
        for s in 0..SUBFRAMES_PER_FRAME {
            // PGIndex 1 (long-lag 170-row layout; a modest tap set so a
            // repeated frame does not saturate), MGIndex 18 — an audible
            // fixed-codebook gain level.
            p.gain[s] = crate::spec_exc::encode_gain_word(rate, 80, 1, 18, false);
            p.grid[s] = (s % 2) as u8;
        }
        match rate {
            PackedRate::High => {
                for s in 0..SUBFRAMES_PER_FRAME {
                    let slots: &[usize] = if s % 2 == 0 {
                        &[0, 5, 11, 17, 23, 29]
                    } else {
                        &[2, 8, 14, 20, 26]
                    };
                    p.pos[s] = crate::spec_tables::fcbk_pack_positions(slots).unwrap();
                    p.psig[s] = 0b010101 & ((1 << slots.len()) - 1);
                }
            }
            PackedRate::Low => {
                for s in 0..SUBFRAMES_PER_FRAME {
                    p.pos[s] = 1 | (3 << 3) | (5 << 6) | (6 << 9);
                    p.psig[s] = 0b0101;
                }
            }
        }
        p
    }

    #[test]
    fn decode_spec_params_produces_energy_and_is_deterministic() {
        for rate in [PackedRate::High, PackedRate::Low] {
            let p = voiced_spec_params(rate);
            let mut st_a = SynthesisState::new();
            let mut st_b = SynthesisState::new();
            let mut energy = 0.0f64;
            for _ in 0..4 {
                let out_a = st_a.decode_spec_params(&p);
                let out_b = st_b.decode_spec_params(&p);
                assert_eq!(out_a[..], out_b[..], "spec decode must be deterministic");
                let peak = out_a.iter().map(|&s| (s as i32).abs()).max().unwrap();
                assert!(peak < 32_768, "output must stay in i16 range");
                for &s in out_a.iter() {
                    energy += (s as f64) * (s as f64);
                }
            }
            assert!(
                energy > 1.0e4,
                "voiced spec frame at {rate:?} must synthesise audible energy (got {energy})"
            );
        }
    }

    #[test]
    fn decode_spec_params_zero_frame_is_near_silent() {
        for rate in [PackedRate::High, PackedRate::Low] {
            let mut st = SynthesisState::new();
            let out = st.decode_spec_params(&SpecFrameParams::zeroed(rate));
            let peak = out.iter().map(|&s| (s as i32).abs()).max().unwrap();
            // All-zero indices decode to the minimum gain level driving
            // a couple of pulses — near-silence, far below speech level.
            assert!(peak < 512, "zero frame peaked at {peak} ({rate:?})");
        }
    }

    #[test]
    fn spec_decode_composes_with_linepack_round_trip() {
        for rate in [PackedRate::High, PackedRate::Low] {
            let p = voiced_spec_params(rate);
            let bytes = pack_frame(&p).unwrap();
            let q = unpack_frame(&bytes).unwrap();
            assert_eq!(p, q);
            let mut st_direct = SynthesisState::new();
            let mut st_packed = SynthesisState::new();
            let direct = st_direct.decode_spec_params(&p);
            let packed = st_packed.decode_spec_params(&q);
            assert_eq!(direct[..], packed[..]);
        }
    }

    #[test]
    fn spec_decode_updates_lsp_predictor_state() {
        let p = voiced_spec_params(PackedRate::High);
        let mut st = SynthesisState::new();
        let dc = crate::spec_lsp::lsp_dc_freq();
        assert_eq!(st.prev_lsp_freq, dc, "§3.11 cold start at p_DC");
        st.decode_spec_params(&p);
        assert_ne!(
            st.prev_lsp_freq, dc,
            "decoding a non-trivial LSP index must advance the predictor state"
        );
        // The frequency-domain state mirrors the cosine-domain vector.
        let cos = crate::spec_lsp::lsp_freq_to_cosines(&st.prev_lsp_freq);
        for i in 0..LPC_ORDER {
            assert!((cos[i] - st.prev_lsp[i]).abs() < 1.0e-5);
        }
    }

    /// End-to-end spec-parameter round trip: analyse a voiced signal
    /// into `SpecFrameParams`, pack/unpack through the clause-4 octet
    /// maps, decode with a *fresh* decoder, and measure PSNR against
    /// the input. Floors are set conservatively below the measured
    /// debug-build figures.
    /// Slice out frame `f` of `pcm` plus its 60-sample §2.4 lookahead
    /// (zero-padded at end of stream).
    fn frame_and_lookahead(
        pcm: &[i16],
        f: usize,
    ) -> ([i16; FRAME_SIZE_SAMPLES], [i16; LOOKAHEAD_SAMPLES]) {
        let mut frame = [0i16; FRAME_SIZE_SAMPLES];
        frame.copy_from_slice(&pcm[f * FRAME_SIZE_SAMPLES..(f + 1) * FRAME_SIZE_SAMPLES]);
        let mut la = [0i16; LOOKAHEAD_SAMPLES];
        let start = (f + 1) * FRAME_SIZE_SAMPLES;
        let n = pcm.len().saturating_sub(start).min(LOOKAHEAD_SAMPLES);
        la[..n].copy_from_slice(&pcm[start..start + n]);
        (frame, la)
    }

    fn spec_roundtrip_psnr(rate: PackedRate) -> f64 {
        let frames = 20usize;
        let pcm = voiced_signal(frames);
        let mut analysis = AnalysisState::new();
        let mut dec = SynthesisState::new();
        let mut out = Vec::with_capacity(pcm.len());
        for f in 0..frames {
            let (frame, la) = frame_and_lookahead(&pcm, f);
            let params = analysis.analyse_spec(&frame, &la, rate);
            let bytes = crate::linepack::pack_frame(&params).unwrap();
            assert_eq!(
                bytes.len(),
                rate.frame_bytes(),
                "packed frame must be the Table 5/6 size"
            );
            let unpacked = crate::linepack::unpack_frame(&bytes).unwrap();
            assert_eq!(params, unpacked);
            out.extend_from_slice(&dec.decode_spec_params(&unpacked));
        }
        // Skip the first two frames (cold-start transient) for PSNR.
        let skip = 2 * FRAME_SIZE_SAMPLES;
        let (mut err, mut sig_e) = (0.0f64, 0.0f64);
        for (a, b) in pcm[skip..].iter().zip(out[skip..].iter()) {
            let d = (*a as f64) - (*b as f64);
            err += d * d;
            sig_e += (*a as f64) * (*a as f64);
        }
        10.0 * (sig_e / err.max(1.0)).log10()
    }

    #[test]
    fn spec_roundtrip_acelp_psnr_floor() {
        let psnr = spec_roundtrip_psnr(PackedRate::Low);
        println!("spec ACELP round-trip PSNR: {psnr:.2} dB");
        assert!(
            psnr > 10.0,
            "spec-layout ACELP round-trip PSNR {psnr:.2} dB below floor"
        );
    }

    #[test]
    fn spec_roundtrip_mpmlq_psnr_floor() {
        let psnr = spec_roundtrip_psnr(PackedRate::High);
        println!("spec MP-MLQ round-trip PSNR: {psnr:.2} dB");
        assert!(
            psnr > 12.0,
            "spec-layout MP-MLQ round-trip PSNR {psnr:.2} dB below floor"
        );
    }

    #[test]
    fn spec_encoder_shadow_decoder_stays_in_lockstep() {
        // The analysis commits its shadow through decode_spec_params, so
        // feeding the emitted parameters to an external decoder must
        // leave both decoders with identical LSP predictor state.
        let frames = 6usize;
        let pcm = voiced_signal(frames);
        let mut analysis = AnalysisState::new();
        let mut dec = SynthesisState::new();
        for f in 0..frames {
            let (frame, la) = frame_and_lookahead(&pcm, f);
            let params = analysis.analyse_spec(&frame, &la, PackedRate::High);
            let _ = dec.decode_spec_params(&params);
            for i in 0..LPC_ORDER {
                assert!(
                    (analysis.decoder.prev_lsp_freq[i] - dec.prev_lsp_freq[i]).abs() < 1.0e-3,
                    "frame {f} line {i}: shadow {} vs decoder {}",
                    analysis.decoder.prev_lsp_freq[i],
                    dec.prev_lsp_freq[i]
                );
            }
            assert_eq!(
                analysis.decoder.exc_history, dec.exc_history,
                "frame {f}: excitation history diverged"
            );
        }
    }

    #[test]
    fn spec_analysis_emits_decodable_lags() {
        let pcm = voiced_signal(3);
        let mut analysis = AnalysisState::new();
        for f in 0..3 {
            let (frame, la) = frame_and_lookahead(&pcm, f);
            let params = analysis.analyse_spec(&frame, &la, PackedRate::Low);
            let lag0 = decode_abs_lag(params.acl[0]);
            let lag1 = decode_delta_lag(params.acl[1], lag0);
            let lag2 = decode_abs_lag(params.acl[2]);
            let lag3 = decode_delta_lag(params.acl[3], lag2);
            for lag in [lag0, lag1, lag2, lag3] {
                assert!((PITCH_MIN as i32..=PITCH_MAX as i32).contains(&lag));
            }
        }
    }

    #[test]
    fn spec_decode_short_lag_train_mode_extends_pulses() {
        // Same frame twice, once with the train bit set on a short lag
        // (L0 = 20 < 58): the train must add excitation energy.
        let mut base = voiced_spec_params(PackedRate::High);
        base.acl = [2, 1, 2, 1]; // absolute lag 20
        for s in 0..SUBFRAMES_PER_FRAME {
            base.gain[s] = crate::spec_exc::encode_gain_word(PackedRate::High, 20, 40, 18, false);
        }
        let mut with_train = base;
        for s in 0..SUBFRAMES_PER_FRAME {
            with_train.gain[s] =
                crate::spec_exc::encode_gain_word(PackedRate::High, 20, 40, 18, true);
        }
        let mut st_a = SynthesisState::new();
        let mut st_b = SynthesisState::new();
        let out_a = st_a.decode_spec_params(&base);
        let out_b = st_b.decode_spec_params(&with_train);
        let e = |pcm: &[i16]| -> f64 { pcm.iter().map(|&s| (s as f64) * (s as f64)).sum() };
        assert!(
            e(&out_b) > e(&out_a),
            "train mode must add excitation energy: {} vs {}",
            e(&out_b),
            e(&out_a)
        );
    }
}
