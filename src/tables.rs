//! Codebook / layout constants for the G.723.1 decoder.
//!
//! Only dimension and layout tables are included in this scaffold — the
//! full LSP-VQ, adaptive-pitch, and fixed-codebook values (tables 3–12 in
//! ITU-T G.723.1) will land with the synthesis path in a follow-up.
//!
//! References in comments point to the May 2006 ITU-T Recommendation.

/// Sampling rate of the encoded signal — fixed at 8 kHz.
pub const SAMPLE_RATE_HZ: u32 = 8_000;

/// Frame duration: 30 ms = 240 samples at 8 kHz.
pub const FRAME_SIZE_SAMPLES: usize = 240;

/// A frame is split into four 60-sample subframes (§3.2).
pub const SUBFRAME_SIZE: usize = 60;
/// Number of subframes per frame.
pub const SUBFRAMES_PER_FRAME: usize = 4;

/// Order of the LPC synthesis / perceptual-weighting filters (§3.3).
pub const LPC_ORDER: usize = 10;

/// LPC-analysis window length in samples (§3.3 Figure 3).
pub const LPC_WINDOW: usize = 180;

/// Pitch-search range — open-loop pitch estimate lies in [PITCH_MIN..=PITCH_MAX].
pub const PITCH_MIN: usize = 18;
pub const PITCH_MAX: usize = 142;

/// High-rate (6.3 kbit/s, MP-MLQ) frame size in bytes (§5.4, Table 6).
pub const HIGH_RATE_BYTES: usize = 24;
/// Total bits per 6.3 kbit/s frame.
pub const HIGH_RATE_BITS: u32 = 189;

/// Low-rate (5.3 kbit/s, ACELP) frame size in bytes.
pub const LOW_RATE_BYTES: usize = 20;
/// Total bits per 5.3 kbit/s frame.
pub const LOW_RATE_BITS: u32 = 158;

/// SID (silence-insertion descriptor) frame size in bytes — §Annex A.
pub const SID_BYTES: usize = 4;

/// Number of 8-bit predictor sub-codebooks that together form the 24-bit
/// LSP VQ index (§3.4, tables 3–5).
pub const LSP_VQ_SUBCODEBOOKS: usize = 3;
/// Entries per LSP sub-codebook.
pub const LSP_VQ_ENTRIES: usize = 256;

/// Perceptual-weighting filter γ₁/γ₂ constants (§3.5).
pub const PERCEPTUAL_GAMMA1: f32 = 0.9;
pub const PERCEPTUAL_GAMMA2: f32 = 0.5;

/// Post-filter constants (§3.9).
pub const POSTFILTER_GAMMA1: f32 = 0.65;
pub const POSTFILTER_GAMMA2: f32 = 0.75;
pub const POSTFILTER_TILT: f32 = 0.25;

/// Long-term (pitch) post-filter LTP weighting γ_ltp for the high rate
/// (6.3 kbit/s, MP-MLQ) per G.723.1 §3.6 (clause 3.6). Multiplies the
/// forward/backward LTP contribution in eq. 42 of the 1996 base edition.
pub const POSTFILTER_LTP_GAMMA_HIGH: f32 = 0.1875;
/// Long-term (pitch) post-filter LTP weighting γ_ltp for the low rate
/// (5.3 kbit/s, ACELP) per G.723.1 §3.6 (clause 3.6).
pub const POSTFILTER_LTP_GAMMA_LOW: f32 = 0.25;

/// Minimum LSP angular-frequency separation `Δ_min` for the *normal* decode
/// path (G.723.1 §3.1 / 2.6, eq. 6–7.3). The 1996 base edition specifies
/// `Δ_min = 31.25 Hz` between consecutive decoded LSP frequencies; the
/// stability check spreads any out-of-order pair around its midpoint by
/// `±Δ_min/2`. Constant is the spec frequency itself; the algorithm in
/// [`crate::encoder`] converts it to a normalised angular-frequency gap
/// `Δ_min · 2π / SAMPLE_RATE_HZ`.
pub const LSP_STABILITY_DELTA_MIN_HZ: f32 = 31.25;

/// Minimum LSP separation for the *erasure* concealment path
/// (G.723.1 §3.10.1). The wider 62.5 Hz value relaxes the constraint so
/// the same iterative ordering procedure can re-stabilise an extrapolated
/// LSP whose pairs have drifted further from the previous decoded vector.
pub const LSP_STABILITY_DELTA_MIN_ERASURE_HZ: f32 = 62.5;

/// Maximum number of iterations the LSP stability procedure runs before
/// giving up on the decoded vector and falling back to the previous good
/// LSP (G.723.1 §3.1 / 2.6, "iterate up to 10 times").
pub const LSP_STABILITY_MAX_ITERATIONS: u32 = 10;
/// Minimum prediction-gain threshold (dB) below which the pitch
/// post-filter is bypassed for a subframe (G.723.1 §3.6, eq. 45–46
/// gate clause).
pub const POSTFILTER_LTP_PRED_GAIN_DB_MIN: f32 = 1.25;
/// Half-width of the forward/backward lag search window around the
/// reference pitch lag (G.723.1 §3.6 eq. 43.1–43.2: `M_f ∈ [L − 3, L + 3]`).
pub const POSTFILTER_LTP_SEARCH_RADIUS: i32 = 3;

/// Length in samples of the post-filtered PCM history used by the
/// frame-erasure voiced/unvoiced classifier (G.723.1 §3.10.2). The spec
/// cross-correlates the last 120 samples (= two 60-sample subframes) of
/// the decoder's output with a lag-shifted copy around `L_2 ± 3`.
pub const ERASURE_CLASSIFIER_HISTORY_LEN: usize = 120;

/// Half-width of the lag search around `L_2` for the §3.10.2 voiced
/// classifier — the spec cross-correlation runs over `L_2 ± 3`.
pub const ERASURE_CLASSIFIER_LAG_RADIUS: i32 = 3;

/// Voiced / unvoiced prediction-gain threshold for the frame-erasure
/// classifier (G.723.1 §3.10.2). Above 0.58 dB the trailing 120 samples
/// are deemed voiced and concealment regenerates a periodic excitation
/// at the classifier's pitch period; below the threshold the frame is
/// unvoiced and concealment regenerates a uniform-random excitation
/// scaled by the saved average gain.
pub const ERASURE_VOICED_THRESHOLD_DB: f32 = 0.58;

/// Per-erased-frame attenuation in dB applied during sustained erasure
/// (G.723.1 §3.10.2). The spec attenuates the regenerated vector by an
/// extra 2.5 dB per consecutive erased frame.
pub const ERASURE_ATTENUATION_DB_PER_FRAME: f32 = 2.5;

/// Number of consecutive interpolated frames after which the concealed
/// output is fully muted (G.723.1 §3.10.2: "mute completely after 3
/// interpolated frames"). Erasure runs longer than this emit silence.
pub const ERASURE_MUTE_AFTER_FRAMES: u32 = 3;

/// Bit-layout (high-rate, MP-MLQ) — field widths in bits, in packing order.
///
/// Derived from ITU-T G.723.1 Annex B Table B.1. The implementation below
/// only uses these for consistency checks; per-field semantics live in the
/// synthesis module once implemented.
pub const HIGH_RATE_FIELD_WIDTHS: &[u32] = &[
    2, // LPC0 (upper sub-codebook index)
    8, // LPC1
    8, // LPC2
    8, // ACL0 (adaptive-codebook lag subframe 0)
    5, // ACL1
    5, // ACL2
    5, // ACL3
    2, // GAIN0 (combined gain index subframe 0, high rate)
    12, 12, 12, 12, // combined GAINs
    1, 1, 1, 1, // grid bits
    1, // reserved
    13, 16, 14, 14, // MP-MLQ pulse positions / signs subframes 0..3
    6, 5, 5, 5, // pulse-position LSBs
];

/// Bit-layout (low-rate, ACELP) — field widths in bits, in packing order.
pub const LOW_RATE_FIELD_WIDTHS: &[u32] = &[
    2, // LPC0
    8, // LPC1
    8, // LPC2
    7, // ACL0
    2, // ACL1
    7, // ACL2
    2, // ACL3
    12, 12, 12, 12, // combined GAINs
    1, 1, 1, 1, // grid bits
    12, 12, 12, 12, // ACELP fixed-codebook pulses (subframes 0..3)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_four_subframes() {
        assert_eq!(SUBFRAME_SIZE * SUBFRAMES_PER_FRAME, FRAME_SIZE_SAMPLES);
    }

    #[test]
    fn total_bits_match_byte_sizes() {
        // Note: published bit totals are 189 / 158; byte framing rounds up
        // with unused bits at the end of the last byte.
        assert!(HIGH_RATE_BITS as usize <= HIGH_RATE_BYTES * 8);
        assert!(LOW_RATE_BITS as usize <= LOW_RATE_BYTES * 8);
    }
}
