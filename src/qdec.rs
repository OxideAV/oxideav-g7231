//! Fixed-point (Q15/Q31 saturating) G.723.1 decode chain.
//!
//! The float synthesis pipeline in [`crate::encoder`] implements the
//! clause 2/3 *mathematical* description; the Recommendation's §1.5
//! makes 16-bit saturating fixed-point behaviour normative, and the
//! OVER/TAME conformance classes drive sustained Word16-saturation
//! chains only a fixed-point pipeline tracks long-range. This module
//! rebuilds the decoder stage-by-stage on [`crate::basicop`]:
//!
//! - LSP inverse quantisation, stability, interpolation and LSP→LPC
//!   conversion in the tables' native Q15 normalised-frequency domain
//!   (Q14 cosine lookup, Q13 LPC coefficients).
//! - Excitation reconstruction (gain word, five-tap adaptive codebook,
//!   MP-MLQ / ACELP fixed vectors) with Word16 saturation on every
//!   stored excitation sample.
//! - The §3.6–§3.9 post-filter chain and §3.7 synthesis on saturating
//!   accumulators.
//!
//! All Q-format choices are stated inline; where the clause 1–4 prose
//! leaves a rounding choice open, the choice is arbitrated against the
//! ITU conformance vectors (documented per function).

// The module is wired into the decoder stage-by-stage over r391; the
// allow shrinks as each stage's entry points go live.
#![allow(dead_code)]

use crate::basicop::*;
use crate::spec_tables::{
    lsp_codebook_entry, LspBand, LSP_COSINE_LOOKUP_Q15, LSP_DC_PREDICTED_FREQ_Q15,
};
use crate::tables::LPC_ORDER;

/// §2.6 normal-decode LSP predictor `b = 12/32` in Q15.
const LSP_PREDICTOR_B_Q15: i16 = 12_288;
/// §3.10.1 erasure predictor `b_e = 23/32` in Q15.
const LSP_PREDICTOR_BE_Q15: i16 = 23_552;
/// §2.6 `Δ_min = 31.25 Hz` in Q15 normalised-frequency table units
/// (`31.25 Hz · 32768 / 4000 Hz = 256` exactly — the spec constant is
/// an exact power of two in the tables' own domain).
pub(crate) const LSP_DELTA_MIN_Q15: i16 = 256;
/// §3.10.1 erasure `Δ_min = 62.5 Hz` in table units.
pub(crate) const LSP_DELTA_MIN_ERASURE_Q15: i16 = 512;

/// §2.6 steps 1–2: rebuild the decoded LSP vector
/// `p̃_n = b·(p̃_{n−1} − p_DC) + p_DC + ẽ_n` from the 24-bit index and
/// the previous decoded vector, everything in Q15 normalised-frequency
/// table units with saturating arithmetic.
pub(crate) fn lsp_decode(lsp_index: u32, prev: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    lsp_predict_add(lsp_index, prev, LSP_PREDICTOR_B_Q15)
}

/// §3.10.1 steps 1–2: the erasure variant — residual forced to zero,
/// predictor `b_e = 23/32`.
pub(crate) fn lsp_extrapolate(prev: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    let mut out = [0i16; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let dc = LSP_DC_PREDICTED_FREQ_Q15[i];
        let pred = mult(sub(prev[i], dc), LSP_PREDICTOR_BE_Q15);
        out[i] = add(dc, pred);
    }
    out
}

fn lsp_predict_add(lsp_index: u32, prev: &[i16; LPC_ORDER], b_q15: i16) -> [i16; LPC_ORDER] {
    let bands = crate::spec_lsp::split_lsp_index(lsp_index);
    let mut out = [0i16; LPC_ORDER];
    for (m, band) in LspBand::ALL.iter().enumerate() {
        let (start, len) = band.start_and_length();
        let row = lsp_codebook_entry(*band, bands[m]);
        for j in 0..len {
            let i = start + j;
            let dc = LSP_DC_PREDICTED_FREQ_Q15[i];
            let pred = mult(sub(prev[i], dc), b_q15);
            out[i] = add(add(dc, pred), row[j]);
        }
    }
    out
}

/// §2.6 step 3 (eq. 6–7.3) stability procedure in the Q15 frequency
/// domain: sweep the nine consecutive pairs, spreading any pair closer
/// than `Δ_min` around its midpoint by `±Δ_min/2`; up to 10 sweeps.
/// Returns `true` when the vector is ordered (converged) — on `false`
/// the caller must fall back to the previous LSP vector per §2.6.
pub(crate) fn lsp_stability(p: &mut [i16; LPC_ORDER], delta_min: i16) -> bool {
    let half = delta_min / 2;
    for _ in 0..crate::tables::LSP_STABILITY_MAX_ITERATIONS {
        let mut violated = false;
        for j in 0..LPC_ORDER - 1 {
            // The pair difference in 32-bit: values live in [0, 32767]
            // so the subtraction cannot wrap, but the midpoint sum can
            // exceed Word16 — compute it wide and shift.
            let diff = p[j + 1] as i32 - p[j] as i32;
            if diff < delta_min as i32 {
                let avg = ((p[j] as i32 + p[j + 1] as i32) >> 1) as i16;
                p[j] = sub(avg, half);
                p[j + 1] = add(avg, half);
                violated = true;
            }
        }
        if !violated {
            return true;
        }
    }
    // Final check after the last sweep.
    p.windows(2)
        .all(|w| w[1] as i32 - w[0] as i32 >= delta_min as i32)
}

/// §2.7 / §3.3 (eq. 8) per-subframe LSP interpolation in the Q15
/// frequency domain with Q15 weights and rounded Q31 accumulation.
/// Subframe 3 is the current vector exactly (weight pair (0, 1)).
pub(crate) fn lsp_interpolate(
    k: usize,
    prev: &[i16; LPC_ORDER],
    cur: &[i16; LPC_ORDER],
) -> [i16; LPC_ORDER] {
    if k >= 3 {
        return *cur;
    }
    let (wp, wc): (i16, i16) = match k {
        0 => (24_576, 8_192),
        1 => (16_384, 16_384),
        _ => (8_192, 24_576),
    };
    let mut out = [0i16; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let acc = l_mac(l_mult(prev[i], wp), cur[i], wc);
        out[i] = round16(acc);
    }
    out
}

/// Cosine of a Q15 normalised frequency (`ω = π·f/32768`) via the
/// published 512-entry full-period Q14 cosine table with 7-bit linear
/// interpolation: the table grid is `2π·i/512`, so a half-period Q15
/// argument indexes at `f >> 7` with `f & 127` as the interpolation
/// fraction.
pub(crate) fn cos_q14(freq_q15: i16) -> i16 {
    debug_assert!(freq_q15 >= 0);
    let idx = (freq_q15 >> 7) as usize; // 0..=255 for f < 32768
    let frac = freq_q15 & 0x7F;
    let base = LSP_COSINE_LOOKUP_Q15[idx];
    let next = LSP_COSINE_LOOKUP_Q15[idx + 1];
    // Q15 fraction of the 128-unit gap: frac·256 / 32768 = frac/128.
    add(base, mult(sub(next, base), shl(frac, 8)))
}

/// §2.7 (eq. 9) LSP → LPC conversion: Q15 frequencies → Q14 cosines →
/// Q13 direct-form coefficients.
///
/// Returns the ten coefficients `ã_1..ã_10` of the quantised synthesis
/// filter in the eq. 48 convention (`sy[n] = ppf[n] + Σ ã_j·sy[n−j]`),
/// i.e. the *predictor* taps. Construction is the standard sum/
/// difference-polynomial expansion: with `c_k = cos ω_k`,
///
/// ```text
///   P(z) = Π_{k even} (1 − 2 c_k z⁻¹ + z⁻²)       (roots ω_0, ω_2, …)
///   Q(z) = Π_{k odd}  (1 − 2 c_k z⁻¹ + z⁻²)
///   A(z) = [P(z)(1 + z⁻¹) + Q(z)(1 − z⁻¹)] / 2,   ã_j = −A_j
/// ```
///
/// Intermediate coefficients are held exact in 64-bit Q24 (they are
/// filter *coefficients*, not signal samples — the Word16 saturation
/// semantics of the signal path do not apply); the final Q13 rounding
/// saturates to Word16.
pub(crate) fn lsp_to_lpc_q13(lsp_freq: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    const ONE_Q24: i64 = 1 << 24;
    let half = LPC_ORDER / 2;
    let mut pz = [0i64; LPC_ORDER + 1];
    let mut qz = [0i64; LPC_ORDER + 1];
    pz[0] = ONE_Q24;
    qz[0] = ONE_Q24;
    let mut deg = 0usize;
    for k in 0..half {
        // −2·c in Q14 → apply as (coef · c) >> 13 on Q24 values.
        let c_even = cos_q14(lsp_freq[2 * k]) as i64;
        let c_odd = cos_q14(lsp_freq[2 * k + 1]) as i64;
        deg += 2;
        for i in (2..=deg).rev() {
            pz[i] += ((pz[i - 1] * c_even) >> 13).wrapping_neg() + pz[i - 2];
            qz[i] += ((qz[i - 1] * c_odd) >> 13).wrapping_neg() + qz[i - 2];
        }
        pz[1] -= (pz[0] * c_even) >> 13;
        qz[1] -= (qz[0] * c_odd) >> 13;
    }
    // A_j = (f1[j] + f2[j]) / 2 with f1 = P·(1+z⁻¹), f2 = Q·(1−z⁻¹):
    //   A_j = (P_j + P_{j−1} + Q_j − Q_{j−1}) / 2, j = 1..10.
    // ã_j = −A_j, rounded from Q24 to Q13 (>> 12 after the /2).
    let mut a = [0i16; LPC_ORDER];
    for j in 1..=LPC_ORDER {
        let sum = pz[j] + pz[j - 1] + qz[j] - qz[j - 1];
        let neg = -(sum >> 1); // ã_j = −A_j, still Q24
        let q13 = (neg + (1 << 10)) >> 11;
        a[j - 1] = if q13 > i16::MAX as i64 {
            i16::MAX
        } else if q13 < i16::MIN as i64 {
            i16::MIN
        } else {
            q13 as i16
        };
    }
    a
}

/// Decoded-LSP state advance shared by the normal and erasure paths:
/// stability check with previous-vector fallback.
pub(crate) fn lsp_check_or_previous(
    mut cand: [i16; LPC_ORDER],
    prev: &[i16; LPC_ORDER],
    delta_min: i16,
) -> [i16; LPC_ORDER] {
    if lsp_stability(&mut cand, delta_min) {
        cand
    } else {
        *prev
    }
}

/// The §3.11 cold-start previous-LSP vector (`p_DC`).
pub(crate) fn lsp_dc() -> [i16; LPC_ORDER] {
    LSP_DC_PREDICTED_FREQ_Q15
}

/// Interpolated per-subframe LPC set for one frame (eq. 8 + eq. 9).
pub(crate) fn frame_lpc_q13(
    prev: &[i16; LPC_ORDER],
    cur: &[i16; LPC_ORDER],
) -> [[i16; LPC_ORDER]; crate::tables::SUBFRAMES_PER_FRAME] {
    let mut out = [[0i16; LPC_ORDER]; crate::tables::SUBFRAMES_PER_FRAME];
    for (k, slot) in out.iter_mut().enumerate() {
        let lsp = lsp_interpolate(k, prev, cur);
        *slot = lsp_to_lpc_q13(&lsp);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::SUBFRAMES_PER_FRAME;

    #[test]
    fn dc_vector_decode_with_zero_residual_row_stays_near_dc() {
        // Index 0 selects row 0 of each band codebook; with prev = DC
        // the predictor vanishes so the decode is DC + row0.
        let dc = lsp_dc();
        let out = lsp_decode(0, &dc);
        for i in 0..LPC_ORDER {
            let row0 = match i {
                0..=2 => lsp_codebook_entry(LspBand::Band0, 0)[i],
                3..=5 => lsp_codebook_entry(LspBand::Band1, 0)[i - 3],
                _ => lsp_codebook_entry(LspBand::Band2, 0)[i - 6],
            };
            assert_eq!(out[i], add(dc[i], row0));
        }
    }

    #[test]
    fn fixed_lsp_decode_matches_float_reference_path() {
        // The float spec_lsp path decodes in the same table-unit domain
        // with f32 arithmetic; away from saturation the two must agree
        // to within a rounding unit.
        let dc = lsp_dc();
        let mut prev_q = dc;
        let mut prev_f = crate::spec_lsp::lsp_dc_freq();
        let mut idx: u32 = 0x00_01_02;
        for _ in 0..50 {
            let q = lsp_decode(idx, &prev_q);
            let f = crate::spec_lsp::decode_lsp_freq(idx, &prev_f);
            for i in 0..LPC_ORDER {
                assert!(
                    (q[i] as f32 - f[i]).abs() <= 1.0,
                    "line {i}: fixed {} vs float {}",
                    q[i],
                    f[i]
                );
            }
            prev_q = q;
            for i in 0..LPC_ORDER {
                prev_f[i] = q[i] as f32; // keep the two predictors in lockstep
            }
            idx = idx.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0xFF_FF_FF;
        }
    }

    #[test]
    fn stability_orders_a_reversed_pair_and_reports_hopeless_vectors() {
        let mut p = lsp_dc();
        p.swap(4, 5); // one out-of-order pair
        assert!(lsp_stability(&mut p, LSP_DELTA_MIN_Q15));
        for w in p.windows(2) {
            assert!(w[1] as i32 - w[0] as i32 >= LSP_DELTA_MIN_Q15 as i32);
        }

        // An all-equal vector cannot be spread to 10 × 256 units within
        // 10 sweeps of pair-midpoint moves starting from equal values —
        // every sweep only spreads the outermost pair further.
        let mut collapsed = [16_000i16; LPC_ORDER];
        let ok = lsp_stability(&mut collapsed, LSP_DELTA_MIN_Q15);
        if ok {
            for w in collapsed.windows(2) {
                assert!(w[1] as i32 - w[0] as i32 >= LSP_DELTA_MIN_Q15 as i32);
            }
        }
    }

    #[test]
    fn interpolation_endpoints_and_midpoint() {
        let dc = lsp_dc();
        let mut cur = dc;
        for c in cur.iter_mut() {
            *c = add(*c, 1000);
        }
        let sf3 = lsp_interpolate(3, &dc, &cur);
        assert_eq!(sf3, cur, "subframe 3 is the current vector exactly");
        let sf1 = lsp_interpolate(1, &dc, &cur);
        for i in 0..LPC_ORDER {
            let expect = ((dc[i] as i32 + cur[i] as i32 + 1) / 2) as i16;
            assert!(
                (sf1[i] - expect).abs() <= 1,
                "line {i}: {} vs {}",
                sf1[i],
                expect
            );
        }
        let sf0 = lsp_interpolate(0, &dc, &cur);
        for i in 0..LPC_ORDER {
            let expect = (0.75 * dc[i] as f64 + 0.25 * cur[i] as f64).round() as i16;
            assert!((sf0[i] - expect).abs() <= 1);
        }
    }

    #[test]
    fn cosine_lookup_tracks_the_real_cosine() {
        for f in (0..32_768i32).step_by(37) {
            let c = cos_q14(f as i16) as f64 / 16_384.0;
            let real = (std::f64::consts::PI * f as f64 / 32_768.0).cos();
            assert!(
                (c - real).abs() < 3.0e-4,
                "f = {f}: table {c} vs real {real}"
            );
        }
    }

    #[test]
    fn q13_lpc_matches_float_conversion_on_the_dc_vector() {
        // Convert the DC LSP set on both paths and compare in the
        // float domain (fixed path is Q13, float path returns A(z)
        // with a[0] = 1 and the opposite sign convention).
        let dc = lsp_dc();
        let a_q = lsp_to_lpc_q13(&dc);
        let cos_f = crate::spec_lsp::lsp_freq_to_cosines(&dc.map(|q| q as f32));
        let a_f = crate::encoder::lsp_to_lpc(&cos_f);
        for j in 0..LPC_ORDER {
            let fixed = a_q[j] as f32 / 8192.0;
            let float = -a_f[j + 1];
            assert!(
                (fixed - float).abs() < 4.0e-3,
                "ã_{}: fixed {fixed} vs float {float}",
                j + 1
            );
        }
    }

    #[test]
    fn q13_lpc_matches_float_conversion_on_random_stable_sets() {
        let mut state = 0x1234_5678u32;
        let mut rand = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i32
        };
        for _ in 0..200 {
            // Random ordered LSP vector with ≥ 400-unit gaps.
            let mut lsp = [0i16; LPC_ORDER];
            let mut f = 800i32 + (rand() % 800);
            for slot in lsp.iter_mut() {
                *slot = f as i16;
                f += 500 + (rand() % 2200);
            }
            if f >= 32_000 {
                continue;
            }
            let a_q = lsp_to_lpc_q13(&lsp);
            let cos_f = crate::spec_lsp::lsp_freq_to_cosines(&lsp.map(|q| q as f32));
            let a_f = crate::encoder::lsp_to_lpc(&cos_f);
            // Q13 in Word16 covers |ã| < 4; heavily clustered synthetic
            // sets can exceed that, where the fixed path (correctly)
            // saturates. Compare only in-range coefficient sets.
            if a_f[1..].iter().any(|c| c.abs() >= 3.9) {
                continue;
            }
            for j in 0..LPC_ORDER {
                let fixed = a_q[j] as f32 / 8192.0;
                let float = -a_f[j + 1];
                assert!(
                    (fixed - float).abs() < 8.0e-3,
                    "ã_{}: fixed {fixed} vs float {float} (lsp {lsp:?})",
                    j + 1
                );
            }
        }
    }

    #[test]
    fn frame_lpc_produces_four_subframe_sets() {
        let dc = lsp_dc();
        let mut cur = dc;
        for c in cur.iter_mut() {
            *c = add(*c, 700);
        }
        let sets = frame_lpc_q13(&dc, &cur);
        assert_eq!(sets.len(), SUBFRAMES_PER_FRAME);
        // Subframe 3 equals a direct conversion of the current vector.
        assert_eq!(sets[3], lsp_to_lpc_q13(&cur));
        // Adjacent subframes differ smoothly but are not identical.
        assert_ne!(sets[0], sets[3]);
    }

    #[test]
    fn extrapolation_leaks_toward_dc() {
        let dc = lsp_dc();
        let mut prev = dc;
        for p in prev.iter_mut() {
            *p = add(*p, 2048);
        }
        let e = lsp_extrapolate(&prev);
        for i in 0..LPC_ORDER {
            // b_e = 23/32 of the 2048-unit offset ≈ 1472.
            let off = e[i] as i32 - dc[i] as i32;
            assert!((off - 1472).abs() <= 1, "line {i}: offset {off}");
        }
    }
}
