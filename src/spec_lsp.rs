//! ITU-T G.723.1 §2.5 / §2.6 predictive split vector quantiser for the
//! LSP coefficients, running on the published spec tables.
//!
//! # Domain
//!
//! All vectors in this module live in the spec tables' native domain:
//! Q15 *normalised frequency*, i.e. a value `q` represents the LSP
//! angular frequency `ω = π · q / 32768` (32768 ↔ Nyquist). The DC
//! vector `p_DC` ([`crate::spec_tables::LSP_DC_PREDICTED_FREQ_Q15`])
//! and the three split residual codebooks
//! ([`crate::spec_tables::LSP_CODEBOOK_BAND0_Q13`] …) share this scale:
//! adding a codebook residual to the DC vector directly yields a
//! well-ordered LSP set (the alternative reading — residuals at 4× the
//! DC scale — produces negative frequencies on the very first rows).
//! Values are carried as `f32` in table units (0..32768).
//!
//! # Codec shape (§2.5 steps 1–5, §2.6 steps 1–3)
//!
//! - The long-term DC component `p_DC` is removed (eq. 4.3 inverse).
//! - A first-order fixed predictor `b = 12/32` applied to the
//!   *previously decoded* LSP vector `p̃_{n−1}` forms the predicted
//!   vector `p̄_n = b · (p̃_{n−1} − p_DC)` (eq. 3.3).
//! - The residual `e_n = p_n − p̄_n` (eq. 3.4) is split 3 + 3 + 4
//!   (eq. 4.1) and each sub-vector is quantised against its 256-entry
//!   codebook under the weighted error criterion of eq. 4.5, with the
//!   diagonal weights of eq. 5 (inverse distance to the nearest
//!   neighbouring *unquantised* LSP).
//! - Decode (§2.6) adds the selected codebook rows back onto
//!   `p̄_n + p_DC` (eq. 4.4). The §2.6 stability procedure (eq. 6–7.3)
//!   runs downstream in the synthesis pipeline's cosine domain
//!   (`crate::encoder::enforce_lsp_stability`).
//!
//! The 24-bit `LPC` index packs the three 8-bit band indices with band
//! 0 in the least-significant byte (a documented crate convention; the
//! Recommendation transmits `LPC` as one opaque 24-bit parameter).

use crate::spec_tables::{lsp_codebook_entry, LspBand, LSP_DC_PREDICTED_FREQ_Q15};
use crate::tables::{LPC_ORDER, LSP_PREDICTOR_B};

/// The long-term LSP DC vector `p_DC` in table units (Q15 normalised
/// frequency as `f32`).
pub(crate) fn lsp_dc_freq() -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for (o, &q) in out.iter_mut().zip(LSP_DC_PREDICTED_FREQ_Q15.iter()) {
        *o = q as f32;
    }
    out
}

/// Convert a table-unit LSP vector to the synthesis pipeline's cosine
/// domain (`cos ω`, descending for an ascending frequency set).
pub(crate) fn lsp_freq_to_cosines(freq: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for (o, &f) in out.iter_mut().zip(freq.iter()) {
        *o = (std::f32::consts::PI * f / 32_768.0).cos();
    }
    out
}

/// Convert a cosine-domain LSP vector back to table units.
pub(crate) fn lsp_cosines_to_freq(cos: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for (o, &c) in out.iter_mut().zip(cos.iter()) {
        *o = c.clamp(-1.0, 1.0).acos() * 32_768.0 / std::f32::consts::PI;
    }
    out
}

/// Split the 24-bit `LPC` parameter into the three band indices
/// (band 0 in the least-significant byte).
pub(crate) fn split_lsp_index(lsp_index: u32) -> [u8; 3] {
    [
        (lsp_index & 0xFF) as u8,
        ((lsp_index >> 8) & 0xFF) as u8,
        ((lsp_index >> 16) & 0xFF) as u8,
    ]
}

/// Combine three band indices into the 24-bit `LPC` parameter.
pub(crate) fn combine_lsp_index(bands: [u8; 3]) -> u32 {
    bands[0] as u32 | (bands[1] as u32) << 8 | (bands[2] as u32) << 16
}

/// §2.6 LSP decode (steps 1–2): rebuild the decoded LSP vector
/// `p̃_n = p̄_n + p_DC + ẽ_n` from the 24-bit index and the previously
/// decoded vector, all in table units. The §2.6 step-3 stability check
/// is applied by the caller in the cosine domain.
pub(crate) fn decode_lsp_freq(lsp_index: u32, prev: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let bands = split_lsp_index(lsp_index);
    let dc = lsp_dc_freq();
    let mut out = [0.0f32; LPC_ORDER];
    for (m, band) in LspBand::ALL.iter().enumerate() {
        let (start, len) = band.start_and_length();
        let row = lsp_codebook_entry(*band, bands[m]);
        for j in 0..len {
            let i = start + j;
            let predicted = LSP_PREDICTOR_B * (prev[i] - dc[i]);
            out[i] = predicted + dc[i] + row[j] as f32;
        }
    }
    out
}

/// Diagonal weighting matrix of eq. 5, computed from the *unquantised*
/// LSP vector in table units: each weight is the inverse distance to
/// the nearest neighbouring LSP (end weights use their single
/// neighbour). Degenerate (≤ 0) gaps clamp to one table unit so the
/// weight stays finite and large.
fn lsp_weights(p_unq: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let mut w = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        let lower = if j > 0 {
            p_unq[j] - p_unq[j - 1]
        } else {
            f32::INFINITY
        };
        let upper = if j + 1 < LPC_ORDER {
            p_unq[j + 1] - p_unq[j]
        } else {
            f32::INFINITY
        };
        let gap = lower.min(upper).max(1.0);
        w[j] = 1.0 / gap;
    }
    w
}

/// §2.5 LSP quantisation (steps 2–5): weighted split-VQ search over the
/// three 256-entry residual codebooks. `p_unq` is the unquantised LSP
/// vector and `prev` the previously *decoded* vector, both in table
/// units. Returns the 24-bit `LPC` index and the decoded vector the
/// index reproduces (before the stability procedure).
pub(crate) fn quantise_lsp_freq(
    p_unq: &[f32; LPC_ORDER],
    prev: &[f32; LPC_ORDER],
) -> (u32, [f32; LPC_ORDER]) {
    let dc = lsp_dc_freq();
    let w = lsp_weights(p_unq);

    // Residual target e_n = p_n − p̄_n with p_n = p′ − p_DC (eq. 3.4 /
    // 4.3): equivalently p′ − p_DC − b·(p̃_{n−1} − p_DC).
    let mut e_target = [0.0f32; LPC_ORDER];
    for i in 0..LPC_ORDER {
        e_target[i] = p_unq[i] - dc[i] - LSP_PREDICTOR_B * (prev[i] - dc[i]);
    }

    let mut bands = [0u8; 3];
    let mut decoded = [0.0f32; LPC_ORDER];
    for (m, band) in LspBand::ALL.iter().enumerate() {
        let (start, len) = band.start_and_length();
        let mut best_idx = 0u8;
        let mut best_err = f32::INFINITY;
        for idx in 0..=u8::MAX {
            let row = lsp_codebook_entry(*band, idx);
            let mut err = 0.0f32;
            for j in 0..len {
                let d = e_target[start + j] - row[j] as f32;
                err += w[start + j] * d * d;
            }
            if err < best_err {
                best_err = err;
                best_idx = idx;
            }
        }
        bands[m] = best_idx;
        let row = lsp_codebook_entry(*band, best_idx);
        for j in 0..len {
            let i = start + j;
            decoded[i] = LSP_PREDICTOR_B * (prev[i] - dc[i]) + dc[i] + row[j] as f32;
        }
    }
    (combine_lsp_index(bands), decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32 as f32) / (u32::MAX as f32)
        }
    }

    /// A random well-ordered LSP vector in table units, biased around
    /// the DC vector (± up to ~1500 units ≈ 366 Hz per line).
    fn random_stable_lsp(rng: &mut Lcg) -> [f32; LPC_ORDER] {
        let dc = lsp_dc_freq();
        loop {
            let mut p = [0.0f32; LPC_ORDER];
            for i in 0..LPC_ORDER {
                p[i] = dc[i] + (rng.next_f32() - 0.5) * 3000.0;
            }
            let ordered = p.windows(2).all(|w| w[1] - w[0] > 128.0);
            if ordered && p[0] > 256.0 && p[LPC_ORDER - 1] < 32_500.0 {
                return p;
            }
        }
    }

    #[test]
    fn index_split_combine_round_trip() {
        for idx in [0u32, 0xFF_FF_FF, 0x01_02_03, 0xAB_CD_EF, 0x80_00_01] {
            assert_eq!(combine_lsp_index(split_lsp_index(idx)), idx);
        }
    }

    #[test]
    fn dc_vector_is_ascending_and_within_range() {
        let dc = lsp_dc_freq();
        for w in dc.windows(2) {
            assert!(w[1] > w[0]);
        }
        assert!(dc[0] > 0.0 && dc[LPC_ORDER - 1] < 32_768.0);
    }

    #[test]
    fn freq_cosine_conversion_round_trips() {
        let dc = lsp_dc_freq();
        let cos = lsp_freq_to_cosines(&dc);
        // Ascending frequencies ⇒ strictly descending cosines.
        for w in cos.windows(2) {
            assert!(w[1] < w[0]);
        }
        let back = lsp_cosines_to_freq(&cos);
        for i in 0..LPC_ORDER {
            assert!(
                (back[i] - dc[i]).abs() < 1.0,
                "line {i}: {} vs {}",
                back[i],
                dc[i]
            );
        }
    }

    #[test]
    fn decode_matches_manual_reconstruction() {
        let prev = lsp_dc_freq();
        let idx = combine_lsp_index([7, 200, 33]);
        let out = decode_lsp_freq(idx, &prev);
        // With prev = DC the predictor term vanishes: p̃ = p_DC + row.
        let dc = lsp_dc_freq();
        let b0 = lsp_codebook_entry(LspBand::Band0, 7);
        let b1 = lsp_codebook_entry(LspBand::Band1, 200);
        let b2 = lsp_codebook_entry(LspBand::Band2, 33);
        for j in 0..3 {
            assert_eq!(out[j], dc[j] + b0[j] as f32);
            assert_eq!(out[3 + j], dc[3 + j] + b1[j] as f32);
        }
        for j in 0..4 {
            assert_eq!(out[6 + j], dc[6 + j] + b2[j] as f32);
        }
    }

    #[test]
    fn quantising_the_dc_vector_returns_a_near_dc_decode() {
        let dc = lsp_dc_freq();
        let (idx, decoded) = quantise_lsp_freq(&dc, &dc);
        // The residual target is exactly zero; the codebooks contain
        // near-zero rows (band 0 row 0 is all-zero), so the decode must
        // stay very close to DC.
        let redecoded = decode_lsp_freq(idx, &dc);
        for i in 0..LPC_ORDER {
            assert_eq!(decoded[i], redecoded[i]);
            assert!(
                (decoded[i] - dc[i]).abs() < 200.0,
                "line {i} strayed {} units from DC",
                decoded[i] - dc[i]
            );
        }
    }

    #[test]
    fn quantise_decode_round_trip_is_consistent_and_accurate() {
        let mut rng = Lcg(0xfeed_beef_cafe_f00d);
        let mut prev = lsp_dc_freq();
        let mut worst = 0.0f32;
        for _ in 0..200 {
            let p = random_stable_lsp(&mut rng);
            let (idx, decoded) = quantise_lsp_freq(&p, &prev);
            let redecoded = decode_lsp_freq(idx, &prev);
            for i in 0..LPC_ORDER {
                assert_eq!(decoded[i], redecoded[i], "index does not reproduce decode");
                worst = worst.max((decoded[i] - p[i]).abs());
            }
            prev = decoded;
        }
        // 24-bit split VQ on a ±1500-unit neighbourhood: every decoded
        // line stays within 1024 table units (250 Hz) of the input.
        assert!(worst < 1024.0, "worst per-line error {worst} table units");
    }

    #[test]
    fn predictor_tracks_a_held_vector_in_aggregate() {
        // Quantising the same vector twice with the predictor warmed on
        // the first decode shrinks the residual target by a factor
        // 1 − b toward the codebooks' dense near-zero region, so the
        // *aggregate* error across many held vectors must not grow.
        // (Per-vector regressions are possible: the residual codebooks
        // are trained on speech statistics, not on this synthetic
        // distribution.)
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let dc = lsp_dc_freq();
        let (mut cold_total, mut warm_total) = (0.0f64, 0.0f64);
        for _ in 0..50 {
            let p = random_stable_lsp(&mut rng);
            let (_, d1) = quantise_lsp_freq(&p, &dc);
            let e1: f32 = p.iter().zip(d1.iter()).map(|(a, b)| (a - b).abs()).sum();
            let (_, d2) = quantise_lsp_freq(&p, &d1);
            let e2: f32 = p.iter().zip(d2.iter()).map(|(a, b)| (a - b).abs()).sum();
            cold_total += e1 as f64;
            warm_total += e2 as f64;
        }
        assert!(
            warm_total <= cold_total * 1.02,
            "warmed predictor grew aggregate error: cold {cold_total:.1}, warm {warm_total:.1}"
        );
    }
}
