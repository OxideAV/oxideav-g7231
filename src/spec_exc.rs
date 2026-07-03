//! ITU-T G.723.1 excitation-parameter codecs on the published tables:
//! the combined 12-bit gain word (§2.17 eq. 36, §2.18 eq. 39–40), the
//! fifth-order adaptive-codebook contribution (§2.18 eq. 41.1–41.2),
//! and the rate-specific fixed-codebook pulse reconstruction
//! (§2.15 / §2.16 / §2.17).
//!
//! # Scales
//!
//! The crate's synthesis pipeline runs on `f32` PCM normalised to
//! `[-1, 1]` (i16 / 32768). The spec tables map into that domain as:
//!
//! - Adaptive-codebook gain rows: each of the 85 / 170 rows carries the
//!   five predictor taps `β_i0..β_i4` in Q13 in its first five entries
//!   (`tap = q / 8192`). The remaining 15 entries are the precomputed
//!   closed-loop search energies `−2·β_i²` and `−2·β_i·β_j` (>> 15) —
//!   verified numerically against the taps — and are not needed for
//!   decoding.
//! - Fixed-codebook gain: the 24-step logarithmic table (§2.15,
//!   3.2 dB/step) holds amplitudes in i16 sample units; dividing by
//!   32768 yields the normalised-domain gain.
//! - 1-tap LTP shortcut (§2.16): β in Q15 (`/ 32768`), ε an offset in
//!   `{-2..2}` selecting one of the 5-tap lags; rows whose published
//!   gain is zero disable the enhancement (their selector is the
//!   sentinel 60).

use crate::linepack::PackedRate;
use crate::spec_tables::{
    fcbk_unpk_positions, pitch_1tap_ltp, ADAPTIVE_CODEBOOK_GAIN_5P3, ADAPTIVE_CODEBOOK_GAIN_6P3,
    ADAPTIVE_CODEBOOK_ROW_DIM, FIXED_CODEBOOK_GAIN_Q15, GAIN_TABLE_SIZE,
};
use crate::tables::{PITCH_MAX, PITCH_MIN, SUBFRAME_SIZE};

/// Number of taps of the pitch predictor (§2.14: "A fifth order pitch
/// predictor is used").
pub(crate) const ACB_TAPS: usize = 5;

/// Rows of the shared 170-entry gain-vector codebook (§2.14: "170
/// entries for the low bit rate … The 170 entry codebook is the same
/// for both rates").
const GAIN_ROWS_170: usize = 170;
/// Rows of the 85-entry codebook used by the high rate when the
/// subframe pair's reference lag is short (§2.14: "For the high rate if
/// L0 is less than 58 for subframes 0 and 1 or if L2 is less than 58
/// for subframes 2 and 3, then the 85 entry codebook is used").
const GAIN_ROWS_85: usize = 85;

/// Lag threshold of the §2.14 / §2.15 short-pitch rule.
pub(crate) const SHORT_LAG_LIMIT: i32 = 58;

/// Decoded contents of one 12-bit combined gain word.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GainInfo {
    /// Pitch predictor taps `β_i0..β_i4` (eq. 41.2), dimensionless.
    pub taps: [f32; ACB_TAPS],
    /// Fixed-codebook gain `G` in the normalised signal domain.
    pub fcb_gain: f32,
    /// `PGIndex` — row into the 170- or 85-entry gain-vector codebook.
    pub pgindex: usize,
    /// `MGIndex` — index into the 24-step fixed-codebook gain table.
    pub mgindex: usize,
    /// High-rate short-lag impulse-train bit (§2.15 / §2.17 step 5).
    pub train: bool,
}

/// Whether a subframe's gain word uses the 85-entry codebook + train
/// bit layout of eq. 40. `lag_base` is `L_0` for subframes 0–1 and
/// `L_2` for subframes 2–3 (§2.14).
pub(crate) fn uses_short_lag_gain(rate: PackedRate, lag_base: i32) -> bool {
    rate == PackedRate::High && lag_base < SHORT_LAG_LIMIT
}

/// Number of valid `PGIndex` rows for a subframe (§2.14).
pub(crate) fn gain_vq_rows(rate: PackedRate, lag_base: i32) -> usize {
    if uses_short_lag_gain(rate, lag_base) {
        GAIN_ROWS_85
    } else {
        GAIN_ROWS_170
    }
}

/// The five Q13 predictor taps of a gain-vector codebook row.
pub(crate) fn acb_taps(rate: PackedRate, lag_base: i32, pgindex: usize) -> [f32; ACB_TAPS] {
    let (table, rows): (&[i16], usize) = if uses_short_lag_gain(rate, lag_base) {
        (&ADAPTIVE_CODEBOOK_GAIN_5P3, GAIN_ROWS_85)
    } else {
        (&ADAPTIVE_CODEBOOK_GAIN_6P3, GAIN_ROWS_170)
    };
    let row = pgindex.min(rows - 1) * ADAPTIVE_CODEBOOK_ROW_DIM;
    let mut taps = [0.0f32; ACB_TAPS];
    for (t, &q) in taps.iter_mut().zip(table[row..row + ACB_TAPS].iter()) {
        *t = q as f32 / 8192.0;
    }
    taps
}

/// Fixed-codebook gain level `G̃_j` in the normalised signal domain.
pub(crate) fn fcb_gain_value(mgindex: usize) -> f32 {
    let idx = mgindex.min(FIXED_CODEBOOK_GAIN_Q15.len() - 1);
    FIXED_CODEBOOK_GAIN_Q15[idx] as f32 / 32_768.0
}

/// Nearest fixed-codebook gain index to `g` (normalised domain),
/// minimising `|G − G̃_j|` (§2.16 last step / §2.15).
pub(crate) fn nearest_fcb_gain(g: f32) -> usize {
    let target = g.abs() * 32_768.0;
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (j, &q) in FIXED_CODEBOOK_GAIN_Q15.iter().enumerate() {
        let d = (target - q as f32).abs();
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

/// Decode a 12-bit combined gain word (`GIND_i`) per eq. 36 / 39 / 40.
pub(crate) fn decode_gain_word(rate: PackedRate, lag_base: i32, gind: u32) -> GainInfo {
    let gsize = GAIN_TABLE_SIZE;
    let (pgindex, mgindex, train) = if uses_short_lag_gain(rate, lag_base) {
        // eq. 40: the MSB carries the impulse-train bit; the masked
        // remainder splits into (PGIndex, MGIndex) over the 85-entry
        // codebook. Masked words ≥ 85·24 = 2040 cannot be produced by a
        // conforming encoder; clamp the row for robustness.
        let masked = gind & 0x7FF;
        let pg = (masked / gsize) as usize;
        let mg = (masked % gsize) as usize;
        (pg.min(GAIN_ROWS_85 - 1), mg, gind & 0x800 != 0)
    } else {
        // eq. 39 + eq. 36. Words ≥ 170·24 = 4080 are non-conforming;
        // clamp the row.
        let pg = (gind / gsize) as usize;
        let mg = (gind % gsize) as usize;
        (pg.min(GAIN_ROWS_170 - 1), mg, false)
    };
    GainInfo {
        taps: acb_taps(rate, lag_base, pgindex),
        fcb_gain: fcb_gain_value(mgindex),
        pgindex,
        mgindex,
        train,
    }
}

/// Build a 12-bit combined gain word from its components (inverse of
/// [`decode_gain_word`]).
pub(crate) fn encode_gain_word(
    rate: PackedRate,
    lag_base: i32,
    pgindex: usize,
    mgindex: usize,
    train: bool,
) -> u32 {
    debug_assert!(pgindex < gain_vq_rows(rate, lag_base));
    debug_assert!((mgindex as u32) < GAIN_TABLE_SIZE);
    let base = pgindex as u32 * GAIN_TABLE_SIZE + mgindex as u32;
    if uses_short_lag_gain(rate, lag_base) && train {
        base | 0x800
    } else {
        base
    }
}

/// The five per-tap basis vectors of the eq. 41 pitch predictor:
/// `basis[j][n] = e′[n + j]` with `e′` per eq. 41.1 (`e′[0] = e[−L−2]`,
/// `e′[1] = e[−L−1]`, `e′[n] = e[(n mod L) − L]` for `2 ≤ n ≤ 63`).
/// `history` is the excitation buffer whose final element is the most
/// recent past sample `e[-1]`. The contribution `u[n]` is the tap-
/// weighted sum of these vectors; the encoder's closed-loop gain-VQ
/// search correlates each basis vector separately.
pub(crate) fn acb_basis(history: &[f32], lag: i32) -> [[f32; SUBFRAME_SIZE]; ACB_TAPS] {
    let l = lag.clamp(PITCH_MIN as i32, PITCH_MAX as i32) as usize;
    let hlen = history.len();
    debug_assert!(hlen >= l + 2);
    let eprime = |n: usize| -> f32 {
        let off = match n {
            0 => l + 2,
            1 => l + 1,
            _ => l - (n % l),
        };
        history[hlen - off]
    };
    let mut basis = [[0.0f32; SUBFRAME_SIZE]; ACB_TAPS];
    for (j, b) in basis.iter_mut().enumerate() {
        for (n, out) in b.iter_mut().enumerate() {
            *out = eprime(n + j);
        }
    }
    basis
}

/// Fifth-order adaptive-codebook (pitch predictor) contribution `u[n]`
/// per eq. 41.1–41.2. `history` is the excitation buffer whose final
/// element is the most recent past sample `e[-1]`.
pub(crate) fn acb_contribution(
    history: &[f32],
    lag: i32,
    taps: &[f32; ACB_TAPS],
) -> [f32; SUBFRAME_SIZE] {
    let basis = acb_basis(history, lag);
    let mut u = [0.0f32; SUBFRAME_SIZE];
    for (j, &b) in taps.iter().enumerate() {
        for (n, out) in u.iter_mut().enumerate() {
            *out += b * basis[j][n];
        }
    }
    u
}

/// Impulse response of the §2.16 pitch-synchronous enhancement filter
/// applied to `h`: the recursive 1-tap LTP `h′[n] = h[n] + β·h′[n − D]`
/// with `D = L + ε(PGIndex)` — the response a unit algebraic pulse sees
/// after [`acelp_pitch_enhance`] and the synthesis filter (§2.16: "the
/// impulse response should be modified" prior to the codebook search).
/// Returns `h` unchanged when the enhancement is inactive (long lag,
/// zero-β row, non-positive delay).
pub(crate) fn acelp_enhanced_impulse_response(h: &[f32], lag: i32, pgindex: usize) -> Vec<f32> {
    let mut out = h.to_vec();
    if lag >= SUBFRAME_SIZE as i32 {
        return out;
    }
    let Some(ltp) = pitch_1tap_ltp(pgindex) else {
        return out;
    };
    if ltp.gain == 0 {
        return out;
    }
    let beta = ltp.gain as f32 / 32_768.0;
    let delay = lag + ltp.selector as i32;
    if delay <= 0 {
        return out;
    }
    for n in delay as usize..out.len() {
        let prev = out[n - delay as usize];
        out[n] += beta * prev;
    }
    out
}

/// Reconstruct a high-rate (MP-MLQ) fixed-codebook vector `v[n]`
/// (§2.15 eq. 22 / §2.17 steps 2–6): combinatorial position decode,
/// grid placement, per-pulse signs, gain, and — when the train bit is
/// set on a short-lag subframe pair — a train of Dirac functions at
/// the reference pitch period instead of a single pulse.
///
/// `psig` bit `k` (LSB) is the sign of the `k`-th pulse in ascending
/// position order, `0` = positive (crate convention).
pub(crate) fn mpmlq_fixed_vector(
    pos_code: u32,
    psig: u32,
    grid: u8,
    n_pulses: usize,
    gain: f32,
    train: bool,
    lag_base: i32,
) -> [f32; SUBFRAME_SIZE] {
    let mut v = [0.0f32; SUBFRAME_SIZE];
    let Some(slots) = fcbk_unpk_positions(pos_code, n_pulses) else {
        return v;
    };
    let period = lag_base.max(1) as usize;
    for (k, &slot) in slots.iter().enumerate() {
        let sign = if (psig >> k) & 1 == 1 { -1.0f32 } else { 1.0 };
        let amp = sign * gain;
        let base = 2 * slot + grid as usize;
        if train {
            // §2.15: "a train of Dirac functions with the period of the
            // pitch index L0 or L2 is used for each location m_k".
            let mut pos = base;
            while pos < SUBFRAME_SIZE {
                v[pos] += amp;
                pos += period;
            }
        } else if base < SUBFRAME_SIZE {
            v[base] += amp;
        }
    }
    v
}

/// Reconstruct a low-rate (ACELP) fixed-codebook vector `v[n]` (§2.16 /
/// §2.17 step 2 "direct decoding of the position indices"): four 3-bit
/// track slots (track 0 in the low bits of `pos`), the shared grid bit,
/// per-track signs (`psig` bit `t`, `0` = positive), and the gain.
/// Track slots that map past the subframe boundary (Table 1's "(60)" /
/// "(62)") mean the pulse is absent.
pub(crate) fn acelp_fixed_vector(pos: u32, psig: u32, grid: u8, gain: f32) -> [f32; SUBFRAME_SIZE] {
    let mut v = [0.0f32; SUBFRAME_SIZE];
    for track in 0..4usize {
        let slot = (pos >> (3 * track)) & 0x7;
        let Some(sample) = crate::spec_tables::acelp_track_position(
            crate::spec_tables::AcelpTrack::ALL[track],
            slot as usize,
            grid != 0,
        ) else {
            continue;
        };
        let sign = if (psig >> track) & 1 == 1 {
            -1.0f32
        } else {
            1.0
        };
        v[sample] += sign * gain;
    }
    v
}

/// §2.16 pitch-synchronous enhancement of the ACELP codeword for short
/// pitch delays: `v[n] ← v[n] + β(PGIndex)·v[n − L − ε(PGIndex)]`,
/// applied in ascending `n` (so the enhancement recurses, extending a
/// pulse into a decaying pitch-periodic train). β = 0 rows (published
/// selector sentinel 60) leave the vector untouched.
pub(crate) fn acelp_pitch_enhance(v: &mut [f32; SUBFRAME_SIZE], lag: i32, pgindex: usize) {
    if lag >= SUBFRAME_SIZE as i32 {
        return;
    }
    let Some(ltp) = pitch_1tap_ltp(pgindex) else {
        return;
    };
    if ltp.gain == 0 {
        return;
    }
    let beta = ltp.gain as f32 / 32_768.0;
    let delay = lag + ltp.selector as i32;
    if delay <= 0 {
        return;
    }
    for n in delay as usize..SUBFRAME_SIZE {
        v[n] += beta * v[n - delay as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec_tables::PITCH_1TAP_LTP_GAIN;

    #[test]
    fn gain_word_round_trips_all_layouts() {
        // 170-entry layout: every conforming word splits and rebuilds.
        for pg in [0usize, 1, 84, 85, 169] {
            for mg in [0usize, 11, 23] {
                let w = encode_gain_word(PackedRate::Low, 100, pg, mg, false);
                assert!(w < 4096);
                let d = decode_gain_word(PackedRate::Low, 100, w);
                assert_eq!((d.pgindex, d.mgindex, d.train), (pg, mg, false));
                // High rate with a long lag uses the same layout.
                let d = decode_gain_word(PackedRate::High, 58, w);
                assert_eq!((d.pgindex, d.mgindex, d.train), (pg, mg, false));
            }
        }
        // 85-entry short-lag layout with the train bit in the MSB.
        for pg in [0usize, 42, 84] {
            for mg in [0usize, 23] {
                for train in [false, true] {
                    let w = encode_gain_word(PackedRate::High, 20, pg, mg, train);
                    assert!(w < 4096);
                    let d = decode_gain_word(PackedRate::High, 20, w);
                    assert_eq!((d.pgindex, d.mgindex, d.train), (pg, mg, train));
                }
            }
        }
    }

    #[test]
    fn low_rate_never_uses_short_lag_layout() {
        assert!(!uses_short_lag_gain(PackedRate::Low, 18));
        assert!(uses_short_lag_gain(PackedRate::High, 57));
        assert!(!uses_short_lag_gain(PackedRate::High, 58));
        assert_eq!(gain_vq_rows(PackedRate::Low, 18), 170);
        assert_eq!(gain_vq_rows(PackedRate::High, 18), 85);
    }

    #[test]
    fn taps_read_the_first_five_row_entries_in_q13() {
        // Row 5 of the 85-entry codebook (verified numerically in the
        // module docs): taps [-125, -40, -264, 381, 5027] / 8192.
        let taps = acb_taps(PackedRate::High, 20, 5);
        let expected = [-125.0, -40.0, -264.0, 381.0, 5027.0].map(|q| q / 8192.0);
        assert_eq!(taps, expected);
    }

    #[test]
    fn fcb_gain_table_maps_to_normalised_domain_and_quantises_back() {
        for (j, &q) in FIXED_CODEBOOK_GAIN_Q15.iter().enumerate() {
            let g = fcb_gain_value(j);
            assert!((g - q as f32 / 32_768.0).abs() < 1e-9);
            assert_eq!(nearest_fcb_gain(g), j, "level {j} must be its own nearest");
        }
        // The quantiser uses |G| — a negative gain maps like its magnitude.
        assert_eq!(nearest_fcb_gain(-fcb_gain_value(12)), 12);
    }

    #[test]
    fn acb_contribution_matches_eq41_geometry() {
        // History with a single 1.0 at e[-1] and taps selecting only
        // e'[n+2] (the j = 2 tap): u[n] = e[((n+2) mod L) − L], which is
        // non-zero exactly when ((n+2) mod L) == L − 1.
        let lag = 40usize;
        let mut hist = vec![0.0f32; 202];
        let hlen = hist.len();
        hist[hlen - 1] = 1.0;
        let taps = [0.0, 0.0, 1.0, 0.0, 0.0];
        let u = acb_contribution(&hist, lag as i32, &taps);
        for (n, &s) in u.iter().enumerate() {
            let expect = if (n + 2) % lag == lag - 1 { 1.0 } else { 0.0 };
            assert_eq!(s, expect, "sample {n}");
        }

        // The j = 0 tap reads e[−L−2] at n = 0 (eq. 41.1 first seed).
        let mut hist = vec![0.0f32; 202];
        hist[hlen - (lag + 2)] = 0.5;
        let taps = [1.0, 0.0, 0.0, 0.0, 0.0];
        let u = acb_contribution(&hist, lag as i32, &taps);
        assert_eq!(u[0], 0.5);
    }

    #[test]
    fn mpmlq_fixed_vector_places_signed_pulses_and_trains() {
        // Pack the 6-pulse slot set {0, 3, 7, 12, 20, 29} and decode on
        // the odd grid with alternating signs.
        let slots = [0usize, 3, 7, 12, 20, 29];
        let code = crate::spec_tables::fcbk_pack_positions(&slots).unwrap();
        let psig = 0b101010; // pulses 1, 3, 5 negative
        let v = mpmlq_fixed_vector(code, psig, 1, 6, 0.25, false, 100);
        for (k, &slot) in slots.iter().enumerate() {
            let sample = 2 * slot + 1;
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            assert_eq!(v[sample], sign * 0.25, "pulse {k}");
        }
        assert_eq!(v.iter().filter(|&&s| s != 0.0).count(), 6);

        // Train mode replicates each pulse at the reference period.
        let slots = [2usize];
        // A single-pulse set is not a legal MP-MLQ codeword, so build
        // the vector directly through a 5-pulse set with one low pulse
        // and check periodicity of the first pulse's train.
        let slots5 = [2usize, 25, 26, 27, 28];
        let code = crate::spec_tables::fcbk_pack_positions(&slots5).unwrap();
        let v = mpmlq_fixed_vector(code, 0, 0, 5, 1.0, true, 20);
        // Pulse at sample 4 with period 20: train at 4, 24, 44.
        assert_eq!(v[4], 1.0);
        assert_eq!(v[24], 1.0);
        assert_eq!(v[44], 1.0);
        let _ = slots;
    }

    #[test]
    fn acelp_fixed_vector_follows_table1_tracks() {
        // Slots (1, 2, 7, 7): track 0 → 8, track 1 → 18, tracks 2/3 at
        // slot 7 on the even grid are the absent "(60)" / "(62)".
        let pos = 1 | (2 << 3) | (7 << 6) | (7 << 9);
        let v = acelp_fixed_vector(pos, 0b0010, 0, 0.5);
        assert_eq!(v[8], 0.5);
        assert_eq!(v[18], -0.5); // track 1 sign bit set
        assert_eq!(v.iter().filter(|&&s| s != 0.0).count(), 2);

        // Odd grid shifts every present pulse by +1 and makes slot 7 of
        // tracks 2/3 real samples (61/63 → wait: 60/62 + 1 = 61/63 are
        // beyond 59 — still absent).
        let v = acelp_fixed_vector(pos, 0, 1, 0.5);
        assert_eq!(v[9], 0.5);
        assert_eq!(v[19], 0.5);
        assert_eq!(v.iter().filter(|&&s| s != 0.0).count(), 2);
    }

    #[test]
    fn acelp_pitch_enhance_builds_recursive_train() {
        // Find a PGIndex with a non-zero published β and ε = 0.
        let pg = (0..170)
            .find(|&i| {
                PITCH_1TAP_LTP_GAIN[i] != 0 && crate::spec_tables::PITCH_1TAP_LTP_SELECTOR[i] == 0
            })
            .unwrap();
        let beta = PITCH_1TAP_LTP_GAIN[pg] as f32 / 32_768.0;
        let mut v = [0.0f32; SUBFRAME_SIZE];
        v[0] = 1.0;
        acelp_pitch_enhance(&mut v, 20, pg);
        assert!((v[20] - beta).abs() < 1e-6);
        assert!((v[40] - beta * beta).abs() < 1e-6, "enhancement recurses");

        // A zero-gain row (selector sentinel) must be a no-op, and lags
        // of a full subframe or more must bypass entirely.
        let zero_pg = (0..170).find(|&i| PITCH_1TAP_LTP_GAIN[i] == 0).unwrap();
        let mut v2 = [0.0f32; SUBFRAME_SIZE];
        v2[0] = 1.0;
        acelp_pitch_enhance(&mut v2, 20, zero_pg);
        assert_eq!(v2[20], 0.0);
        let mut v3 = [0.0f32; SUBFRAME_SIZE];
        v3[0] = 1.0;
        acelp_pitch_enhance(&mut v3, 60, pg);
        assert_eq!(v3.iter().filter(|&&s| s != 0.0).count(), 1);
    }
}
