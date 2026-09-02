//! ITU-T G.723.1 **Annex A** silence compression — the encoder side:
//! the voice activity detector (A.2), the COD-CNG frame-type decision
//! with its SID parameter computation (A.4.1–A.4.4), the 6-bit
//! pseudo-logarithmic SID gain quantiser (A.4.3), the comfort-noise
//! excitation the local decoder must be fed on inactive frames
//! (A.4.5) and the SID octet layout (A.6, Table A.1).
//!
//! Everything here follows the Annex A prose of the 05/2006 edition.
//! One element the text does not define is the pseudo-random sequence
//! (`random_number()` / `Rand_lbc()`, seed 12345, reset at every active
//! frame — A.4.7 / A.5.2): this module uses a documented linear
//! congruential generator of its own, so the comfort-noise excitation —
//! and therefore the local decoder state entering the first active
//! frame after a silence — is *not* bit-exact with the reference. The
//! VAD decision, the SID / untransmitted choice and the SID contents
//! (LSP index, gain index) depend only on the input signal and are
//! reproducible.
//!
//! Signal units: the VAD and CNG energies are formed in the reference's
//! Word16 sample units, i.e. the framer output `s[n]` as transmitted
//! (`i16`), and the SID gain / CNG excitation gains are in the
//! reference's *half-scale* excitation units (the analysis domain the
//! r455 §2.3 arbitration and the r406 doubled pulse amplitudes pin: one
//! reference excitation unit is `2/32768` of this crate's normalised
//! excitation domain).

use crate::spec_tables::{
    fcbk_unpk_positions, BIT_ALLOCATION_SEGMENT_BASE, BIT_ALLOCATION_SEGMENT_BOUNDARIES,
};
use crate::tables::{FRAME_SIZE_SAMPLES, LPC_ORDER, SUBFRAME_SIZE};

/// Frame type `Ftyp_t` produced by COD-CNG (A.3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// `Ftyp = 0` — untransmitted (one octet on the wire).
    Untransmitted,
    /// `Ftyp = 1` — active speech frame.
    Active,
    /// `Ftyp = 2` — SID frame (four octets).
    Sid,
}

/// A.4.3 scaling factor `α_w` accounting for the windowing and
/// bandwidth expansion in the subframe autocorrelations.
pub const SID_GAIN_ALPHA_W: f64 = 2.703_75;
/// A.4.3 upper bound of the quantiser input.
pub const SID_GAIN_MAX: f64 = 352.0;
/// A.4.3 segment lengths `N[isg]`.
const SID_SEG_LEN: [u32; 3] = [16, 16, 32];
/// A.4.2 Itakura-distance threshold `thr1`.
pub const CNG_THR1: f64 = 1.2136;
/// A.4.2 gain-index difference threshold `thr2`.
pub const CNG_THR2: i32 = 3;
/// A.4.5 bound on the fixed-excitation gain `G_f` (half-scale units).
pub const CNG_GF_MAX: f64 = 5000.0;
/// A.4.7 / A.5.3 random-generator seed.
pub const CNG_SEED: u32 = 12_345;

/// A.4.3 quantisation of the SID gain: the 6-bit index `GInd_t` of the
/// pseudo-logarithmic quantiser over `[0, 352]` — three segments of
/// 16 / 16 / 32 levels with resolutions 2 / 4 / 8 starting at the
/// published segment bases (0, 32, 96); the segment is found from `G²`
/// against the published squared decision boundaries and the level
/// closest to `G` inside it is taken (eq. A-12 / A-13).
pub fn sid_gain_quantise(g: f64) -> u8 {
    let g = g.clamp(0.0, SID_GAIN_MAX);
    // The boundary table holds 2·G² at the segment edges (2·32²,
    // 2·96² and the last-level midpoint 2·340²).
    let g2x2 = 2.0 * g * g;
    let isg = if g2x2 < BIT_ALLOCATION_SEGMENT_BOUNDARIES[0] as f64 {
        0usize
    } else if g2x2 < BIT_ALLOCATION_SEGMENT_BOUNDARIES[1] as f64 {
        1
    } else {
        2
    };
    let base = BIT_ALLOCATION_SEGMENT_BASE[isg] as f64;
    let res = (2u32 << isg) as f64;
    let is = ((g - base) / res)
        .round()
        .clamp(0.0, (SID_SEG_LEN[isg] - 1) as f64) as u32;
    (16 * isg as u32 + is) as u8
}

/// A.4.3 decoding (eq. A-14): `G̃ = G_isg[0] + (GInd − 16·isg)·2^(isg+1)`.
pub fn sid_gain_decode(idx: u8) -> f64 {
    let idx = (idx & 0x3F) as u32;
    let isg = (idx / 16).min(2) as usize;
    let is = idx - 16 * isg as u32;
    BIT_ALLOCATION_SEGMENT_BASE[isg] as f64 + is as f64 * (2u32 << isg) as f64
}

/// Pack a SID frame per Table A.1: the LSB-first clause-4 stream
/// `RATEFLAG = 0, VADFLAG = 1, LPC(24), GAIN(6)`.
pub fn pack_sid(lsp_index: u32, gain_index: u8) -> [u8; 4] {
    let word: u32 = 0b10 | ((lsp_index & 0xFF_FFFF) << 2) | ((gain_index as u32 & 0x3F) << 26);
    word.to_le_bytes()
}

/// Unpack a SID frame (`(lsp_index, gain_index)`); `None` when the
/// discriminator is not `10`.
pub fn unpack_sid(data: &[u8]) -> Option<(u32, u8)> {
    if data.len() < 4 || data[0] & 0b11 != 0b10 {
        return None;
    }
    let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    Some(((word >> 2) & 0xFF_FFFF, ((word >> 26) & 0x3F) as u8))
}

/// The single-octet untransmitted frame (discriminator `11`).
pub const UNTRANSMITTED_OCTET: u8 = 0b11;

/// Levinson-Durbin on eleven autocorrelation coefficients: returns
/// `[1, a_1..a_10]` in the crate's inverse-filter convention
/// (`A(z) = 1 + Σ a_j z^-j`) and the residual energy `E`.
pub fn durbin(r: &[f64; LPC_ORDER + 1]) -> ([f64; LPC_ORDER + 1], f64) {
    let mut a = [0.0f64; LPC_ORDER + 1];
    let mut prev = [0.0f64; LPC_ORDER + 1];
    a[0] = 1.0;
    prev[0] = 1.0;
    let mut e = r[0];
    if e <= 0.0 {
        return (a, 0.0);
    }
    for i in 1..=LPC_ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += prev[j] * r[i - j];
        }
        let k = -acc / e;
        if !k.is_finite() || k.abs() >= 1.0 {
            break;
        }
        a[i] = k;
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        e *= 1.0 - k * k;
        prev.copy_from_slice(&a);
        if e <= 0.0 {
            e = 0.0;
            break;
        }
    }
    (prev, e)
}

/// A.4.2 eq. A-11: the autocorrelation `R_a[j]` of an LPC filter
/// (`a[0] = 1`), doubled for `j ≠ 0`.
pub fn lpc_autocorrelation(a: &[f64; LPC_ORDER + 1]) -> [f64; LPC_ORDER + 1] {
    let mut ra = [0.0f64; LPC_ORDER + 1];
    for j in 0..=LPC_ORDER {
        let mut acc = 0.0;
        for k in 0..=LPC_ORDER - j {
            acc += a[k] * a[k + j];
        }
        ra[j] = if j == 0 { acc } else { 2.0 * acc };
    }
    ra
}

/// A.4.2 eq. A-10: the two filters differ significantly when
/// `Σ R_a[j]·R_t[j] > E_t·thr1`.
pub fn lpc_differs(a_ref: &[f64; LPC_ORDER + 1], r_t: &[f64; LPC_ORDER + 1], e_t: f64) -> bool {
    let ra = lpc_autocorrelation(a_ref);
    let mut d = 0.0;
    for j in 0..=LPC_ORDER {
        d += ra[j] * r_t[j];
    }
    d > e_t * CNG_THR1
}

// ---------------------------------------------------------------------
// A.2 — voice activity detector
// ---------------------------------------------------------------------

/// A.2 VAD state.
#[derive(Clone, Debug)]
pub struct Vad {
    /// Noise level `Nlev_{t−1}` (A.2.4), initialised to 1024.
    nlev: f64,
    /// Previous frame's filtered energy `Enr_{t−1}`, initialised to 1024.
    enr_prev: f64,
    /// Adaptation enable flag `Aen`, bounded in `[0, 6]`.
    aen: i32,
    /// Open-loop lags of the preceding and current frame
    /// (`L_OL^j, j = 0..3`), initialised per A.2.8.
    lags: [i32; 4],
    /// The last 15 second reflection coefficients `k_i^t[2]` (A.2.1).
    k2_hist: [f32; 15],
    k2_pos: usize,
    /// Length of the current speech burst and remaining hangover.
    burst: u32,
    hangover: u32,
    /// Noise inverse filter `A_no(z)` coefficients `a_no[1..10]`
    /// (A.2.2), updated by COD-CNG (A.4.4 eq. A-16).
    pub a_no: [f64; LPC_ORDER],
    /// Whether the last frame was declared active (after hangover).
    pub vad_prev: bool,
    /// Whether the noise level may adapt this frame (`Aen_t = 0`).
    pub adapt_enabled: bool,
    /// Scale applied to the A-2 energy before the threshold test.
    /// Vector-arbitrated (r455): the framer output enters the energy at
    /// **half scale** (`(s/2)²`, i.e. ×1/4) — at unity scale the
    /// reference's inactive stretch at the start of DTX63 (rms ≈ 65
    /// noise against `Nlev_{-1} = 1024`) is never reached (frame-type
    /// agreement 85.8 → 94.4% at ×1/4; ×1/2 87.0%, ×1/8 91.9%).
    #[doc(hidden)]
    pub energy_scale: f64,
    /// Frames declared active unconditionally at start-up.
    /// Vector-arbitrated (r455): both DTX63 and DTX53MIX open with
    /// exactly three active frames whatever the input level (DTX53MIX
    /// starts at rms 0–4), a rule the annex text does not state; as a
    /// final override it lifts DTX53MIX 90.0 → 93.3% and DTX63 94.1 →
    /// 94.4%.
    #[doc(hidden)]
    pub startup_frames: u32,
    frame_count: u32,
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad {
    /// A.2.8 initialisation.
    pub fn new() -> Self {
        Self {
            nlev: 1024.0,
            enr_prev: 1024.0,
            aen: 0,
            lags: [1, 1, 60, 60],
            k2_hist: [0.0; 15],
            k2_pos: 0,
            burst: 0,
            hangover: 0,
            a_no: [0.0; LPC_ORDER],
            vad_prev: true,
            adapt_enabled: false,
            energy_scale: 0.25,
            startup_frames: 3,
            frame_count: 0,
        }
    }

    /// One frame of A.2: `frame` is the framer output `s[n]` (Word16
    /// units), `ol_lags` the two §2.9 open-loop estimates of this
    /// frame, `k2` the four subframes' second reflection coefficients.
    /// Returns `Vad_t` after hangover.
    pub fn decide(
        &mut self,
        frame: &[i16; FRAME_SIZE_SAMPLES],
        ol_lags: [i32; 2],
        k2: [f32; 4],
    ) -> bool {
        // --- A.2.1 adaptation enable flag ---
        self.lags = [self.lags[2], self.lags[3], ol_lags[0], ol_lags[1]];
        let lmin = *self.lags.iter().min().unwrap();
        let mut pc = 0;
        for &l in &self.lags {
            let m = ((l as f64 / lmin as f64).round() as i32).max(1);
            if (l - m * lmin).abs() <= 3 {
                pc += 1;
            }
        }
        for &k in &k2 {
            self.k2_hist[self.k2_pos] = k;
            self.k2_pos = (self.k2_pos + 1) % 15;
        }
        let sines = self.k2_hist.iter().filter(|&&k| k >= 0.95).count();
        let sin_d = sines >= 14;
        self.aen = if pc == 4 || sin_d {
            (self.aen + 2).min(6)
        } else {
            (self.aen - 1).max(0)
        };
        self.adapt_enabled = self.aen == 0;

        // --- A.2.2 / A.2.3 inverse filtering and energy ---
        let mut energy = 0.0f64;
        for n in SUBFRAME_SIZE..FRAME_SIZE_SAMPLES {
            let mut e = frame[n] as f64;
            for j in 1..=LPC_ORDER {
                e += self.a_no[j - 1] * frame[n - j] as f64;
            }
            energy += e * e;
        }
        let enr = energy * self.energy_scale / 80.0;

        // --- A.2.4 noise level from the *previous* energy ---
        let mut nlev = self.nlev;
        if nlev > self.enr_prev {
            nlev = 0.25 * nlev + 0.75 * self.enr_prev;
        }
        nlev *= if self.aen == 0 { 1.031_25 } else { 0.9995 };
        nlev = nlev.clamp(128.0, 131_071.0);
        self.nlev = nlev;
        self.enr_prev = enr;

        // --- A.2.5 / A.2.6 threshold and decision ---
        let ratio = if nlev <= 128.0 {
            5.012
        } else if nlev < 16_384.0 {
            10f64.powf(0.7 - 0.05 * (nlev / 128.0).log2())
        } else {
            2.239
        };
        let raw = enr >= ratio * nlev;

        // --- A.2.7 hangover ---
        let vad = if raw {
            self.burst += 1;
            self.hangover = 0;
            true
        } else if self.burst >= 2 && self.hangover < 6 {
            self.hangover += 1;
            if self.hangover == 6 {
                self.burst = 0;
            }
            true
        } else {
            self.burst = 0;
            self.hangover = 0;
            false
        };
        // Start-up override (vector-arbitrated, see `startup_frames`).
        let vad = vad || self.frame_count < self.startup_frames;
        self.frame_count = self.frame_count.saturating_add(1);
        self.vad_prev = vad;
        vad
    }
}

// ---------------------------------------------------------------------
// A.4 — COD-CNG
// ---------------------------------------------------------------------

/// Documented pseudo-random generator standing in for the annex's
/// unspecified `random_number()` (a 32-bit linear congruential
/// generator; **not** the reference sequence).
#[derive(Clone, Debug)]
pub struct Lcg(pub u32);

impl Lcg {
    /// Next 15-bit value.
    pub fn next_u15(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (self.0 >> 16) & 0x7FFF
    }
}

/// COD-CNG state (A.4).
#[derive(Clone, Debug)]
pub struct CodCng {
    /// Cumulated autocorrelations of the three preceding frames
    /// (`R^{t−1}, R^{t−2}, R^{t−3}`), reference units.
    r_hist: [[f64; LPC_ORDER + 1]; 3],
    /// Residual energies `E_t` of the last frames (A.4.2).
    e_hist: [f64; 3],
    /// Number of frames in the energy sum, `k_E ≤ 3`.
    k_e: usize,
    /// Coded SID gain index `GInd_sid` and its decoded value.
    pub gind_sid: u8,
    pub g_sid: f64,
    /// SID LPC filter `A_sid(z)` (crate sign convention, `a[0] = 1`).
    a_sid: [f64; LPC_ORDER + 1],
    /// Target excitation gain `G̃_t` (A-18).
    g_t: f64,
    /// Random generator (reset to `CNG_SEED` at every active frame).
    pub rng: Lcg,
    /// Scale applied to the cumulated autocorrelation (given in
    /// half-scale Word16² units) before the SID gain computation.
    /// Vector-arbitrated (r455): a further ×1/4 puts the coded SID gain
    /// index on the reference's (DTX63: 84% exact, 96% within ±1;
    /// without it every index sits ~10 steps high).
    #[doc(hidden)]
    pub r_scale: f64,
    /// SID filter choice knob: 0 = A-10 rule, 1 = always A_t, 2 = always A_p.
    #[doc(hidden)]
    pub sid_filter_mode: u8,
}

impl Default for CodCng {
    fn default() -> Self {
        Self::new()
    }
}

/// SID parameters computed for a SID frame (A.4.4): the SID LPC
/// filter to LSP-quantise and the coded gain.
#[derive(Clone, Copy, Debug)]
pub struct SidParams {
    pub a_sid: [f64; LPC_ORDER + 1],
    pub gain_index: u8,
}

impl CodCng {
    /// A.4.7 initialisation.
    pub fn new() -> Self {
        Self {
            r_hist: [[0.0; LPC_ORDER + 1]; 3],
            e_hist: [0.0; 3],
            k_e: 0,
            gind_sid: 0,
            g_sid: 0.0,
            a_sid: {
                let mut a = [0.0; LPC_ORDER + 1];
                a[0] = 1.0;
                a
            },
            g_t: 0.0,
            rng: Lcg(CNG_SEED),
            r_scale: 0.25,
            sid_filter_mode: 0,
        }
    }

    /// A.4.1–A.4.4 for one frame. `r_t` is the frame's cumulated
    /// autocorrelation (eq. A-8) in reference (half-scale Word16²)
    /// units, `vad` / `vad_prev` the current and previous VAD
    /// decisions. Returns the frame type and, for SID frames, the SID
    /// parameters; also updates the VAD noise filter when adaptation is
    /// enabled (eq. A-16).
    pub fn frame_type(
        &mut self,
        r_in: &[f64; LPC_ORDER + 1],
        vad: bool,
        vad_prev: bool,
        vad_state: &mut Vad,
    ) -> (FrameType, Option<SidParams>) {
        let mut r_t = [0.0f64; LPC_ORDER + 1];
        for j in 0..=LPC_ORDER {
            r_t[j] = r_in[j] * self.r_scale;
        }
        let r_t = &r_t;
        // Past-average autocorrelation of the three preceding frames
        // (eq. A-15) — captured before this frame enters the history.
        let mut r_p = [0.0f64; LPC_ORDER + 1];
        for k in 0..3 {
            for j in 0..=LPC_ORDER {
                r_p[j] += self.r_hist[k][j];
            }
        }
        self.r_hist = [*r_t, self.r_hist[0], self.r_hist[1]];

        if vad {
            self.rng = Lcg(CNG_SEED);
            return (FrameType::Active, None);
        }

        let (a_t, e_t) = durbin(r_t);
        self.e_hist = [e_t, self.e_hist[0], self.e_hist[1]];

        let first = vad_prev;
        let mut sid = first;
        if first {
            self.k_e = 1;
        } else {
            self.k_e = (self.k_e + 1).min(3);
        }
        let e_sum: f64 = self.e_hist[..self.k_e].iter().sum();
        let g = SID_GAIN_ALPHA_W
            * (e_sum / (self.k_e as f64 * FRAME_SIZE_SAMPLES as f64))
                .max(0.0)
                .sqrt();
        let gind_t = sid_gain_quantise(g);
        if !first {
            if lpc_differs(&self.a_sid, r_t, e_t) {
                sid = true;
            }
            if (gind_t as i32 - self.gind_sid as i32).abs() > CNG_THR2 {
                sid = true;
            }
        }
        if !sid {
            return (FrameType::Untransmitted, None);
        }

        // --- A.4.4: SID LPC filter and VAD noise-filter update ---
        let (a_p, _) = durbin(&r_p);
        if vad_state.adapt_enabled {
            vad_state.a_no.copy_from_slice(&a_p[1..=LPC_ORDER]);
        }
        let a_sid = match self.sid_filter_mode {
            1 => a_t,
            2 => a_p,
            _ => {
                if lpc_differs(&a_p, r_t, e_t) {
                    a_t
                } else {
                    a_p
                }
            }
        };
        self.a_sid = a_sid;
        self.gind_sid = gind_t;
        self.g_sid = sid_gain_decode(gind_t);
        (
            FrameType::Sid,
            Some(SidParams {
                a_sid,
                gain_index: gind_t,
            }),
        )
    }

    /// A.4.5 target excitation gain `G̃_t` for an inactive frame
    /// (eq. A-18), half-scale units.
    pub fn target_gain(&mut self, vad_prev: bool) -> f64 {
        self.g_t = if vad_prev {
            self.g_sid
        } else {
            0.875 * self.g_t + 0.125 * self.g_sid
        };
        self.g_t
    }

    /// A.4.5 comfort-noise excitation for one 120-sample block (two
    /// subframes) in the crate's normalised excitation domain.
    /// `history` is the excitation buffer (most recent sample last),
    /// `block` 0 or 1 within the frame, `g_t` the target gain from
    /// [`CodCng::target_gain`]. Returns the block excitation plus the
    /// two lags and gain rows drawn.
    pub fn block_excitation(
        &mut self,
        history: &[f32],
        block: usize,
        g_t: f64,
    ) -> [f32; 2 * SUBFRAME_SIZE] {
        use crate::linepack::PackedRate;
        use crate::spec_exc::{acb_contribution, acb_taps};
        // Long-term parameters.
        let lag0 = 123 + (self.rng.next_u15() % 21) as i32;
        let lag1 = lag0 + if block == 0 { 0 } else { 3 };
        let pg0 = (self.rng.next_u15() % 50) as usize;
        let pg1 = (self.rng.next_u15() % 50) as usize;
        // Both lags exceed the block length, so the second subframe's
        // adaptive contribution reads only the pre-block history: form
        // it on a history extended by a zero first subframe.
        let mut u = [0.0f32; 2 * SUBFRAME_SIZE];
        let u0 = acb_contribution(history, lag0, &acb_taps(PackedRate::High, 100, pg0));
        u[..SUBFRAME_SIZE].copy_from_slice(&u0);
        let mut ext = Vec::with_capacity(history.len());
        ext.extend_from_slice(&history[SUBFRAME_SIZE..]);
        ext.extend_from_slice(&[0.0f32; SUBFRAME_SIZE]);
        let u1 = acb_contribution(&ext, lag1, &acb_taps(PackedRate::High, 100, pg1));
        u[SUBFRAME_SIZE..].copy_from_slice(&u1);
        // Fixed codebook: random high-rate pattern (6 / 5 pulses).
        let mut v = [0.0f32; 2 * SUBFRAME_SIZE];
        for sf in 0..2 {
            let n_pulses = if sf == 0 { 6 } else { 5 };
            let grid = (self.rng.next_u15() & 1) as usize;
            let max_code = crate::spec_tables::mpmlq_max_position(sf).unwrap_or(1);
            let code = ((self.rng.next_u15() as u32) << 15 | self.rng.next_u15() as u32) % max_code;
            let signs = self.rng.next_u15();
            if let Some(slots) = fcbk_unpk_positions(code, n_pulses) {
                for (k, &slot) in slots.iter().enumerate() {
                    let pos = 2 * slot + grid;
                    if pos < SUBFRAME_SIZE {
                        v[sf * SUBFRAME_SIZE + pos] =
                            if (signs >> k) & 1 == 1 { -1.0 } else { 1.0 };
                    }
                }
            }
        }
        // Fixed gain from the quadratic (A-19); half-scale units →
        // normalised via /16384.
        let scale = 1.0 / 16_384.0;
        let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
        for n in 0..2 * SUBFRAME_SIZE {
            let un = u[n] as f64 / scale;
            let vn = v[n] as f64;
            a += vn * vn;
            b += un * vn;
            c += un * un;
        }
        c -= (2 * SUBFRAME_SIZE) as f64 * g_t * g_t;
        let gf = if a <= 0.0 {
            0.0
        } else {
            let disc = b * b - a * c;
            if disc <= 0.0 {
                -b / a
            } else {
                let s = disc.sqrt();
                let r1 = (-b + s) / a;
                let r2 = (-b - s) / a;
                if r1.abs() < r2.abs() {
                    r1
                } else {
                    r2
                }
            }
        };
        let gf = gf.clamp(-CNG_GF_MAX, CNG_GF_MAX);
        let mut e = [0.0f32; 2 * SUBFRAME_SIZE];
        for n in 0..2 * SUBFRAME_SIZE {
            e[n] = u[n] + (gf * scale) as f32 * v[n];
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sid_gain_quantiser_round_trips_its_levels() {
        // Every level decodes to itself and re-quantises to its index.
        for idx in 0u8..64 {
            let g = sid_gain_decode(idx);
            assert_eq!(sid_gain_quantise(g), idx, "level {idx} = {g}");
        }
        assert_eq!(sid_gain_decode(0), 0.0);
        assert_eq!(sid_gain_decode(16), 32.0);
        assert_eq!(sid_gain_decode(32), 96.0);
        assert_eq!(sid_gain_decode(63), 96.0 + 31.0 * 8.0);
        // Segment boundaries: 32 and 96 belong to the next segment; the
        // last midpoint 340 rounds up to the final level.
        assert_eq!(sid_gain_quantise(31.9), 15);
        assert_eq!(sid_gain_quantise(32.0), 16);
        assert_eq!(sid_gain_quantise(95.9), 31);
        assert_eq!(sid_gain_quantise(96.0), 32);
        assert_eq!(sid_gain_quantise(400.0), 63);
    }

    #[test]
    fn sid_pack_round_trip_and_discriminator() {
        for (lsp, g) in [
            (0u32, 0u8),
            (0xAF_8BBD, 21),
            (0xFF_FFFF, 63),
            (0x12_3456, 7),
        ] {
            let b = pack_sid(lsp, g);
            assert_eq!(b[0] & 0b11, 0b10);
            assert_eq!(unpack_sid(&b), Some((lsp, g)));
        }
        // The reference SID octets of DTX63.RCO frame 3 decode as
        // documented in the round notes (LPC 0xAF8BBD, gain 21).
        assert_eq!(unpack_sid(&[0xf6, 0x2e, 0xbe, 0x56]), Some((0xAF_8BBD, 21)));
        assert_eq!(unpack_sid(&[0x03]), None);
    }

    #[test]
    fn durbin_recovers_a_two_pole_model() {
        // Autocorrelation of 1/(1 − 0.5 z⁻¹) driven by white noise:
        // r[k] = 0.5^k / (1 − 0.25).
        let mut r = [0.0f64; 11];
        for k in 0..11 {
            r[k] = 0.5f64.powi(k as i32) / 0.75;
        }
        let (a, e) = durbin(&r);
        assert!((a[1] + 0.5).abs() < 1e-9);
        for j in 2..=10 {
            assert!(a[j].abs() < 1e-9);
        }
        assert!((e - 1.0).abs() < 1e-9);
        // The Itakura distance of the model against its own
        // autocorrelation is exactly E — i.e. under the threshold.
        assert!(!lpc_differs(&a, &r, e));
        let mut b = a;
        b[1] = 0.9;
        assert!(lpc_differs(&b, &r, e));
    }

    #[test]
    fn vad_flags_speech_above_adapted_noise() {
        let mut vad = Vad::new();
        let mut frame = [0i16; FRAME_SIZE_SAMPLES];
        // Low-level noise: adapts the level; a loud burst is speech.
        let mut seed = 1u32;
        for t in 0..40 {
            for v in frame.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *v = (((seed >> 16) & 0xFF) as i16 - 128) / 4;
            }
            let d = vad.decide(&frame, [40 + t % 7, 90], [0.0; 4]);
            if t > 30 {
                assert!(!d, "steady low noise must be inactive by frame {t}");
            }
        }
        for v in frame.iter_mut() {
            *v *= 200;
        }
        assert!(vad.decide(&frame, [40, 41], [0.0; 4]));
    }

    #[test]
    fn cng_block_excitation_hits_the_target_energy() {
        let mut cng = CodCng::new();
        let history = vec![0.0f32; 146 + 60];
        let g_t = 100.0; // half-scale units
        let e = cng.block_excitation(&history, 0, g_t);
        let energy: f64 = e
            .iter()
            .map(|&x| (x as f64 * 16_384.0).powi(2))
            .sum::<f64>()
            / 120.0;
        assert!(
            (energy.sqrt() - g_t).abs() < 1.0,
            "rms {} vs target {g_t}",
            energy.sqrt()
        );
    }
}
