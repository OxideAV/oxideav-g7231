//! ITU-T G.723.1 clause 4 bitstream packing — Tables 5 / 6 octet maps.
//!
//! Clause 4 of the Recommendation defines the wire format of a speech
//! frame as an octet sequence in which "the bits are with the MSB on the
//! left and the LSB on the right" inside each octet, parameters are
//! packed in the order listed in Tables 5 (high rate, 24 octets) and 6
//! (low rate, 20 octets), and each parameter bit is named `PARx_By`
//! (parameter, subframe `x`, bit `y` counted from 0 = LSB). Reading the
//! octet maps back, the layout is exactly an **LSB-first bit stream**:
//! bit 0 of octet 1 is `RATEFLAG_B0`, bit 1 is `VADFLAG_B0`, bits 2..7
//! are `LPC_B0..LPC_B5`, octet 2 continues with `LPC_B6..LPC_B13`, and
//! so on — every parameter is written least-significant-bit first, in
//! the canonical Table 4 order:
//!
//! ```text
//!   RATEFLAG, VADFLAG, LPC(24),
//!   ACL0(7), ACL1(2), ACL2(7), ACL3(2),
//!   GAIN0..GAIN3(12 each), GRID0..GRID3(1 each),
//!   high rate: UB(1)=0, MSBPOS(13), POS0(16), POS1(14), POS2(16),
//!              POS3(14), PSIG0(6), PSIG1(5), PSIG2(6), PSIG3(5)
//!   low rate:  POS0..POS3(12 each), PSIG0..PSIG3(4 each)
//! ```
//!
//! for a total of 192 bits (24 octets) at the high rate and 160 bits
//! (20 octets) at the low rate. `RATEFLAG_B0 = 0` selects the high rate
//! and `1` the low rate; `VADFLAG_B0 = 0` marks active speech (clause 4;
//! both flags set is reserved). This matches the crate-wide 2-bit
//! discriminator convention (`data[0] & 0b11`: `00` high rate, `01` low
//! rate, `10` SID, `11` untransmitted).
//!
//! # The 13-bit `MSBPOS` word
//!
//! At the high rate the four subframes' pulse-position indices are
//! `C(30,6)`- / `C(30,5)`-combinatorial codes (Table 2 / §2.15):
//! 20-bit codes < 593 775 on even subframes and 18-bit codes < 142 506
//! on odd subframes. Per the Table 2 note, "the 4 MSB of each pulse
//! position index" are combined "into a single 13-bit word", saving
//! 16 − 13 = 3 bits. The combine arithmetic is forced by the index
//! ranges: the top-4-bit digit of an even subframe's code is
//! `code >> 16 ∈ 0..=9` (10 values, since `593 774 >> 16 = 9`) and of an
//! odd subframe's `code >> 14 ∈ 0..=8` (9 values), so the four digits
//! form a mixed-radix (10, 9, 10, 9) number with exactly
//! `10·9·10·9 = 8100 ≤ 2¹³` values. This module packs the digits in
//! subframe order, most significant first:
//!
//! ```text
//!   MSBPOS = ((d0·9 + d1)·10 + d2)·9 + d3
//! ```
//!
//! The digit *order* inside the 13-bit word is the one packing choice
//! Tables 2–5 do not pin down (the Recommendation defers it to the
//! normative clause-5 code, which is outside this crate's clean-room
//! wall); the subframe-major order above is this crate's documented
//! derivation choice and is applied symmetrically on pack and unpack.
//!
//! Likewise, the assignment of individual pulses to bit positions
//! *inside* the `POS` / `PSIG` words at the low rate (four 3-bit track
//! slots and four sign bits) and the sign-bit order at the high rate are
//! specified here as documented conventions (see [`SpecFrameParams`]);
//! the Recommendation's Tables 4–6 treat each parameter as an opaque
//! integer.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::spec_tables::MPMLQ_MAX_POSITION;
use crate::tables::{HIGH_RATE_BYTES, LOW_RATE_BYTES, SUBFRAMES_PER_FRAME};

/// Operating rate of a packed speech frame (clause 4 `RATEFLAG_B0`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PackedRate {
    /// 6.3 kbit/s MP-MLQ — `RATEFLAG_B0 = 0`, 24 octets (Table 5).
    High,
    /// 5.3 kbit/s ACELP — `RATEFLAG_B0 = 1`, 20 octets (Table 6).
    Low,
}

impl PackedRate {
    /// Frame size in octets (Tables 5 / 6).
    pub const fn frame_bytes(self) -> usize {
        match self {
            PackedRate::High => HIGH_RATE_BYTES,
            PackedRate::Low => LOW_RATE_BYTES,
        }
    }
}

/// Number of bits of an even (0, 2) / odd (1, 3) subframe's high-rate
/// combinatorial position code that stay in the `POS` field after the 4
/// MSBs move to `MSBPOS` (Table 4: POS0/2 = 20 bits, POS1/3 = 18 bits).
const HIGH_POS_LSB_BITS: [u32; SUBFRAMES_PER_FRAME] = [16, 14, 16, 14];

/// Mixed-radix digit count of each subframe's `MSBPOS` digit — the
/// even subframes' combinatorial codes reach `593 774 >> 16 = 9` and the
/// odd subframes' `142 505 >> 14 = 8`, giving radices (10, 9, 10, 9).
const MSBPOS_RADIX: [u32; SUBFRAMES_PER_FRAME] = [10, 9, 10, 9];

/// Exclusive upper bound of the 13-bit `MSBPOS` word: 10·9·10·9.
pub const MSBPOS_LIMIT: u32 = 8100;

/// Per-parameter field widths shared by both rates, in stream order.
const ACL_BITS: [u32; SUBFRAMES_PER_FRAME] = [7, 2, 7, 2];
const GAIN_BITS: u32 = 12;
const HIGH_PSIG_BITS: [u32; SUBFRAMES_PER_FRAME] = [6, 5, 6, 5];
const LOW_POS_BITS: u32 = 12;
const LOW_PSIG_BITS: u32 = 4;

/// One G.723.1 speech frame as the Table 4 transmitted-parameter set.
///
/// All values are the raw quantiser indices the Recommendation
/// transmits; no dequantisation is applied at this layer.
///
/// Conventions for the intra-word layouts this crate fixes (see the
/// module docs for why these are derivation choices):
///
/// - **High rate** `pos[s]` holds the *full* combinatorial position
///   code (`< C(30, 6)` on even subframes, `< C(30, 5)` on odd ones);
///   the pack step splits off the 4 MSBs into `MSBPOS` itself.
/// - **High rate** `psig[s]` bit `k` (LSB = bit 0) is the sign of the
///   `k`-th pulse in ascending position order, `0` = positive.
/// - **Low rate** `pos[s]` packs the four §2.16 Table 1 track slots as
///   `slot0 | slot1 << 3 | slot2 << 6 | slot3 << 9` (3 bits per track,
///   track 0 in the least-significant bits).
/// - **Low rate** `psig[s]` bit `t` is the sign of the track-`t` pulse,
///   `0` = positive.
/// - `grid[s]` is the §2.15 / §2.16 grid bit: `0` = even sample
///   positions, `1` = the whole pulse set shifted to odd positions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpecFrameParams {
    /// Operating rate — selects the Table 5 vs Table 6 octet map.
    pub rate: PackedRate,
    /// 24-bit `LPC` LSP split-VQ index (three 8-bit band indices; this
    /// crate stores band 0 in the least-significant byte).
    pub lsp_index: u32,
    /// `ACL0..ACL3`: 7-bit absolute lag index on subframes 0 / 2
    /// (`L = index + 18`, eq. 37), 2-bit differential on 1 / 3 (eq. 38).
    pub acl: [u32; SUBFRAMES_PER_FRAME],
    /// `GAIN0..GAIN3`: 12-bit combined adaptive/fixed gain words
    /// (eq. 36 / 39 / 40).
    pub gain: [u32; SUBFRAMES_PER_FRAME],
    /// `GRID0..GRID3`: 1-bit even/odd pulse grid per subframe.
    pub grid: [u8; SUBFRAMES_PER_FRAME],
    /// `POS0..POS3` pulse-position indices (see struct docs).
    pub pos: [u32; SUBFRAMES_PER_FRAME],
    /// `PSIG0..PSIG3` pulse-sign words (see struct docs).
    pub psig: [u32; SUBFRAMES_PER_FRAME],
}

impl SpecFrameParams {
    /// A silent-ish all-zero-index frame at the given rate. Useful as a
    /// baseline in tests.
    pub fn zeroed(rate: PackedRate) -> Self {
        Self {
            rate,
            lsp_index: 0,
            acl: [0; SUBFRAMES_PER_FRAME],
            gain: [0; SUBFRAMES_PER_FRAME],
            grid: [0; SUBFRAMES_PER_FRAME],
            pos: [0; SUBFRAMES_PER_FRAME],
            psig: [0; SUBFRAMES_PER_FRAME],
        }
    }

    /// Validate every field against its transmitted width / index range.
    fn validate(&self) -> Result<()> {
        if self.lsp_index >> 24 != 0 {
            return Err(Error::invalid("G.723.1 pack: LPC index exceeds 24 bits"));
        }
        for s in 0..SUBFRAMES_PER_FRAME {
            if self.acl[s] >> ACL_BITS[s] != 0 {
                return Err(Error::invalid(format!(
                    "G.723.1 pack: ACL{s} exceeds {} bits",
                    ACL_BITS[s]
                )));
            }
            if self.gain[s] >> GAIN_BITS != 0 {
                return Err(Error::invalid(format!(
                    "G.723.1 pack: GAIN{s} exceeds {GAIN_BITS} bits"
                )));
            }
            if self.grid[s] > 1 {
                return Err(Error::invalid(format!(
                    "G.723.1 pack: GRID{s} exceeds 1 bit"
                )));
            }
            match self.rate {
                PackedRate::High => {
                    if self.pos[s] >= MPMLQ_MAX_POSITION[s] {
                        return Err(Error::invalid(format!(
                            "G.723.1 pack: POS{s} combinatorial code {} out of range (< {})",
                            self.pos[s], MPMLQ_MAX_POSITION[s]
                        )));
                    }
                    if self.psig[s] >> HIGH_PSIG_BITS[s] != 0 {
                        return Err(Error::invalid(format!(
                            "G.723.1 pack: PSIG{s} exceeds {} bits",
                            HIGH_PSIG_BITS[s]
                        )));
                    }
                }
                PackedRate::Low => {
                    if self.pos[s] >> LOW_POS_BITS != 0 {
                        return Err(Error::invalid(format!(
                            "G.723.1 pack: POS{s} exceeds {LOW_POS_BITS} bits"
                        )));
                    }
                    if self.psig[s] >> LOW_PSIG_BITS != 0 {
                        return Err(Error::invalid(format!(
                            "G.723.1 pack: PSIG{s} exceeds {LOW_PSIG_BITS} bits"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Combine the four subframes' top-4-bit digits into the 13-bit
/// `MSBPOS` word (Table 2 note / Table 5 octets 13–14). `pos` holds the
/// full combinatorial codes.
fn msbpos_combine(pos: &[u32; SUBFRAMES_PER_FRAME]) -> u32 {
    let mut w = 0u32;
    for s in 0..SUBFRAMES_PER_FRAME {
        let digit = pos[s] >> HIGH_POS_LSB_BITS[s];
        debug_assert!(digit < MSBPOS_RADIX[s]);
        w = w * MSBPOS_RADIX[s] + digit;
    }
    w
}

/// Split a 13-bit `MSBPOS` word back into the four top-4-bit digits.
/// Returns an error when the word is outside the mixed-radix range
/// (`>= 8100`), which cannot be produced by a conforming packer.
fn msbpos_split(word: u32) -> Result<[u32; SUBFRAMES_PER_FRAME]> {
    if word >= MSBPOS_LIMIT {
        return Err(Error::invalid(format!(
            "G.723.1 unpack: MSBPOS {word} out of range (< {MSBPOS_LIMIT})"
        )));
    }
    let mut digits = [0u32; SUBFRAMES_PER_FRAME];
    let mut w = word;
    for s in (0..SUBFRAMES_PER_FRAME).rev() {
        digits[s] = w % MSBPOS_RADIX[s];
        w /= MSBPOS_RADIX[s];
    }
    debug_assert_eq!(w, 0);
    Ok(digits)
}

/// LSB-first bit writer over a fixed-size frame, mirroring
/// [`BitReader`]'s consumption order.
struct FrameWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl FrameWriter {
    fn new(len: usize) -> Self {
        Self {
            bytes: vec![0u8; len],
            bit_pos: 0,
        }
    }

    /// Append the `n` least-significant bits of `value`, LSB first.
    fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(n == 32 || value >> n == 0);
        for i in 0..n {
            let bit = (value >> i) & 1;
            let byte = self.bit_pos / 8;
            let off = self.bit_pos % 8;
            debug_assert!(byte < self.bytes.len(), "frame writer overflow");
            self.bytes[byte] |= (bit as u8) << off;
            self.bit_pos += 1;
        }
    }

    fn finish(self) -> Vec<u8> {
        debug_assert_eq!(
            self.bit_pos,
            self.bytes.len() * 8,
            "frame writer must fill the frame exactly"
        );
        self.bytes
    }
}

/// Pack one speech frame into its clause-4 octet sequence (Table 5 /
/// Table 6). Returns 24 bytes at the high rate, 20 at the low rate.
pub fn pack_frame(params: &SpecFrameParams) -> Result<Vec<u8>> {
    params.validate()?;
    let mut w = FrameWriter::new(params.rate.frame_bytes());

    // Octet 1 low bits: RATEFLAG_B0 then VADFLAG_B0 (always 0 — this
    // layer only packs active-speech frames; SID belongs to Annex A).
    let rateflag = match params.rate {
        PackedRate::High => 0u32,
        PackedRate::Low => 1u32,
    };
    w.write(rateflag, 1);
    w.write(0, 1); // VADFLAG_B0 = 0 (active speech)
    w.write(params.lsp_index, 24);
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(params.acl[s], ACL_BITS[s]);
    }
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(params.gain[s], GAIN_BITS);
    }
    for s in 0..SUBFRAMES_PER_FRAME {
        w.write(params.grid[s] as u32, 1);
    }
    match params.rate {
        PackedRate::High => {
            w.write(0, 1); // UB — "The unused bit is named UB (value = 0)".
            w.write(msbpos_combine(&params.pos), 13);
            for s in 0..SUBFRAMES_PER_FRAME {
                let lsbs = params.pos[s] & ((1u32 << HIGH_POS_LSB_BITS[s]) - 1);
                w.write(lsbs, HIGH_POS_LSB_BITS[s]);
            }
            for s in 0..SUBFRAMES_PER_FRAME {
                w.write(params.psig[s], HIGH_PSIG_BITS[s]);
            }
        }
        PackedRate::Low => {
            for s in 0..SUBFRAMES_PER_FRAME {
                w.write(params.pos[s], LOW_POS_BITS);
            }
            for s in 0..SUBFRAMES_PER_FRAME {
                w.write(params.psig[s], LOW_PSIG_BITS);
            }
        }
    }
    Ok(w.finish())
}

/// Unpack one clause-4 speech frame. The rate is taken from
/// `RATEFLAG_B0` / `VADFLAG_B0` in the first octet; SID (`10`) and
/// untransmitted (`11`) frames are rejected here — they carry no
/// Table 5/6 payload and are handled by the concealment path upstream.
pub fn unpack_frame(data: &[u8]) -> Result<SpecFrameParams> {
    let mut r = BitReader::new(data);
    let rateflag = r.read_u32(1)?;
    let vadflag = r.read_u32(1)?;
    if vadflag != 0 {
        return Err(Error::invalid(
            "G.723.1 unpack: VADFLAG set (SID / reserved frame, not a speech frame)",
        ));
    }
    let rate = if rateflag == 0 {
        PackedRate::High
    } else {
        PackedRate::Low
    };
    if data.len() < rate.frame_bytes() {
        return Err(Error::invalid(format!(
            "G.723.1 unpack: {} frame needs {} bytes, got {}",
            match rate {
                PackedRate::High => "high-rate",
                PackedRate::Low => "low-rate",
            },
            rate.frame_bytes(),
            data.len()
        )));
    }

    let lsp_index = r.read_u32(24)?;
    let mut acl = [0u32; SUBFRAMES_PER_FRAME];
    for s in 0..SUBFRAMES_PER_FRAME {
        acl[s] = r.read_u32(ACL_BITS[s])?;
    }
    let mut gain = [0u32; SUBFRAMES_PER_FRAME];
    for g in gain.iter_mut() {
        *g = r.read_u32(GAIN_BITS)?;
    }
    let mut grid = [0u8; SUBFRAMES_PER_FRAME];
    for g in grid.iter_mut() {
        *g = r.read_u32(1)? as u8;
    }
    let mut pos = [0u32; SUBFRAMES_PER_FRAME];
    let mut psig = [0u32; SUBFRAMES_PER_FRAME];
    match rate {
        PackedRate::High => {
            let _ub = r.read_u32(1)?; // tolerated on read, 0 on write
            let msbpos = r.read_u32(13)?;
            let digits = msbpos_split(msbpos)?;
            for s in 0..SUBFRAMES_PER_FRAME {
                let lsbs = r.read_u32(HIGH_POS_LSB_BITS[s])?;
                let code = (digits[s] << HIGH_POS_LSB_BITS[s]) | lsbs;
                if code >= MPMLQ_MAX_POSITION[s] {
                    return Err(Error::invalid(format!(
                        "G.723.1 unpack: POS{s} combinatorial code {code} out of range (< {})",
                        MPMLQ_MAX_POSITION[s]
                    )));
                }
                pos[s] = code;
            }
            for s in 0..SUBFRAMES_PER_FRAME {
                psig[s] = r.read_u32(HIGH_PSIG_BITS[s])?;
            }
        }
        PackedRate::Low => {
            for p in pos.iter_mut() {
                *p = r.read_u32(LOW_POS_BITS)?;
            }
            for p in psig.iter_mut() {
                *p = r.read_u32(LOW_PSIG_BITS)?;
            }
        }
    }

    Ok(SpecFrameParams {
        rate,
        lsp_index,
        acl,
        gain,
        grid,
        pos,
        psig,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG for property-style round-trip coverage.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    fn random_params(rng: &mut Lcg, rate: PackedRate) -> SpecFrameParams {
        let mut p = SpecFrameParams::zeroed(rate);
        p.lsp_index = rng.below(1 << 24);
        for s in 0..SUBFRAMES_PER_FRAME {
            p.acl[s] = rng.below(1 << ACL_BITS[s]);
            p.gain[s] = rng.below(1 << GAIN_BITS);
            p.grid[s] = rng.below(2) as u8;
            match rate {
                PackedRate::High => {
                    p.pos[s] = rng.below(MPMLQ_MAX_POSITION[s]);
                    p.psig[s] = rng.below(1 << HIGH_PSIG_BITS[s]);
                }
                PackedRate::Low => {
                    p.pos[s] = rng.below(1 << LOW_POS_BITS);
                    p.psig[s] = rng.below(1 << LOW_PSIG_BITS);
                }
            }
        }
        p
    }

    #[test]
    fn frame_sizes_match_tables_5_and_6() {
        let high = pack_frame(&SpecFrameParams::zeroed(PackedRate::High)).unwrap();
        assert_eq!(high.len(), 24);
        let low = pack_frame(&SpecFrameParams::zeroed(PackedRate::Low)).unwrap();
        assert_eq!(low.len(), 20);
    }

    #[test]
    fn rate_discriminator_matches_crate_convention() {
        let high = pack_frame(&SpecFrameParams::zeroed(PackedRate::High)).unwrap();
        assert_eq!(high[0] & 0b11, 0b00, "high rate ⇒ RATEFLAG=0, VADFLAG=0");
        let low = pack_frame(&SpecFrameParams::zeroed(PackedRate::Low)).unwrap();
        assert_eq!(low[0] & 0b11, 0b01, "low rate ⇒ RATEFLAG=1, VADFLAG=0");
    }

    /// Hand-computed golden octets against the Table 5 rows for the
    /// prefix every frame shares (octets 1–12).
    #[test]
    fn golden_prefix_octets_match_table5_rows() {
        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.lsp_index = 0b1010_1100_0011_0101_0110_1001; // 24 bits
        p.acl = [0b101_1010, 0b10, 0b011_0101, 0b01];
        p.gain = [0xABC, 0x123, 0xF0F, 0x5A5];
        p.grid = [1, 0, 1, 1];
        let f = pack_frame(&p).unwrap();

        // Octet 1: LPC_B5..LPC_B0, VADFLAG_B0, RATEFLAG_B0.
        assert_eq!(f[0], ((p.lsp_index & 0x3F) << 2) as u8);
        // Octet 2: LPC_B13..LPC_B6.
        assert_eq!(f[1], ((p.lsp_index >> 6) & 0xFF) as u8);
        // Octet 3: LPC_B21..LPC_B14.
        assert_eq!(f[2], ((p.lsp_index >> 14) & 0xFF) as u8);
        // Octet 4: ACL0_B5..ACL0_B0, LPC_B23, LPC_B22.
        assert_eq!(f[3], (((p.acl[0] & 0x3F) << 2) | (p.lsp_index >> 22)) as u8);
        // Octet 5: ACL2_B4..ACL2_B0, ACL1_B1, ACL1_B0, ACL0_B6.
        assert_eq!(
            f[4],
            (((p.acl[2] & 0x1F) << 3) | (p.acl[1] << 1) | (p.acl[0] >> 6)) as u8
        );
        // Octet 6: GAIN0_B3..GAIN0_B0, ACL3_B1, ACL3_B0, ACL2_B6, ACL2_B5.
        assert_eq!(
            f[5],
            (((p.gain[0] & 0xF) << 4) | (p.acl[3] << 2) | (p.acl[2] >> 5)) as u8
        );
        // Octet 7: GAIN0_B11..GAIN0_B4.
        assert_eq!(f[6], ((p.gain[0] >> 4) & 0xFF) as u8);
        // Octet 8: GAIN1_B7..GAIN1_B0.
        assert_eq!(f[7], (p.gain[1] & 0xFF) as u8);
        // Octet 9: GAIN2_B3..GAIN2_B0, GAIN1_B11..GAIN1_B8.
        assert_eq!(f[8], (((p.gain[2] & 0xF) << 4) | (p.gain[1] >> 8)) as u8);
        // Octet 10: GAIN2_B11..GAIN2_B4.
        assert_eq!(f[9], ((p.gain[2] >> 4) & 0xFF) as u8);
        // Octet 11: GAIN3_B7..GAIN3_B0.
        assert_eq!(f[10], (p.gain[3] & 0xFF) as u8);
        // Octet 12: GRID3, GRID2, GRID1, GRID0, GAIN3_B11..GAIN3_B8.
        assert_eq!(
            f[11],
            (((p.grid[3] as u32) << 7)
                | ((p.grid[2] as u32) << 6)
                | ((p.grid[1] as u32) << 5)
                | ((p.grid[0] as u32) << 4)
                | (p.gain[3] >> 8)) as u8
        );
    }

    /// Golden check of the high-rate tail: UB, MSBPOS and POS0 straddle
    /// octets 13–14 exactly as Table 5 lists them.
    #[test]
    fn golden_high_rate_msbpos_octets() {
        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        // Choose codes whose 4-MSB digits are (9, 8, 9, 8) — the maxima.
        p.pos = [
            MPMLQ_MAX_POSITION[0] - 1,
            MPMLQ_MAX_POSITION[1] - 1,
            MPMLQ_MAX_POSITION[2] - 1,
            MPMLQ_MAX_POSITION[3] - 1,
        ];
        let f = pack_frame(&p).unwrap();
        let msbpos = ((9 * 9 + 8) * 10 + 9) * 9 + 8; // = 8099
        assert_eq!(msbpos, MSBPOS_LIMIT - 1);
        // Octet 13: MSBPOS_B6..MSBPOS_B0, UB. → bit0 = UB = 0,
        // bits 1..7 = MSBPOS_B0..B6.
        assert_eq!(f[12], ((msbpos & 0x7F) << 1) as u8);
        // Octet 14: POS0_B1, POS0_B0, MSBPOS_B12..MSBPOS_B7.
        let pos0_lsbs = p.pos[0] & 0xFFFF;
        assert_eq!(f[13], ((msbpos >> 7) | ((pos0_lsbs & 0x3) << 6)) as u8);
        // Octet 15: POS0_B9..POS0_B2.
        assert_eq!(f[14], ((pos0_lsbs >> 2) & 0xFF) as u8);
    }

    #[test]
    fn golden_low_rate_pos_octets() {
        let mut p = SpecFrameParams::zeroed(PackedRate::Low);
        p.pos = [0xABC, 0x123, 0xE71, 0x455];
        p.psig = [0b1010, 0b0101, 0b1111, 0b0011];
        let f = pack_frame(&p).unwrap();
        // Octet 13: POS0_B7..POS0_B0.
        assert_eq!(f[12], (p.pos[0] & 0xFF) as u8);
        // Octet 14: POS1_B3..POS1_B0, POS0_B11..POS0_B8.
        assert_eq!(f[13], (((p.pos[1] & 0xF) << 4) | (p.pos[0] >> 8)) as u8);
        // Octet 15: POS1_B11..POS1_B4.
        assert_eq!(f[14], ((p.pos[1] >> 4) & 0xFF) as u8);
        // Octet 16: POS2_B7..POS2_B0.
        assert_eq!(f[15], (p.pos[2] & 0xFF) as u8);
        // Octet 17: POS3_B3..POS3_B0, POS2_B11..POS2_B8.
        assert_eq!(f[16], (((p.pos[3] & 0xF) << 4) | (p.pos[2] >> 8)) as u8);
        // Octet 18: POS3_B11..POS3_B4.
        assert_eq!(f[17], ((p.pos[3] >> 4) & 0xFF) as u8);
        // Octet 19: PSIG1_B3..PSIG1_B0, PSIG0_B3..PSIG0_B0.
        assert_eq!(f[18], ((p.psig[1] << 4) | p.psig[0]) as u8);
        // Octet 20: PSIG3_B3..PSIG3_B0, PSIG2_B3..PSIG2_B0.
        assert_eq!(f[19], ((p.psig[3] << 4) | p.psig[2]) as u8);
    }

    #[test]
    fn round_trip_random_frames_both_rates() {
        let mut rng = Lcg(0x9e3779b97f4a7c15);
        for _ in 0..2000 {
            for rate in [PackedRate::High, PackedRate::Low] {
                let p = random_params(&mut rng, rate);
                let bytes = pack_frame(&p).unwrap();
                assert_eq!(bytes.len(), rate.frame_bytes());
                let q = unpack_frame(&bytes).unwrap();
                assert_eq!(p, q);
            }
        }
    }

    #[test]
    fn msbpos_combine_split_is_bijective_over_digit_space() {
        for d0 in 0..10u32 {
            for d1 in 0..9u32 {
                for d2 in 0..10u32 {
                    for d3 in 0..9u32 {
                        let pos = [d0 << 16, d1 << 14, d2 << 16, d3 << 14];
                        let w = msbpos_combine(&pos);
                        assert!(w < MSBPOS_LIMIT);
                        assert_eq!(msbpos_split(w).unwrap(), [d0, d1, d2, d3]);
                    }
                }
            }
        }
    }

    #[test]
    fn unpack_rejects_out_of_range_msbpos() {
        let mut f = pack_frame(&SpecFrameParams::zeroed(PackedRate::High)).unwrap();
        // Force MSBPOS = 8100 (first out-of-range word): bits 97..109.
        // Octet 13 bits 1..7 = MSBPOS_B0..B6, octet 14 bits 0..5 = B7..B12.
        let bad = MSBPOS_LIMIT;
        f[12] = ((bad & 0x7F) << 1) as u8;
        f[13] = (bad >> 7) as u8 & 0x3F;
        assert!(unpack_frame(&f).is_err());
    }

    #[test]
    fn unpack_rejects_out_of_range_combinatorial_code() {
        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.pos = [MPMLQ_MAX_POSITION[0] - 1, 0, 0, 0];
        let mut f = pack_frame(&p).unwrap();
        // POS0's 16 LSBs live in octets 14 (2 bits) + 15 (8) + 16 (6).
        // Force them to all-ones so code = (9 << 16) | 0xFFFF ≥ 593775.
        f[13] |= 0b1100_0000;
        f[14] = 0xFF;
        f[15] |= 0x3F;
        assert!(unpack_frame(&f).is_err());
    }

    #[test]
    fn unpack_rejects_sid_and_short_frames() {
        // VADFLAG set → SID / reserved.
        assert!(unpack_frame(&[0b10; 24]).is_err());
        assert!(unpack_frame(&[0b11; 24]).is_err());
        // Truncated high-rate frame.
        let f = pack_frame(&SpecFrameParams::zeroed(PackedRate::High)).unwrap();
        assert!(unpack_frame(&f[..23]).is_err());
        // Truncated low-rate frame.
        let f = pack_frame(&SpecFrameParams::zeroed(PackedRate::Low)).unwrap();
        assert!(unpack_frame(&f[..19]).is_err());
    }

    #[test]
    fn pack_rejects_out_of_range_fields() {
        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.lsp_index = 1 << 24;
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.acl[0] = 128;
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.pos[0] = MPMLQ_MAX_POSITION[0];
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::Low);
        p.pos[1] = 1 << 12;
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::Low);
        p.psig[2] = 1 << 4;
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.gain[3] = 1 << 12;
        assert!(pack_frame(&p).is_err());

        let mut p = SpecFrameParams::zeroed(PackedRate::High);
        p.grid[1] = 2;
        assert!(pack_frame(&p).is_err());
    }

    /// Every total must land exactly on the Table 2/3 bit budget:
    /// 189 + VAD + RATE + UB = 192 (high), 158 + VAD + RATE = 160 (low).
    #[test]
    fn bit_budget_matches_tables_2_and_3() {
        let acl_total: u32 = ACL_BITS.iter().sum();
        // High rate: 24 + 18 + 48 + 4 grids + 13 MSBPOS + (16+14+16+14)
        // POS LSBs + (6+5+6+5) PSIG = 189.
        let pos_total: u32 = HIGH_POS_LSB_BITS.iter().sum();
        let psig_total: u32 = HIGH_PSIG_BITS.iter().sum();
        assert_eq!(
            24 + acl_total + 4 * GAIN_BITS + 4 + 13 + pos_total + psig_total,
            189
        );
        // Low rate: 24 + 18 + 48 + 4 + 48 + 16 = 158.
        assert_eq!(
            24 + acl_total + 4 * GAIN_BITS + 4 + 4 * LOW_POS_BITS + 4 * LOW_PSIG_BITS,
            158
        );
    }
}
