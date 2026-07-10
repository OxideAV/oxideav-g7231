# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

- **Fuzz hardening (r406)**: adversarial LSP indices can drive the MA
  predictor + codebook sum negative before the §2.6 ordering repair
  reaches line 0; the fixed-point cosine lookup now clamps to its
  table domain instead of panicking on the debug assertion (found by
  the `decode` and `bitstream` fuzz targets after the band-order fix
  re-mapped their input space; both crash inputs pinned as regression
  seeds under `fuzz/seeds/`, plus a unit test). All four targets
  re-run clean under ASan with the r406 pipeline (roundtrip 150 s,
  decode/bitstream 120 s each, params 60 s).

- **Three vector-arbitrated interop corrections (r406) — decoder
  fidelity and encoder agreement transformed**:
  1. **LSP band order in the 24-bit LPC word**: band 0 (lines 0–2)
     lives in the *most*-significant byte, not the least. The
     clause-4 tables treat the field as one opaque number, so the
     intra-word order is a derivation choice — and every reference
     stream decodes the two edge bands through the wrong codebooks
     under the old reading (band 1, positionally symmetric, was
     unaffected: the tell that exposed the swap).
  2. **§2.2 framer alignment**: the encoder codes the input stream
     *delayed by one subframe* — the frame built from input block k
     covers stream samples `[k·240 − 60, k·240 + 180)`. This is how
     the spec's 7.5 ms lookahead / 37.5 ms total delay is realised;
     at this offset (and no other) the encoder's LSP decisions lock
     to the reference. The r405-interim lookahead-parameter API is
     gone again: `SpecEncoder::encode_frame(pcm)` and the registry
     encoder are back to plain 240-sample blocks with the delay
     handled internally.
  3. **Output scale**: the synthesis output is emitted *unshifted*
     and the excitation rail is plain Word16 — r391's halved-output
     stage and ±65534 rail were compensating the band-swapped LSP
     distortion (whole-file least-squares scale vs the reference was
     exactly 2.0). Float path: fixed-codebook levels are the doubled
     table amplitude (`2·q/32768`); fixed path: `SYN_OUT_SHIFT` 1→0,
     `EXC_RAIL` 65534→32767.

  Measured decoder tracking (fixed-point pipeline, whole-file
  corr / SNR): PATHD53 0.60 / +1.9 dB → **1.0000 / +54.4 dB**
  (max |Δ| = 27 LSB), OVERD53 0.48 / +0.3 → **0.9993 / +28.1**,
  INEQD53 0.83 / +4.5 → **0.9811 / +12.5** (50.8% of samples exact),
  PATHD63P 0.77 / +3.5 → **0.9123 / +7.6**, OVERD63P 0.40 / +0.8 →
  **0.9715 / +12.5**, TAMED63P 0.11 / −2.2 → **0.9643 / +11.5**.
  Encoder parameter agreement against the reference `.RCO` streams
  (whole files): LSP word 0% → 77–90.8% on the PATH/OVER/CODE
  classes, ACL0/2 within ±1 up to 100%, fixed-gain index up to 81%.
  All conformance floors re-pinned at the new levels; a new
  full-corpus encoder-agreement test replaces the old self-validity
  spot check. Round-trip PSNR (2 s voiced, release): ACELP 23.9 →
  25.2 dB, MP-MLQ 26.4 → 31.2 dB.

- **Weighted-domain analysis chain (§2.8–§2.13, §2.19) — the
  encoder's closed-loop searches now run where the spec puts them
  (r406)**: per-subframe §2.8 formant perceptual weighting
  `W_i(z) = A(z/0.9)/A(z/0.5)` on the *unquantised* LPC (published
  Q15 tap-weight tables); §2.9 two half-frame open-loop pitch
  estimates on the weighted speech (eq. 12 cross-correlation with
  the smaller-lag preference — a lag ≥ 18 above the incumbent must
  win by 1.25 dB); §2.11 harmonic noise shaping
  `P_i(z) = 1 − β·z^−L` searched in `[L_OL ± 3]` with the
  positive-correlation restriction, `β = 0.3125·G_opt` gated by the
  eq. 17 2.0 dB prediction-gain test; §2.12 impulse response of the
  *combined* filter `S_i(z) = Ã_i(z)·W_i(z)·P_i(z)`; §2.13 ringing
  subtraction of its zero-input response (`t[n] = w[n] − z[n]`); and
  the §2.19 memory update passing the reconstructed excitation
  through `S_i(z)` per subframe. The §2.14 closed-loop lag
  candidates now come from the §2.9 open-loop estimates (±1 on
  subframes 0/2) instead of a per-subframe ad-hoc filtered-history
  search; the ACB/FCB searches are otherwise unchanged but operate
  on the weighted target with the combined impulse response. New
  structural tests: weighting-cascade identity, open-loop
  fundamental-vs-multiple preference, HNS periodicity gating, and
  the ZIR + h∗v superposition identity the search decomposition
  relies on. Measured against the ITU encoder `.RCO` references,
  exact ACL0/ACL2 agreement: OVERC63 25→65%, INEQC53 19→85%,
  OVERC53H 9.5→81%, TAMEC63H 9.5→48%, CODEC63 11→19.5% (±1: 47%),
  PATHC63H 5.6→16.7%, PATHC53 4.9→15.8%.

- **§2.4/§2.5 encoder LPC analysis rebuilt on the spec windows
  (r406)**: four LPC sets per frame, each from the published Q15
  Hamming window over 180 samples *centered on its subframe* — which
  introduces the Recommendation's 7.5 ms (60-sample) encoder
  lookahead (total one-way delay now 37.5 ms; the registry encoder
  buffers one subframe of lookahead and `SpecEncoder::encode_frame`
  takes the next frame's first 60 samples, zero-padded at end of
  stream). Autocorrelation now carries the spec's `1025/1024`
  white-noise correction and the published Q15 binomial lag window;
  the transmitted set is A3(z) after the §2.5 7.5 Hz bandwidth
  expansion (published Q15 per-tap weights).
- **Two silent LPC→LSP conversion bugs fixed** (predating this
  round; found by ITU-vector disagreement, pinned by a new
  LSP→LPC→LSP roundtrip test): the sum/difference polynomial
  deflation seeded its recursion with 0 instead of the leading
  deflated coefficient, corrupting every subsequent coefficient; and
  the Chebyshev root search evaluated `Σ c_k·T_k(x)` where the
  symmetric deflated halves evaluate as
  `2·Σ c_k·T_{deg−k}(x) + c_deg` (reversed order, half-weight
  constant term) — the root finder was solving the wrong polynomial.
  The root scan now walks 1024 angle-uniform grid cells (was 200
  cosine-uniform), Levinson-Durbin terminates early on collapsed
  prediction error (pure-sine inputs) keeping the valid lower-order
  model instead of bailing to `A(z) = 1`, and a failed root search
  falls back to the previous frame's unquantised LSP vector rather
  than a fabricated uniform spread. Unquantised-LSP distance to the
  reference decoder's decoded LSPs on `PATHC53` drops 4× (mean 4761
  → 1190 Q15 units); LSP band-index agreement with the reference
  `.RCO` streams moves from ~0% to 16–50% on the middle band.

- **Fixed-point decode-chain rebuild (r391)** — the registry decoder
  now runs a saturating integer pipeline (`qdec::QSynthesis`) end to
  end: Q15 LSP inverse quantisation / stability / interpolation, Q14
  cosine lookup -> Q13 LPC, wide-accumulator excitation reconstruction,
  §3.6 pitch post-filter, §3.7 synthesis, §3.8 formant + tilt, §3.9
  AGC and §3.10 concealment, all on `basicop`-style saturating
  arithmetic (new `basicop` module: add/sub/mult/l_mult/l_mac/shifts/
  norm/div_s/isqrt64 with DSP saturation semantics).
- **Three excitation-model corrections arbitrated by the ITU
  conformance vectors** (clause 1-4 prose leaves them open; the
  clause-5 C stays outside the wall — model pinned by deconvolving the
  `PATHD53` reference output with the decoded LPC):
  1. eq. 41.1 `e'` is the *contiguous* history slice from `e[-L-2]`
     (the literal `(n mod L)` reading skips two samples and leads the
     reference by exactly two samples);
  2. fixed-codebook pulses land at **twice** the published gain-table
     level with the synthesis output halved on emission (deconvolved
     pulses match sample-exact);
  3. the gain-vector rows act at an effective **/16384** — the /8192
     reading makes the pitch loop diverge where the reference stays
     bounded. Applied to both the fixed and float paths (the float
     `OVERD53` corr 0.97 of r388 is retired as a clipping artifact:
     that model's unclamped excitation reached 2^36 by frame 7 and the
     fully-clipped output merely sign-matched the reference).
- **Measured decoder tracking (whole-file corr / SNR)** against the
  reference `.ROU` outputs, fixed pipeline: PATHD53 0.60 / +1.9 dB,
  OVERD53 0.48 / +0.3 dB, INEQD53 0.83 / +4.5 dB, PATHD63P 0.77 /
  +3.5 dB, OVERD63P 0.40 / +0.8 dB, TAMED63P 0.11 / -2.2 dB — five of
  six streams up from deeply negative SNR (e.g. PATHD63P -12.8 dB,
  INEQD53 -23.1 dB); floors pinned for all six. Remaining gap to
  bit-exactness: the reference's overflow/scaling protocol on the
  OVER/TAME torture classes is specified only by the clause-5 C.

- **ITU conformance vectors (r388)** — first round against the newly
  staged official G.723.1 digital test sequences
  (`docs/audio/g7231/conformance/`, black-box I/O pairs; the clause-5
  reference C stays outside the clean-room wall):
  - clause-4 wire format **confirmed against the reference bitstreams**:
    all 2 816 frames of the 13 main-body `.RCO`/`.TCO` streams unpack +
    repack byte-identically (the crate's MSBPOS mixed-radix combine,
    `C(30,M)` combinatorial codec and Table 5/6 layout are the
    reference's own), except the three deliberate transmission-error
    frames of `PATHD63P.TCO`, which correctly fail field validation.
    Decoded pulse positions verified against Â(z)-deconvolved reference
    excitation (positions, Dirac trains and the 13-bit MSBPOS split all
    land where the reference decoder puts them).
  - **PSIG conventions arbitrated by the vectors**: high rate =
    MSB-first over ascending pulse order with set bit = negative
    (pinned by cold-start `OVERD63P`/`PATHD63P` frames); low rate =
    bit `t` per track with set bit = positive (flips whole-file
    `OVERD53` correlation from −0.97 to **+0.97**). Encoder writers
    updated symmetrically; low-rate track *order* remains
    vector-underdetermined at float fidelity (identity vs reversed
    differ by <0.001 corr on OVERD53).
  - **§3.1 decoder chain restructure**: the §3.6 pitch post-filter now
    runs in the **excitation domain** (eq. 42–47 with the forward-reach
    availability rule, the 1.25 dB prediction-gain gate and the
    attenuate-only eq. 47 `g_p`) and feeds the §3.7 synthesis filter,
    with §3.8/§3.9 formant + tilt + AGC on the synthesis output — per
    the §3.1 block diagram. The old synthesis-domain LTP stage is gone.
  - **Device-under-test switches**: `SynthesisState::set_postfilter()`
    (the `..D53` vectors run post-filter OFF, `..D63P` ON) and the new
    §2.3 input high-pass filter (eq. 1, default ON per §2.2) with
    `encoder::SpecEncoder` exposing rate + HP controls.
  - content-invalid but size-valid frames now conceal like erasures
    (§3.10) instead of erroring the stream — the behaviour the
    PATHD63P transmission-error frames demand.
  - committed harness `tests/itu_conformance.rs` (skips vacuously
    without the corpus): wire-format identity, decoder floors
    (OVERD53 corr ≥ 0.95 / mean-frame ≥ 0.97 / SNR ≥ 5 dB, measured
    0.973 / 0.985 / 7.3 dB; OVERD63P cold-start frames 0–3 corr
    0.605/0.887/0.833/0.738), CRC-driven full decodes with exact
    sample budgets, encoder self-validity on the ITU inputs.
  - known gap: the `OVER..`/`TAME..` classes drive sustained Word16
    saturation chains that only a bit-exact fixed-point pipeline
    reproduces — long-range bit-exactness needs a Q15 basic-ops
    rebuild of the analysis/synthesis kernels (float tracking was
    measurably *hurt* by emulating per-sample saturation).

### Earlier unreleased work

- decoder robustness battery at the public API (integration tests):
  1500 randomly-drawn *valid* clause-4 frames (both rates, extreme
  combinatorial codes / gain words / degenerate LSP words) decode
  through the registered decoder without error; §1.2 mid-stream rate
  switching (interleaved real 24-/20-byte frames through one stateful
  decoder) decodes cleanly; and 2000 free-form random bodies with legal
  discriminators either decode or are rejected (`Err`) — both outcomes
  asserted to occur, proving the MSBPOS/combinatorial validation path
  and the decode path are both live. Complements the nightly-only ASan
  fuzz targets on stable `cargo test`.
- documentation sweep for the spec-layout flip: crate/encoder module
  docstrings now describe the clause-4 pipeline (and cite the in-repo
  03/96 edition), the README rewrites the implementation + interop
  sections around the Table 5/6 wire format with the three remaining
  caveats spelled out (MSBPOS digit order, intra-word pulse/sign
  conventions, float-vs-normative-fixed-point with no conformance
  vectors staged), the round-trip table carries the new 23.9 / 26.4 dB
  figures, and the bitstream fuzz target's prose references the
  clause-4 field map (its 96-bit shared-prefix truncation boundaries
  already matched).
- removed the retired interim wire format (−1100 LOC): the clean-room
  factorial-scalar LSP split VQ, the 4+7+1-bit joint gain codec, the
  per-pulse MP-MLQ position/sign words, the internal field tables and
  LSB bit writer, the legacy `analyse_acelp` / `analyse_mpmlq` /
  `synthesise` paths and their tests. The crate-level `dead_code` allow
  is gone; the `tables.rs` pseudo-"Annex B" field-width arrays (which
  never matched the real Tables 5/6) are deleted in favour of
  `linepack`. Retained shared DSP (LPC/LSP conversion, stability,
  Table 1 ACELP search, postfilters, concealment) is unchanged.
- **the wire format is now the ITU-T clause-4 spec layout** at both
  rates: `emit_frame` routes through `analyse_spec` + the Table 5/6
  packer, and `decode_acelp` / `decode_mpmlq` unpack clause-4 frames
  and run the spec-table §3.1 pipeline. Frames now carry the published
  quantiser indices (24-bit §2.5 split-VQ LSP word, eq. 37/38 lag
  indices, eq. 36/39/40 combined 12-bit gain words over the published
  85/170-row tap codebooks + 24-level gain table, `C(30,M)`
  combinatorial MP-MLQ positions with the 13-bit MSBPOS word, Table 1
  ACELP position words) in the exact Table 5/6 octet layout. Public
  signatures (`make_encoder`, `Decoder`, `decode_*_local`) are
  unchanged; the frame sizes and the 2-bit discriminator convention
  were already spec-true. Round-trip quality *improves* on the 2 s
  voiced integration signal (release): ACELP 23.9 dB PSNR (was
  ~19–20), MP-MLQ 26.4 dB (was ~22–23); integration floors raised
  16 → 20 dB and 19 → 22 dB. Streams produced by earlier releases of
  this crate (the interim clean-room layout) no longer decode — that
  layout was explicitly documented as non-interoperable.
- spec-layout encoder analysis: `AnalysisState::analyse_spec` produces
  a clause-4 `SpecFrameParams` set through the §2 pipeline on the
  published tables. LSP: §2.5 predictive split VQ. Pitch (§2.14):
  closed-loop lag candidates around the per-subframe open-loop estimate
  (±1 on subframes 0/2, the −1..+2 delta window on 1/3) searched
  *jointly* with the 85-/170-row gain-vector codebook by maximising the
  error reduction `2·βᵀd − βᵀRβ` over the filtered eq. 41 basis
  vectors (the all-zero row 0 guarantees a non-negative optimum).
  MP-MLQ (§2.15): eq. 24/25 `G_max` estimate, the
  `[Ĝ−3.2 dB, Ĝ+6.4 dB]` quantised-gain neighbourhood × both grids ×
  Dirac-train mode on short reference lags, sequential per-pulse
  placement against the (train-extended) impulse response, and an exact
  24-level MMSE gain re-pick per pattern. ACELP (§2.16): the Table 1
  coordinate-descent search against the pitch-enhanced impulse
  response (`h′[n] = h[n] + β·h′[n−L−ε]`, the §2.16 pre-search
  modification), least-squares gain with sign folding into the pulse
  signs, exact 24-level MMSE quantisation. The shadow decoder commits
  through `decode_spec_params` itself, keeping encoder and decoder in
  bit-exact lockstep (pinned by a new test on `prev_lsp_freq` +
  `exc_history`). New `spec_exc::acb_basis` /
  `acelp_enhanced_impulse_response` helpers. Spec-format round-trip
  PSNR on the 20-frame voiced signal (release): ACELP 13.8 dB,
  MP-MLQ 15.7 dB — coarser than the legacy clean-room format's
  17/21 dB because the published 24-step (3.2 dB) innovation-gain
  table trades gain resolution for the richer 5-tap pitch VQ. 4 new
  tests: PSNR floors at both rates through pack→unpack→decode,
  encoder/decoder lockstep, decodable-lag emission.
- spec-layout decoder kernel: `SynthesisState::decode_spec_params`
  runs the full §3.1 pipeline on a clause-4 `SpecFrameParams` set —
  spec LSP decode (§3.2 → 2.6) with §2.6-step-3 previous-vector
  fallback, eq. 37/38 lag decode, the eq. 39/40 gain-word split per
  subframe pair (`lag_base` = L0/L2 driving the 85-row rule), the
  eq. 41 fifth-order adaptive-codebook contribution, rate-specific
  fixed-codebook reconstruction (MP-MLQ combinatorial + Dirac-train /
  ACELP Table 1 + 1-tap enhancement), per-subframe interpolated LPC
  synthesis, and the existing §3.6/§3.8/§3.9 postfilter chain. New
  `prev_lsp_freq` decoder state carries the §2.6 MA-predictor vector in
  the tables' Q15 normalised-frequency domain, cold-started at `p_DC`
  (§3.11) and kept in lockstep through the §3.10.1 erasure
  extrapolation. Concealment bookkeeping gains a spec-path variant
  (`record_last_frame_spec`) feeding the §3.10.2 classifier from
  decoded taps/gains. 5 new tests: determinism + audible energy at both
  rates, near-silent zero frames, linepack pack→unpack→decode
  composition, LSP predictor-state advancement, and short-lag
  train-mode energy extension.
- spec excitation-parameter codecs (`spec_exc`): the combined 12-bit
  gain word decoded/encoded per eq. 36 / 39 / 40 — `PGIndex`/`MGIndex`
  split with `GSize = 24`, the §2.14 codebook-selection rule (170-entry
  shared codebook; 85-entry + impulse-train MSB only for the high rate
  when the subframe pair's reference lag is < 58), and clamping of
  non-conforming rows; the fifth-order adaptive-codebook contribution
  `u[n]` per eq. 41.1–41.2 including both wrap-around seeds
  `e′[0] = e[−L−2]`, `e′[1] = e[−L−1]` and the modular periodic
  extension; MP-MLQ fixed-vector reconstruction (§2.15 / §2.17 —
  combinatorial position decode, grid, ascending-order sign bits,
  short-lag Dirac-train mode at the reference period); ACELP
  fixed-vector reconstruction (§2.16 Table 1 direct position decode
  with absent-slot handling) and the §2.16 pitch-synchronous 1-tap
  enhancement `v[n] += β(PGIndex)·v[n−L−ε(PGIndex)]` applied
  recursively in ascending n (β Q15; zero-β rows and the selector
  sentinel disable it). Gain-row tap scale pinned to Q13 with the
  remaining 15 row entries identified (and unit-tested) as the
  precomputed −2·βᵢ² / −2·βᵢβⱼ closed-loop search energies. 8 unit
  tests cover every layout branch and the eq. 41 geometry.
- spec LSP codec (`spec_lsp`): §2.5 / §2.6 predictive split vector
  quantiser on the published tables. Encode implements steps 2–5 of
  §2.5 — DC removal (eq. 4.3), the fixed first-order MA predictor
  `b = 12/32` on the previously *decoded* vector (eq. 3.3), the 3+3+4
  split (eq. 4.1), and the 256-entry weighted-MSE codebook search per
  band with the eq. 5 inverse-neighbour-gap diagonal weights. Decode is
  the §2.6 inverse (`p̃ = p̄ + p_DC + ẽ`). All arithmetic runs in the
  tables' native Q15 normalised-frequency domain (`ω = π·q/32768`),
  established numerically: the band codebooks add directly onto the DC
  scale (a 4× Q13→Q15 rescale would drive line 2 negative on early
  rows). Conversions to/from the synthesis pipeline's cosine domain +
  the 24-bit LPC-word split (band 0 in the low byte — documented crate
  convention) included. 7 unit tests: index round trip, DC ordering,
  cosine conversion, manual §2.6 reconstruction, near-DC decode of the
  DC vector, 200-frame quantise→decode consistency with a 1024-unit
  (250 Hz) worst-line bound, and an aggregate held-vector predictor
  check.
- clause-4 bitstream packing layer (`linepack`): spec-layout Table 5 /
  Table 6 octet maps for both rates as `SpecFrameParams` ⇄ 24- / 20-byte
  frames. The octet maps pin the layout to an LSB-first bit stream in
  canonical Table 4 parameter order (RATEFLAG/VADFLAG flags, 24-bit LPC,
  7/2/7/2 ACL, 4×12 GAIN, 4×1 GRID, then the rate-specific POS/PSIG
  tail with the high-rate UB pad bit), verified octet-by-octet against
  the published rows with golden tests. The 13-bit `MSBPOS` word is
  implemented as the mixed-radix (10, 9, 10, 9) combine forced by the
  `C(30,6)` / `C(30,5)` index ranges (10·9·10·9 = 8100 ≤ 2¹³ — exactly
  the "3 additional bits are saved" of the Table 2 note), with the
  subframe-major digit order documented as a derivation choice. Pack
  validates every field range (combinatorial codes < 593775 / 142506);
  unpack rejects SID/reserved flag combinations, short frames,
  out-of-range MSBPOS words and position codes. 12 unit tests: golden
  octets for the shared prefix (Table 5 octets 1–12), the MSBPOS
  straddle (octets 13–15), the low-rate tail (Table 6 octets 13–20),
  a 4000-frame random round-trip at both rates, full-digit-space
  MSBPOS bijectivity, and the Table 2/3 bit budgets (189 / 158).
- fuzz harness hardening: untracked `fuzz/Cargo.lock` (cargo-fuzz
  regenerates it; folded back under the library `Cargo.lock` ignore),
  added a version-controlled seed corpus under `fuzz/seeds/<target>/`
  (mixed-rate packet streams, sustained erasure runs, both encoder
  rates, saturated/alternating field bodies, field-boundary
  truncations) plus a `fuzz/README.md`, and added a fourth fuzz target
  `params` covering the `make_encoder` / `make_decoder` parameter-
  validation surface (rejected sample rates / channels / formats,
  bit-rate→mode acceptance-window edges) and a focused sustained
  SID / untransmitted erasure-concealment decay→recovery→reset drive
  (§3.10.1 / §3.10.2). All four targets fuzz panic-free.
- spec-faithful MP-MLQ combinatorial position codec (§2.15 / §2.17
  `Fcbk_Pack` / `Fcbk_Unpk` position-index half): `fcbk_pack_positions` /
  `fcbk_unpk_positions` implement the `C(30, M)` combinatorial number system
  whose per-step weights are the published `MPMLQ_COMBINATORIAL` table.
  Exhaustively verified bijective over both complete codeword spaces
  (`C(30, 5) = 142 506`, `C(30, 6) = 593 775`) and proven order-preserving;
  the table's exact closed form `C(29 − c, 5 − r)` is now pinned. The 13-bit
  `MSBPOS` 4-MSB recombination remains a documented clean-room gap.
- formant postfilter (§3.8 / §2.18) now scales each LPC tap by the spec's
  exact Q15 `PostFiltZeroTable` / `PostFiltPoleTable` weights (the
  fixed-point γ₁ = 0.65 / γ₂ = 0.75 powers from `spec_tables`) instead of
  recomputing a repeatedly-multiplied float `gamma^i`, so the weighting
  matches the ITU §2.18 table verbatim and avoids tap-by-tap rounding drift
- §3.6 pitch postfilter forward LTP reach now spans the whole-frame
  synthesis signal (trace §8) instead of truncating the correlation
  window at the subframe boundary

## [0.0.8](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.7...v0.0.8) - 2026-06-15

### Other

- ACELP fixed codebook follows §2.16 Table 1 pulse geometry
- erasure LSP leaks toward DC vector (§3.10.1) + §3.11 cold start
- formant postfilter uses §2.7 interpolated LPC (round 296)
- add roundtrip (encode→decode) + bitstream (structured parser) targets — round 286
- low-rate ACELP track geometry + gain-word split accessors (r273)
- typed accessors + deeper invariant tests (round 265)
- drop release-plz.toml — use release-plz defaults across the workspace
- exempt fuzz/Cargo.lock from the library-level Cargo.lock block
- add cargo-fuzz harness on the decoder's attacker surface
- spec-shape tilt + AGC per G.723.1 §3.8 / 3.9
- spec-shape frame-erasure path per G.723.1 §3.10.2
- spec-shape LSP stability per G.723.1 §3.1 / 2.6
- spec-shape pitch postfilter per G.723.1 §3.6
- add Criterion bench harness (encode/decode/roundtrip, both rates)
- land ITU-T G.723.1 spec-table data (27 tables, 17 invariant tests)

### Changed

- The 5.3 kbit/s ACELP fixed-codebook pulse positions now follow ITU-T
  G.723.1 §2.16 Table 1 exactly: four tracks on even bases `0, 2, 4, 6`
  with stride 8, the 1-bit grid acting as the global "+1 odd shift", and
  the last slot of tracks 2 / 3 (sample 60 / 62) correctly signifying an
  absent pulse. The earlier layout used bases `0,1,2,3` with a `+4` grid
  offset — internally consistent but not the Table 1 structure. Both the
  encoder's coordinate-descent search (`acelp_4pulse_search`) and the
  decoder's pulse placement (`place_pulses`) now route through one helper
  (`acelp_pos_of`) wrapping the typed `spec_tables::acelp_track_position`
  accessor, so encode and decode share a single Table-1-faithful geometry.
  Round-trip PSNR is unchanged inside its band (ACELP ≈ 17.1 dB on the 2 s
  voiced signal); a new unit test pins every Table 1 base, the stride-8
  progression, the `(60)`/`(62)` absent-pulse slots, and the encode/decode
  geometry agreement (round 312).
- Frame-erasure LSP concealment now implements §3.10.1's predictor-based
  extrapolation toward the long-term DC vector instead of freezing the last
  good vector, and the decoder cold-starts the previous-frame LSP at the DC
  vector `p_DC` per §3.11 (round 302). With the decoded residual `ẽ_n` set
  to zero and the erasure predictor `b_e = 23/32`, the concealed LSP becomes
  `p̃_n = b_e·(p̃_{n-1} − p_DC) + p_DC` — a per-frame leak of every LSP
  angular frequency a fraction `1 − b_e = 9/32` toward its DC value, applied
  by the new `extrapolate_lsp_toward_dc` helper before the wider-`Δ_min`
  (§3.10.1) ordering procedure. The extrapolated vector is persisted as the
  previous LSP, so a sustained erasure run relaxes the spectral envelope
  monotonically toward the long-term mean and a recovering good frame
  interpolates from the concealed envelope. `SynthesisState::new` now seeds
  `prev_lsp` from `tables::lsp_dc_cosines()` (derived from the canonical Q15
  `spec_tables::LSP_DC_PREDICTED_FREQ_Q15`) rather than an evenly-spaced
  placeholder. New constants `LSP_PREDICTOR_B` (12/32) and `LSP_PREDICTOR_BE`
  (23/32). Three unit tests pin the behaviour (cold-start equals `p_DC` and
  is strictly ordered; extrapolation hits the exact convex combination,
  never overshoots, and has `p_DC` as a fixed point; sustained erasure
  strictly reduces the angular-frequency distance to `p_DC`). Round-trip
  PSNR on clean streams is unchanged.

- Formant postfilter (§3.8) now uses the §3.3 / §2.7 (eq. 8) per-subframe
  interpolated synthesis filter `Ã_i(z)` instead of a frame-constant LSP
  (round 296). Previously `apply_post_filter` passed `lsp_q` as both the
  previous and current LSP to the interpolation, degenerating it to the
  current frame's LSP for every subframe — a deliberate simplification.
  The decoder entry points (`decode_acelp` / `decode_mpmlq`) now capture
  the previous frame's decoded LSP before `synthesise` advances
  `self.prev_lsp`, and thread it through so the postfilter reproduces the
  exact (0.75/0.25), (0.5/0.5), (0.25/0.75), (0/1) interpolation curve the
  LPC synthesis stage used, subframe-for-subframe. Round-trip PSNR on the
  quasi-stationary integration signal is unchanged (ACELP 17.58 dB,
  MP-MLQ 20.72 dB); the alignment matters across voiced transitions where
  the LSP moves frame-to-frame and the previous formant-constant filter
  diverged from the synthesis filter. Two unit tests pin the behaviour:
  the existing no-panic test moves to the new signature, and a new test
  confirms that distinct previous/current LSP vectors change the early
  (prev-weighted) subframes while leaving the last subframe (weight 0/1
  on prev) less affected.

### Added

- Two new `cargo-fuzz` ASan targets extending the round-236 `decode`
  fuzzer (round 286). `roundtrip` is a closed-loop encode → decode
  fuzzer: it drives arbitrary 16-bit PCM (full-scale square waves,
  all-`i16::MIN` blocks, ramps, silence) through the registered
  `Encoder` at a fuzzer-chosen rate — exercising the analysis path
  (autocorrelation, Levinson-Durbin, Chebyshev LSP root-finding,
  closed-loop pitch + FCB search, joint-gain quant, §2.2 frame
  assembly) on input the bench harness never feeds it — then routes
  every emitted packet back through the `Decoder`, covering mid-stream
  + idempotent `flush()` and reverse-order packet delivery.
  `bitstream` is a structured parser-corruption fuzzer: it builds
  structurally near-legal frames (correct length + rate byte), then
  surgically corrupts one field (LSP split index, abs/delta lag, gain
  word, FCB pulse word, MP-MLQ reserved tail) or truncates the payload
  at an exact field boundary to probe the `BitReader::read_u32`
  out-of-bits guard on sub-byte remainders. It drives
  `header::parse_frame_type`, a direct field-shaped `BitReader`
  schedule, the stateless `decode_{acelp,mpmlq}_local` per-rate
  decoders, and a chained `Decoder::send_packet` sequence so field
  corruption is also seen by the cross-frame postfilter + erasure
  state. Round-286 ASan campaigns (seeded from the in-tree corpus):
  `decode` ≈1.15 M runs, `bitstream` ≈394 K runs, `roundtrip` ≈22 K
  runs — ~1.56 M executions, no crashes, leaks, OOM, or artifacts.
- Low-rate ACELP algebraic-codebook geometry + gain-word split
  accessors on the staged spec-table data (round 273). `spec_tables`
  now surfaces Table 1/G.723.1 as the typed `AcelpTrack`
  (`Track0..Track3`) enum plus `acelp_track_position(track, idx,
  shift)`, reproducing the four even-based stride-8 pulse tracks and
  the 1-bit odd shift, and returning `None` for the boundary "(60)" /
  "(62)" candidates that signify an absent pulse. The 1-tap LTP
  short-pitch shortcut (§2.16) is exposed as `Pitch1TapLtp { gain,
  selector }` via `pitch_1tap_ltp(index)`, pairing the published
  β / ε arrays. The combined 12-bit gain word is split with
  `pitch_gain_index` / `max_gain_index` (eq. 36 / 39, `GSize = 24`)
  and the high-rate short-pitch variant `pitch_gain_index_short`
  (eq. 40, impulse-train MSB masked off). Constants surfaced:
  `ACELP_SUBFRAME_LEN = 60`, `ACELP_TRACK_STRIDE = 8`,
  `ACELP_CANDIDATES_PER_TRACK = 8`, `ACELP_TRACK_BASES`,
  `PITCH_1TAP_LTP_ENTRIES = 170`, `GAIN_TABLE_SIZE = 24`. Five new
  unit tests pin the accessors against Table 1, the shift offset-by-one
  invariant, the β/ε array pairing + non-negativity, the
  `GIndex = PGIndex·GSize + MGIndex` round-trip across all 4096 gain
  words (with `MGIndex` always a valid fixed-codebook-gain index), and
  the short-pitch train-bit masking.

- Typed-accessor primitives + deeper invariant tests on the staged
  G.723.1 spec-table data (round 265). The new accessors in
  `spec_tables` wrap the published raw arrays with index-typed
  lookups: `LspBand` (`Band0` / `Band1` / `Band2`) carrying the
  `(start, length)` partition info; `lsp_codebook_entry(band, idx)`
  slicing one codeword row of the correct dimension out of the
  3-band split VQ; `SpecRate::{High, Low}` driving
  `adaptive_codebook_gain_row` (returns the 20-sample row, `None`
  past the rate-specific row count) and `taming_gain` (returns the
  i16 entry, `None` past the table); `fixed_codebook_gain`
  surfacing the 24 published levels; `mpmlq_combinatorial(row, col)`
  exposing the C(n, k) table as a typed 2-D lookup with bounds
  checks; `mpmlq_pulse_count` / `mpmlq_max_position` returning the
  per-subframe published values. Constants surfaced:
  `LSP_CODEBOOK_ENTRIES_PER_BAND = 256`, `LSP_CODEBOOK_MAX_INDEX`,
  `ADAPTIVE_CODEBOOK_ROW_DIM = 20`, `ADAPTIVE_CODEBOOK_ROWS_5P3 =
  85`, `ADAPTIVE_CODEBOOK_ROWS_6P3 = 170`,
  `MPMLQ_COMBINATORIAL_ROWS = 6`, `MPMLQ_COMBINATORIAL_COLS = 30`.
  Fourteen new unit tests pin both the accessor behaviour and
  previously-unverified structural invariants of the data:
  LSP DC-predicted-frequency strict monotonicity + Q15 bounds; the
  perceptual-weighting pole table being an exact halving sequence;
  the postfilter pole table being a 3/4-geometric sequence (±1 Q15
  rounding); the postfilter zero table strictly decreasing positive;
  the fixed-codebook gain codebook log-spaced with bounded step
  ratio and >1000× span; the MP-MLQ combinatorial table satisfying
  the Pascal-rule recurrence `T[r][c] = T[r][c+1] + T[r+1][c+1]`
  across the positive-support window; taming-gain non-decreasing
  with a 1024 floor for both rates; LSP band coverage contiguous
  through `LspBand::ALL`; accessor round-trip tests for every
  helper. Lib-test count: 71 → 85.
- `cargo-fuzz` scaffold + a single `decode` target on the registered
  G.723.1 decoder's attacker surface (round 236). Drives attacker-
  supplied bytes through `Decoder::send_packet` as a sequence of up
  to 16 variable-length packets (cap 64 B each), with sizes drawn
  from the spec-legal `{0, 1, 4, 20, 24}` ladder per G.723.1 §3.7
  plus an attacker-chosen length so the per-rate
  length-validation rejection at `parse_frame_type` is reachable.
  Each packet's first body byte is fed verbatim so the 2-bit rate
  discriminator is attacker-controlled, forcing the decoder's
  cross-packet state machine — `pending` VecDeque, `next_pts`
  advance, `drained` flag, `SynthesisState`, frame-erasure run
  counter (§3.10.2), formant-postfilter / AGC memory (§3.9) —
  through the discriminator transitions a single-rate harness never
  reaches. `flush()` and `reset()` are injected mid-stream at
  deterministic hook points so the post-flush `Eof` path and the
  silence re-seed are covered. The contract under test is purely
  panic-freedom; output frames are discarded. Headline: ~200 000
  runs in 13 s on macOS aarch64, no crashes. Run with
  `cargo fuzz run decode` on a nightly toolchain.

### Changed

- **Formant-postfilter tilt + adaptive gain scaling reshaped to match
  G.723.1 §3.8 / 3.9** (round 229).
  - The §3.8 tilt-compensation stage `1 − μ · z⁻¹` no longer uses a
    constant `μ = 0.25`. Each subframe now computes the first-order
    normalised autocorrelation `k = r(1)/r(0)` of the synthesis input
    `sy[n]`, smooths it across subframes via the leaky integrator
    `k1 = (1 − POSTFILTER_TILT_SMOOTH_ALPHA) · k1_prev +
    POSTFILTER_TILT_SMOOTH_ALPHA · k` with `α = 1/4`, and applies
    `μ = POSTFILTER_TILT_BASE · k1` (`POSTFILTER_TILT_BASE = 0.25`).
    Silence leaves `μ = 0`; strong low-frequency content pulls `μ` up
    toward `≈ 0.25`. `k1` is bounded to `[−1, 1]` per Cauchy-Schwarz on
    `r(1)/r(0)`.
  - The §3.9 adaptive gain scaling is no longer a per-sample chase with
    `α = 0.85` toward `sqrt(e_in / e_out)`. The spec form is now in
    place: per subframe `g_s = sqrt(Σ sy²[n] / Σ pf²[n])` (set to `1` if
    the denominator is zero, eq. 50); per sample the smoothed gain runs
    as a leaky integrator `g[n] = (1 − α) · g[n − 1] + α · g_s` with
    `α = POSTFILTER_AGC_ALPHA = 1/16` (eq. 51); the output is
    `q[n] = pf[n] · g[n] · (1 + α)` (eq. 52) so the `(1 + 1/16)` boost
    undoes the average attenuation introduced by the integrator.
    `g[−1]` initialises to `POSTFILTER_AGC_INIT_GAIN = 1` per §3.11.
  - Round-trip PSNR on the integration test improves modestly: ACELP
    goes from ~17.4 dB to ~17.6 dB (+0.2 dB); MP-MLQ stays at ~20.7 dB
    inside its ~0.01 dB measurement-floor band. The shape is the
    headline change — tilt now tracks the per-subframe spectral tilt
    instead of cutting at a fixed factor, and the AGC follows the spec's
    leaky-integrator shape with the same `(1 + α)` compensation factor.
  - Five new unit tests pin the new behaviour:
    `post_filter_tilt_k1_smooths_per_subframe_per_spec` drives a low-pass
    synthesis input and verifies `pf_tilt_k1` moves positive on the
    first subframe and stays non-decreasing over six subsequent
    identical subframes while remaining inside `[−1, 1]`;
    `post_filter_tilt_k1_zero_input_zeroes_k` confirms zero input zeros
    `k` and the integrator decays the saved `k1` by `1 − α`;
    `post_filter_agc_holds_unity_on_silence` confirms silence in →
    silence out with the AGC staying at unity;
    `post_filter_agc_leaky_integrator_matches_closed_form` checks the
    per-sample integrator's `SUBFRAME_SIZE`-sample trajectory matches
    the closed form `g[N − 1] = g₀ + (g_s − g₀) · (1 − (1 − α)^N)`;
    `post_filter_state_starts_at_unity_agc` now also pins
    `pf_tilt_k1 = 0` and `pf_agc_gain = POSTFILTER_AGC_INIT_GAIN`.
  - New `tables` constants: `POSTFILTER_TILT_BASE = 0.25`,
    `POSTFILTER_TILT_SMOOTH_ALPHA = 0.25`,
    `POSTFILTER_AGC_ALPHA = 1/16`, `POSTFILTER_AGC_INIT_GAIN = 1.0`,
    all cited to G.723.1 §3.8 / 3.9 / 3.11. The former
    `POSTFILTER_TILT = 0.25` constant is replaced by the
    smoothed-`k1`-driven `μ` so the tilt coefficient is no longer a
    compile-time constant.

- **Frame-erasure concealment reshaped to match G.723.1 §3.10.2** (round
  222). The previous ad-hoc decay schedule (halving the saved gains and
  driving a pseudo-random innovation through the decoder pipeline at
  every erased frame) is replaced by the spec's voiced/unvoiced
  classifier path:
  - The decoder now keeps a saved trailing 120-sample window of
    post-filtered output (`ERASURE_CLASSIFIER_HISTORY_LEN`), the saved
    `L_2` (third-subframe lag), and the saved average of subframes 2
    and 3 fixed-codebook gains.
  - On erasure, a cross-correlation auto-search over `L_2 ± 3`
    (`ERASURE_CLASSIFIER_LAG_RADIUS`) computes the best-lag prediction
    gain in dB. If it exceeds `ERASURE_VOICED_THRESHOLD_DB = 0.58 dB`,
    the frame is classified voiced and concealment regenerates a
    periodic excitation at the classifier's pitch via the adaptive
    codebook with the fixed innovation suppressed; otherwise the frame
    is classified unvoiced and concealment regenerates a uniform
    pseudo-random excitation scaled by the saved average gain.
  - Attenuation follows the spec: 2.5 dB per consecutive erased frame
    (`ERASURE_ATTENUATION_DB_PER_FRAME`), mute completely after 3
    interpolated frames (`ERASURE_MUTE_AFTER_FRAMES`). Frames past the
    mute threshold emit exact silence.
  - LSP extrapolation continues to apply the wider §3.10.1 stability
    procedure (`Δ_min = 62.5 Hz`) on the saved previous-frame LSP, but
    the LSP itself is no longer perturbed by the gain schedule.
  - Two new tests pin the new behaviour:
    `decode_erased_attenuation_schedule_matches_spec` confirms the
    erased-run counter advances and emits exact silence past the mute
    threshold; `erasure_classifier_distinguishes_voiced_and_unvoiced`
    seeds the trailing window with a pure 100 Hz sinusoid (voiced ⇒
    classifier returns lag ≈ 80) and broadband-LCG noise (unvoiced ⇒
    classifier returns voiced = false) and pins the empty-history
    fallback.
  - Both pre-existing integration tests
    (`erasure_in_middle_of_stream_is_concealed`,
    `sustained_erasure_run_decays_to_silence`) continue to pass without
    modification — the spec attenuation schedule still mutes a long
    erasure run well within their 10-frame envelopes.
- **LSP stability check reshaped to match G.723.1 §3.1 / 2.6** (round 216).
  The decoded-LSP post-processing in `dequantise_lsp` is no longer an
  ad-hoc cosine-domain `gap ≥ 0.01` clamp; it now follows the spec's
  procedure (eq. 6–7.3). New `pub(crate) enforce_lsp_stability` operates
  in angular-frequency space: convert cosines → ω via `acos`, find each
  pair `(ω_j, ω_{j+1})` with `ω_{j+1} − ω_j < Δω_min`, spread it around
  its midpoint by `±Δω_min/2`, iterate up to
  `LSP_STABILITY_MAX_ITERATIONS = 10` passes, then re-convert to cosines.
  `Δω_min` is `2π · Δ_min_hz / SAMPLE_RATE_HZ`. The normal path uses
  `Δ_min = 31.25 Hz` (`LSP_STABILITY_DELTA_MIN_HZ`); the erasure
  concealment path now applies the same procedure with the spec's wider
  `Δ_min = 62.5 Hz` (`LSP_STABILITY_DELTA_MIN_ERASURE_HZ`) per §3.10.1,
  pulling the extrapolated previous-frame LSP back into a stable
  configuration when repeated erasures drift its pairs closer together.
  Five new unit tests pin the procedure: already-stable input is a
  no-op, a single inversion converges in one pass with the spreading
  applied around the midpoint, the erasure variant widens minimum gaps
  beyond the normal variant when the input violates the wider floor,
  every dequantised LSP from a probe set of indices hits the 31.25 Hz
  floor and is strictly monotone-decreasing in cosine domain, and an
  all-equal degenerate input still yields a finite LPC via `lsp_to_lpc`.
  No PSNR regression on the round-trip integration test.
- **Pitch (long-term) post-filter reshaped to match G.723.1 §3.6**
  (round 211). The decoder's pitch post-filter is no longer a fixed
  `β = 0.2` LTP at the decoded lag; it now follows the spec shape:
  forward + backward cross-correlations maximised over the seven-lag
  window `M ∈ [L − 3, L + 3]` around the reference lag `L`
  (`L = L_0` covers subframes 0,1 and `L = L_2` covers subframes 2,3
  per §3.6 prose), one-sided weighting `(w_f, w_b) ∈ {(0,0), (0,1),
  (1,0)}` driven by per-side prediction gain (eq. 45–46), a 1.25 dB
  pitch-prediction-gain gate that bypasses the LTP postfilter on
  subframes where it would harm signal quality, and the spec's
  rate-specific LTP weighting `γ_ltp` (0.1875 for the high rate, 0.25
  for the low rate) threaded through a new `pub(crate) Rate {Low,
  High}` enum from each decode entry point. Output energy
  normalisation `g_p ≤ 1` (eq. 47) means the LTP comb cannot inflate
  the subframe energy past the synthesis input. Six new structural
  unit tests pin the gate behaviour (silence + white-noise bypass,
  periodic-signal engagement, rate-dependent deviation, forward and
  backward search lock onto a sinusoid's period). Headline
  integration-test PSNR is preserved (~17.4 dB ACELP / ~20.7 dB
  MP-MLQ on the synthetic voiced signal); the postfilter is now
  signal-adaptive instead of applying a single fixed β to every
  subframe.
- New `tables` constants: `POSTFILTER_LTP_GAMMA_HIGH = 0.1875`,
  `POSTFILTER_LTP_GAMMA_LOW = 0.25`,
  `POSTFILTER_LTP_PRED_GAIN_DB_MIN = 1.25`,
  `POSTFILTER_LTP_SEARCH_RADIUS = 3`, all cited to G.723.1 §3.6.

### Added

- Three Criterion bench harnesses (`benches/encode.rs`,
  `benches/decode.rs`, `benches/roundtrip.rs`) covering both dual rates.
  Inputs are synthesised in-bench from a deterministic sum-of-sinusoids
  generator so the encoder takes the speech-like pitch path rather than
  a near-silent shortcut; no `docs/` fixtures or external files are
  read. Each harness exposes 3–4 scenarios (per-rate, voiced vs silence,
  1 s / 5 s durations, plus a mixed-rate dispatch scenario in
  `decode.rs`). Headline baseline (macOS aarch64, single-thread,
  release): ~ 22 ms/s encode at either rate, ~ 170 µs/s decode at either
  rate, ~ 20 ms/s round-trip — well above real-time at 8 kHz. Pinned to
  `criterion = "0.5"` to match the rest of the OxideAV bench crates.
  Run with `cargo bench -p oxideav-g7231 --bench {encode,decode,roundtrip}`.
- `spec_tables` module exposing the 27 ITU-T G.723.1 normative numeric
  tables (§2.2 high-pass; §2.4 LPC primitives — 180-pt Hamming, 10-pt
  binomial lag, bandwidth-expansion γ^i, 512-pt LSP cosine lookup;
  §2.6 LSP split-VQ DC predictor + 3-band codebooks Band0/1/2 in Q13;
  §2.9 perceptual-weighting filter; §2.13 MP-MLQ pulse counts /
  max-position / 6×30 combinatorial / FCB gain; §2.14 adaptive-codebook
  gain at both rates + decision factors; §2.16 1-tap LTP selector +
  gain; §2.17 taming gain at both rates; §2.18 postfilter; bit-allocation
  segment base + boundaries). Each `pub const [iN; M]` carries a
  doc-comment naming its source CSV under `docs/audio/g7231/tables/`
  and the SHA-256 of the data. Compile-time `const _` asserts pin every
  table's length; 17 unit tests pin structural invariants (Hamming
  symmetry, LSP cosine antisymmetry, FCB-gain monotonicity, LSP
  3-band partition summing to LpcOrder=10, MP-MLQ 6/5/6/5 pulse pattern,
  bit-allocation {0,32,96} / {2048,18432,231233} constants, paired
  LTP selector + gain dimensions, taming-gain floor of 1024). Data
  lives alongside (does not yet replace) the existing internally-
  consistent `tables` codebooks driving the encoder. Threading this
  spec data through the LPC / LSP / gain quantiser to produce a
  bit-exact spec-compatible bitstream is the next-round task.

## [0.0.7](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.6...v0.0.7) - 2026-05-29

### Other

- drop dead synthesis module scaffold

### Removed

- drop dead `synthesis` module — the standalone `LpcSynthesis` scaffold
  was never wired in (the real LPC synthesis filter lives inline inside
  `encoder::SynthesisState`, used by both the encoder analysis-by-
  synthesis loop and the registered decoder). Removing the unused
  `pub mod synthesis` trims the public API surface and the misleading
  "scaffold" docstring.


## [0.0.6](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.5...v0.0.6) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-g7231/pull/502))
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- drop unused SampleFormat / TimeBase imports (slim-frame leftover)
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

### Changed

- **`register` entry point unified on `RuntimeContext`** (task #502).
  The legacy `pub fn register(reg: &mut CodecRegistry)` is renamed to
  `register_codecs` and a new `pub fn register(ctx: &mut
  oxideav_core::RuntimeContext)` calls it internally. Breaking change
  for direct callers passing a `CodecRegistry`; switch to either the
  new `RuntimeContext` entry or the explicit `register_codecs` name.

## [0.0.5](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- add decoder post-filter + frame-erasure concealment
- update encoder module docstring to match current pipeline
- README reflects full-synthesis decoder + round-trip PSNR
- joint gain-pair refinement + MP-MLQ coord-descent tidy
- promote stateful decoder + coordinate-descent pulse search
- fix lsp_to_lpc p/2 buffer truncation + rework encoder-decoder sync
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"

## [0.0.4](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- fix inverted doc statement about synthesis path
