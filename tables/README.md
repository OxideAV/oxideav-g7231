# tables/ — ITU-T G.723.1 bit-exact numeric tables

Vendored snapshot of 16 small numeric tables extracted from the ITU-T
G.723.1 reference fixed-point implementation, used by `src/itu_tables.rs`
via `include_str!`.

The source-of-truth lives in the workspace clean-room area at
`docs/audio/g7231/tables/`; these copies are byte-identical and carry
the same provenance. Any update to the workspace CSVs must be mirrored
here in the same commit.

## Legal basis

Numeric tables in source are facts, not copyrightable expression
(*Feist Publications, Inc., v. Rural Telephone Service Co.*, 499 U.S.
340, US Sup. Ct. 1991). ITU copyright on the algorithmic G.723.1
source applies to its implementation choices — not to the data values
that the source happens to declare. The workspace extractor reads
only the data-only table listing of the ITU reference fixed-point
distribution (plus the dimension-constants header it depends on),
never any algorithmic source.

## Files

| File | C identifier | Shape | Element type | Spec |
|------|--------------|------:|--------------|------|
| `lpc-hamming-window-Q15.csv` | `HammingWindowTable` | 180 | Word16 (Q15) | §2.4 |
| `lpc-binomial-lag-window-Q15.csv` | `BinomialWindowTable` | 10 | Word16 (Q15) | §2.4 |
| `lpc-bandwidth-expansion-Q15.csv` | `BandExpTable` | 10 | Word16 (Q15) | §2.4 |
| `lpc-lsp-cosine-lookup-Q15.csv` | `CosineTable` | 512 | Word16 (Q15) | §2.4 |
| `lsp-band-info.csv` | `BandInfoTable` | 3 × 2 = 6 | Word16 (start, length per band) | §2.6 |
| `lsp-dc-predicted-frequencies-Q15.csv` | `LspDcTable` | 10 | Word16 (Q15) | §2.6 |
| `perceptual-weighting-zero-coefficients-Q15.csv` | `PerFiltZeroTable` | 10 | Word16 (Q15) | §2.9 |
| `perceptual-weighting-pole-coefficients-Q15.csv` | `PerFiltPoleTable` | 10 | Word16 (Q15) | §2.9 |
| `postfilter-zero-coefficients-Q15.csv` | `PostFiltZeroTable` | 10 | Word16 (Q15) | §2.9 |
| `postfilter-pole-coefficients-Q15.csv` | `PostFiltPoleTable` | 10 | Word16 (Q15) | §2.9 |
| `highpass-filter-constants-Q15.csv` | `LpfConstTable` | 2 | Word16 (Q15) | §2.2 |
| `gain-quantizer-decision-factors.csv` | `fact` | 4 | Word16 | §2.14 |
| `bit-allocation-segment-boundaries.csv` | `L_bseg` | 3 | Word32 | bit-packing |
| `bit-allocation-segment-base.csv` | `base` | 3 | Word16 | bit-packing |
| `mp-mlq-pulse-count-per-subframe.csv` | `Nb_puls` | 4 | Word16 | §2.13 |
| `mp-mlq-max-position-table.csv` | `MaxPosTable` | 4 | Word32 | §2.13 |

The cleanroom workspace's matching `.meta` sidecars (under
`docs/audio/g7231/tables/`) record the full extraction provenance:
the upstream data-listing SHA-256
(`6a7be683afe47a2d8c3f20113be198f5911fd8c02f0effc8aca9b38ebc2dbae2`),
the line range of each table in the source, the C type
(`Word16` / `Word32`), and the extractor invocation. This crate
vendors only the data; the metadata is consulted in the workspace.

## CSV format

Each `.csv` file is one numeric value per line, no header, no trailing
commas. Values are either bare decimal integers (e.g. `2621`) or
`0x`-prefixed hexadecimal (e.g. `0x1800` for Word16, `0x00090f6f` for
Word32). The single exception is `lsp-band-info.csv`, where each line
holds the two columns `(start, length)` separated by a comma. The
parsers in `src/itu_tables.rs` accept either decimal or hex on a
per-line basis and reinterpret unsigned hex literals as signed
two's-complement for the `Word16` (i16) tables.

## Tables NOT vendored here

The three large LSP split-VQ codebooks (`Band0Tb8` / `Band1Tb8` /
`Band2Tb8`, 768 + 768 + 1024 = 2 560 Word16 entries), the two
rate-specific adaptive-codebook gain tables (`AcbkGainTable085` /
`AcbkGainTable170`, 1 700 + 3 400 entries), the joint gain codebook
(`FcbkGainTable`, 24 entries), the MP-MLQ combinatorial table
(`CombinatorialTable`, 6 × 30 = 180 entries), the rate-5.3-kbit/s
1-tap LTP pitch shortcut tables, and the per-rate taming-gain tables
remain in the workspace area only and are not yet wired into this
crate's `src/itu_tables.rs`. They land in follow-up rounds once the
consuming code paths are ready for them.
