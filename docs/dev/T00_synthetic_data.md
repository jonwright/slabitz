# T00: Synthetic test data, matched to real data, plus worst cases

Status: done (9088d09)
Executor: Claude defines the generator models and the matching metrics;
worker can draft the encoders and the writer once the models are settled.
Depends on: T-1 (writes into `$SLABITZ_DEV_ROOT/data/synth/`)
Branch: `t00-synthetic-data`

## Goal

Build a deterministic generator for detector frames that:

1. **Matches the real data** in the properties that decide compression
   ratio and decoder speed. The synthetic twin of each real file must be
   verified against it with stated metrics.
2. **Covers what the real data does not:** other detector sizes and
   dtypes, masks written as 0 or as 2^n-1, signed Pilatus data,
   high-dynamic-range u32, and block sizes from 8 kB to 256 kB.
3. **Provides named worst cases** for the decoders (and later the encoders):
   cases that maximise work, divergence, table sizes, or output-to-input
   ratio, or that sit exactly on format limits.

The data is written as filter-32008 HDF5 in every encoding we decode, plus
the raw frames, so tests and benchmarks (T02, T03, T06, T11, T17-T20) can
use it without depending on the real files.

## Real data to match (inventory 2026-11-25; T-1 links it under `data/real/`)

| Key | File | Frames x shape | dtype | Encoding | Character |
|---|---|---|---|---|---|
| silicon | `silicon_u16_zstd2_256k.h5`, `silicon_lz4.h5`, `silicon_lz4mask.h5` | 250 x 2162 x 2068 | u16 | zstd level 2 256 kB; LZ4 8 kB; LZ4 with mask = 65535 | sparse, about 0.4% non-zero, Bragg spots, 20 ms |
| ceo2 | `ceo2_u32_zstd2_256k.h5`, `ceo2_lz4*.h5` | 72 x 2162 x 2068 | u32 | as above | dense powder rings, mean about 68 counts |
| thaumatin | `thaumatin_9_1_00000{1,2,3}.h5` (DLS i04, Eiger2 16M) | 1000 x 4362 x 4148 per file | u16 | LZ4, default block | MX: diffuse background plus spots |
| eiger4m_u32 | `eiger4m_u32.h5` | 3620 x 2162 x 2068 | u32 | LZ4, default block | ID11 scan |
| big_eiger | `big_eiger_0000.h5` | 18000 x 2162 x 2068 | u16 | LZ4 (filter option 5, check) | long scan |
| kevlar | `kevlar.h5` (Eiger1 4M) | 1000 x 2167 x 2070 | u16 | LZ4 | fibre pattern |
| hplc | `hplc.h5` (Pilatus) | 100 x 1679 x 1475 | **int32** | LZ4 | signed, masked/gap pixels negative |
| al2o3 | `Calibration_Al2O3.h5` | 10 x 2167 x 2070 | u32 | LZ4 | calibrant rings |

`data/real/INVENTORY.md` (from T-1) is authoritative for the filter
options; check it before relying on the table above.

## Properties that matter (the model must reproduce them)

These are the knobs of the generator. They are also the metrics compared
with real data.

- **P1, value distribution per pixel class.**
  - For each class (background, signal, masked/gap, hot, saturated):
    the fraction of pixels, a log-binned histogram of values, the mean and
    the variance.
  - Background is Poisson, or a Poisson-gamma mixture where it is
    overdispersed, with a mean that varies smoothly over the detector.
  - Signal is spots (2D Gaussians with a size distribution and an
    intensity distribution with a power-law tail) or rings (radial profile
    with azimuthal texture), or a fibre pattern.
- **P2, spatial structure.**
  - Module and chip gaps on the real detector geometry (Eiger 4M: 2162 x
    2068 with gaps; Eiger2 16M: 4362 x 4148; Pilatus 1679 x 1475; Eiger1
    4M: 2167 x 2070).
  - The beamstop shadow.
  - Gap and masked pixels written as 0, as 2^n-1 (Eiger), or as -1/-2
    (Pilatus int32).
- **P3, bit-plane occupancy** after bitshuffle. For each bit plane and
  block size (8 kB, 256 kB): the fraction of all-zero plane-blocks, the
  fraction of plane-blocks equal to the plane above (`off = P` matches),
  and the byte entropy of non-trivial plane-blocks.
- **P4, codec-level statistics** (directly what the decoder executes):
  - the compression ratio with hdf5plugin LZ4 at 8 kB, hdf5plugin zstd at
    levels 2 and 3 at 256 kB, and pac zstd and pac LZ4 (`PAC_LIB`);
  - the zstd block-type mix (raw, RLE, compressed);
  - the literals-section types;
  - the sequence count per block;
  - the fractions of match bytes with `off == 1`, `off == P`, and other
    offsets, and of literal bytes.

  For P4, use the edu oracle tags from T06 when available, otherwise
  `tools/edu_oracle.py` SEQ/LITS dumps.
- **P5, frame-to-frame correlation.** Frames of a scan share background
  and some spots. This matters only for streams (benchmarks of many
  frames), not for per-frame compression.

## Settled design

- **Generator: `tools/synth/`**, a Python package using numpy only. Its
  parts:
  - `models.py` holds the models: sparse spots, powder rings, MX
    background plus spots, fibre, flat Poisson.
  - `geometry.py` holds the detector geometries, gaps and masks.
  - `fit.py` estimates the model parameters from a real file by reading N
    frames and measuring P1-P3.
  - `encode.py` writes HDF5 through h5py and hdf5plugin (bitshuffle LZ4
    and zstd), and through libpac.
  - `metrics.py` computes P1-P4 for any file.
  - `cli.py` provides `python3 -m tools.synth {fit,generate,compare,worst}`.
- **Determinism.** Every dataset is a function of (model, parameters,
  seed, generator version). Generation is reproducible bit for bit, and
  a `manifest.json` records parameters, seed, version, sha256 of raw and
  encoded files, and the P1-P4 metrics.
- **Twins.** For every real key: `fit` gives a parameter file, committed
  in `tools/synth/params/<key>.json`, small and reviewable. Then
  `generate` produces a twin with the same shape, dtype, frame count
  (capped at 64 frames) and encodings. Then `compare` prints P1-P4 for
  real against twin.
- **Extensions** (parameterised, generated from the twins):
  - each model at u8, u16, u32, u64, and int32 with Pilatus-style
    negatives;
  - masks as 0 and as 2^n-1;
  - block sizes of 8, 56 (pac LZ4 default), 248 (pac zstd default) and
    256 kB;
  - detector sizes 1M, 4M and 16M.
- **Worst cases** (`worst` subcommand). Each case is named, small enough
  to run in the test suite unless marked large, and has a one-line
  statement of what it stresses:
  - **Incompressible:** `random_u8`/`u16`/`u32`/`u64` (all raw blocks;
    output equals input).
  - **Trivial:** `zeros`, `all_max` (every pixel 2^n-1, the masked-detector
    extreme), and `single_hot_pixel` (one non-zero per frame).
  - **Plane structure:** `alternating_planes` (bit planes alternately
    zero and dense: many `off = P` and `off = 1` switches).
  - **Most sequences:** `min_matches` (data built so the LZ4 and zstd
    parsers find a minimum-length match, 4 bytes, every few bytes, giving
    the most sequences per block for ExecuteSequences and the LZ4 decoder).
  - **Short offsets:** `short_offsets_1_31` (overlapping matches with
    offsets 1-31, the DS swizzle loop; constructed at the byte level after
    the transpose, for the plain-zstd API).
  - **Long reach:** `far_offsets` (zstd offsets near the block or window
    start; LZ4 offsets near 65535).
  - **Literals:** `huffman_max_depth` (a literal distribution that forces
    an 11-bit Huffman table) and `literals_128k` (a block that is all
    literals: the 128 KiB literal bound).
  - **FSE tables:** `fse_max_accuracy` (sequence symbol statistics that
    force LL 9, OF 8 and ML 9 accuracy).
  - **Many small frames:** `tiny_frames` (frames of 1-7 elements, tails
    only) and `many_frames` (thousands of 8-element frames in one call).
  - **Format edges:** `block_tail_mix` (frame sizes not a multiple of the
    block, every element size), `exact_128k_block` (a compressed block of
    exactly 131072 bytes, built with libzstd's block-splitting
    parameters; marked "decoder must accept, encoder must avoid") and
    `zero_seq_blocks`.
  - **Scale:** `large_frame_16m_u32` (one Eiger2 16M u32 frame of high
    dynamic range; large, not in the default test set).

  Worst cases for plain zstd frames are written as `.zst` with a
  manifest. Those for bitshuffle data are written as HDF5.
- **Output.**
  - `$SLABITZ_DEV_ROOT/data/synth/v1/` holds `twins/<key>/`,
    `ext/<model>_<dtype>_<block>/` and `worst/<case>/`, each with
    `raw.npy` (or `.npy` per frame batch), the encoded `.h5` and `.zst`
    files, and `manifest.json`.
  - `data/synth/current -> v1`.
  - Budget: at most 30 GB for v1. Twins use at most 64 frames.
  - The directory is made read-only (`chmod -R a-w`) after generation.
    A new generator version writes v2.
- **Small committed subset: `tests/data/synth/`**, at most 5 MB. It holds
  one cropped frame per model and every small worst case, encoded in LZ4
  8 kB and zstd 256 kB, regenerated by `python3 -m tools.synth subset`.
  T02, T06 and T11 use it inside `sbz test`.

## Files that may change

`tools/synth/*` (new), `tools/synth/params/*.json`,
`tests/data/synth/*` (at most 5 MB), `tests/run_synth_decode.py` (new: it
decodes the committed subset and, when `SLABITZ_SYNTH` is set, the full
v1, byte-exact against raw), `pixi.toml` (test list), `AGENTS.md` (Status
row), `docs/dev/reports/T00.md`.

## Acceptance

1. **Twins match their real files.** `python3 -m tools.synth compare --all`
   prints, for every real key, real against twin, and passes these
   tolerances:
   - P1: the masked, gap and non-zero pixel fractions are within 10%
     relative; the mean is within 5%; the log-histogram Jensen-Shannon
     distance is at most 0.05.
   - P3: the per-plane all-zero fraction is within 0.05 absolute at 8 kB
     and at 256 kB.
   - P4: the compression ratio is within 10% for each encoding listed in
     P4. The `off==1`, `off==P` and literal byte fractions are within
     0.05 absolute.

   Every key must pass. A key that cannot meet a tolerance is reported
   with the reason, and the user decides whether to accept it.
2. **Twins time like their real files.** On an L40S via `gpu-run`, decode
   time per frame of each twin against its real file (silicon, ceo2,
   thaumatin; LZ4 and zstd; 16 frames per call) is within 15%. This shows
   the twins can stand in for real data in benchmarks (T03).
3. **Determinism.** Generating twice with the same seed gives identical
   sha256 for every raw and encoded file. The run prints the count
   compared.
4. **Worst cases are what they claim.** For each worst case, `metrics.py`
   shows the property it claims. For example:
   - `min_matches` has a sequence count per block at least 0.9x the
     theoretical maximum for its parser;
   - `huffman_max_depth` has an 11-bit table in the oracle dump;
   - `exact_128k_block` contains a block of exactly 131072 bytes;
   - `random_*` is all raw blocks.

   The report prints a table: case, claimed property, measured value,
   pass.
5. **Decoding.** `python3 tests/run_synth_decode.py` passes on the
   committed subset (inside `sbz test`) and, with `SLABITZ_SYNTH`, on all
   of v1. Expected rejects (for example no-FCS or large-window worst
   cases, if added) are marked as in T06. Compared bytes are nonzero.
6. **Size.** `du -sh $SLABITZ_DEV_ROOT/data/synth/v1` is at most 30 GB,
   `du -sh tests/data/synth` is at most 5 MB, and v1 is read-only.

## Reviewer checks

- Perturb the silicon parameters (double the spot density) and run
  `compare`: it must fail P1 and P4 for silicon.
- Regenerate one worst case with a different seed: its sha256 must change
  while its claimed property still holds.
- Open one twin frame next to a real frame as images, for a visual check
  that it looks like the same kind of data.
