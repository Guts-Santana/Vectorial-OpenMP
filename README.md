# Vectorial MASA-OpenMP

MASA-OpenMP (Edans Sandes, UnB) aligns huge DNA sequences with Smith-Waterman by
splitting the DP matrix into blocks and processing the anti-diagonals of blocks
in parallel with OpenMP. This version replaces the scalar block kernel
(`CPUBlockProcessor`) with the **multi-architecture striped Farrar** implementation
from [`libs/farrar/`](libs/farrar) (git submodule tracking the `main`
branch of [Multi-Architecture--Farrar](https://github.com/Guts-Santana/Multi-Architecture--Farrar)), so every block is computed with
AVX2, RISC-V Vector (RVV) or Arm SVE instructions.

## Project layout

```
.
├── src/                                  MASA integration (only MASA-specific code)
│   ├── main.cpp                          creates parameters, block processor and aligner
│   ├── VectorialOpenMPAligner.*          OpenMP block scheduler (from MASA-OpenMP)
│   ├── VectorialOpenMPAlignerParameters.*  --vector / --f-strategy command line options
│   └── VectorialCPUBlockProcessor.*      MASA block  ->  Farrar engine (one per thread)
├── bin/                                  build directory: Makefile.am, object files and the executable
├── libs/farrar/                          Farrar project, git submodule (also builds the standalone `farrar`)
│   ├── include/                          Farrar.hpp, Backend.hpp, Buffer.hpp, AVX/, RVV/, SVE/
│   └── src/Farrar.cpp                    the only Farrar source compiled into vectorial-openmp
├── Sequences/                            local test pairs (not in git): 10k ... 1M
└── libs/masa-core/                       MASA core library (stages, I/O, block grid)
```

There is a single copy of Farrar. The MASA-specific part (`alignBlock` and its
helpers) is compiled only when `FARRAR_WITH_MASA` is defined. `bin/Makefile.am`
defines it; the standalone Farrar build does not, so
the standalone `farrar` keeps building without MASA.

## Building

Farrar is a git submodule, so clone with `--recurse-submodules`, or run
`git submodule update --init` in an existing clone. To pull new commits from its
`main` branch: `git submodule update --remote libs/farrar`.

The vector architecture is chosen at configure time:

```bash
autoreconf -fi                      # generates configure / Makefile.in (also for libs/masa-core)

# x86-64 with AVX2 (default)
./configure --with-vec-arch=avx
make

# RISC-V with the V extension (cross compilation)
CC=riscv64-linux-gnu-gcc CXX=riscv64-linux-gnu-g++ \
    ./configure --host=riscv64-linux-gnu --with-vec-arch=rvv
make

# AArch64 with SVE (native, or cross compilation as below)
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
    ./configure --host=aarch64-linux-gnu --with-vec-arch=sve
make
```

The program is built in `bin/`, so the object files stay out of `src/` and of
the Farrar submodule. `make` also creates the link `./vectorial-openmp ->
bin/vectorial-openmp`, so it can be run from the top directory.

The RVV build adds `-march=rv64gcv -mabi=lp64d`, the SVE build `-march=armv8.2-a+sve`
and the AVX build `-mavx2 -malign-double`.

## Running

```bash
./vectorial-openmp [MASA options] [--vector=TYPE] [--f-strategy=TYPE] [--nw-f-strategy=TYPE] seq0.fasta seq1.fasta
```

| Option | Values | Default |
|---|---|---|
| `--vector` | AVX build: `AvxInt16`, `AvxInt32`, `AvxAdaptive`<br>RVV build: `RvvInt16M1`, `RvvInt16M2`, `RvvInt16M4`, `RvvInt16M8`, `RvvInt32M1`, `RvvInt32M2`, `RvvInt32M4`, `RvvInt32M8`, `RvvAdaptiveM1`, `RvvAdaptiveM2`, `RvvAdaptiveM4`, `RvvAdaptiveM8`<br>SVE build: `SveInt16`, `SveInt32`, `SveAdaptive` | `AvxInt32` / `RvvInt32M1` / `SveInt32` |
| `--f-strategy` | F strategy of the Smith-Waterman blocks (stage 1): `lazy-f` (alias `lazy`), `prefix-scan-f` (alias `prefix`) | `lazy-f` |
| `--nw-f-strategy` | F strategy of the Needleman-Wunsch blocks (stages 2 and 3): same values | `prefix-scan-f` |

`./vectorial-openmp --help` lists the vector types available in the current build.
An invalid value stops the program with a clear message. The chosen
configuration is written to the statistics file (`Vectorial-OpenMP vector type`,
`Vectorial-OpenMP F strategy`, `Vectorial-OpenMP NW F strategy`). All the usual MASA options (`--block-size`, `--grid-size`,
`-d`, `-c`, stages, ...) still work.

Example:

```bash
./vectorial-openmp -c -d work.tmp --vector=AvxInt16 --f-strategy=prefix-scan-f \
    Sequences/30k/NC_045512.2.fasta \
    Sequences/30k/OQ983940.1.fasta
```

> **16-bit lanes.** MASA carries the H values across blocks, so a block's
> boundary can hold scores as large as the global best score. With 16-bit
> lanes the limit is 32767: the 30k pair already reaches 29221. Use the
> `Int32` types (the default) for long sequences. The `Int16` types only make
> sense when you know the score stays well below 32767.

### Adaptive precision (`*Adaptive`)

The `AvxAdaptive`, `RvvAdaptiveM1`...`M8` and `SveAdaptive` types run each block
in int16 lanes when its scores, rebased, fit, and in int32 otherwise. They give
the same results as the `Int32` types at about the speed of the `Int16` types.
The statistics file reports how many blocks ran in each precision
(`ADAPTIVE PRECISION`). See [Part 4](#part-4-adaptive-precision-and-the-sve-build).

> **Large pairs and memory.** Without `--ram-size` / `--disk-size`, MASA keeps a
> special row for every block row in RAM, which grows as n² / block size
> (about 8.7 GB for the 1M pair). Pass limits, e.g. `--ram-size=1G` for stage 1
> only, or `--ram-size=512M --disk-size=8G` for a full run.

## Expected scores

Scoring: match +1, mismatch −3, gap open 3 + extension 2 (first gap costs 5).

| Pair (`Sequences/`) | Score |
|---|---|
| 10k | 5091 |
| 18k | 119 |
| 30k | 29221 |
| 50k | 52 |
| 500k | 48 |

With an AVX build, every combination of `--vector` and `--f-strategy` returns these
scores. So does the standalone `farrar` built from `libs/farrar`.

---

## Changes made

### Part 1: making Farrar's `alignBlock` produce exactly what `CPUBlockProcessor` produces

The original integration ran, but returned wrong scores (13857 instead of 33 in
the first block). After the fixes below, a test harness compared every block
against `CPUBlockProcessor` on 120 random sequence sizes and block shapes (both
F strategies, 16 and 32 bits). The block score, every `row[].h/f`, every
`col[].h/e` and `col[0]` matched exactly.

| # | Problem | Fix |
|---|---|---|
| 1 | MASA initialises the boundaries with `-INF = -999999999`. Stored in an `int16` lane it wraps to **+13825**. | `toLane()` clamps every value read from MASA to `LOW = min/4`. This keeps headroom because the adds don't saturate. |
| 2 | The first row used `row[j].h` (the cell above) as its diagonal. | The diagonal is `col[0].h` for the first column, then the old `row[j-1].h`. |
| 3 | MASA stores the E/F value *of* a boundary cell. Farrar expects the value already carried into the next cell. | On input: `E = max(H + open, E + ext)` for the left column and `F = max(H + open, F + ext)` for the top row. |
| 4 | `row[j].f = lastElement(vF)` is not the F of the block's last row. | New `pvF` buffer keeps the true F of every row; `row[j].f` reads the last row. |
| 5 | `col[].e` was written after the last column, when `pvE` already holds the *next* column's E. | It is copied out right before the last column is processed. |
| 6 | `col[0].h` was set from the new bottom row. | It is the original `row[t_len-1].h` (the next block's diagonal). |
| 7 | Padding rows (when the block height isn't a multiple of the lane count) took their diagonal from the left boundary column, outside the block. They could win the block maximum. | Padding rows get a `LOW` profile score. |
| 8 | `block_best.i` was always `abs_i + q_len`. | The real row of the maximum is located. |
| 9 | Lazy-F: when the loop wrapped from the last segment back to segment 0, `lastElement(vF)` (the F leaving the bottom row) went into lane 0, the top row. *This bug was also in the original Multi-Architecture code.* | The value carried in on wrap is `LOW`. |
| 10 | Prefix-scan-F: consecutive lanes are `segLen` rows apart, but the decay was `offset * gap_ext`. AVX `slideup` also fills the vacated lanes with 0, which acted as a fake F of 0. *Both were also in the original code.* | Decay is `offset * segLen * gap_ext`, and the vacated lanes are masked to `LOW`. |
| 11 | When F raised H, the E prepared for the next column was not updated. | Both F strategies update `pvE` (block path). |
| 12 | `VectorialOpenMPAligner` calls `processBlock` from an `omp parallel for`, but all threads shared one `Farrar` and its buffers (a data race). | One Farrar engine per OpenMP thread. |
| 13 | NW requests (`recurrenceType != SMITH_WATERMAN`) went to Farrar, which always clamps scores at 0 (SW only). | They are handed to `CPUBlockProcessor`. Since [Part 5](#part-5-exact-borders-for-pruned-blocks-and-needleman-wunsch-in-farrar), Farrar runs NW itself. |
| 14 | `alignBlock` called `setSequences(query_seg, target_seg)`, copying the whole rest of both sequences into `std::string` on every block. `exit;` did nothing. | The copy is removed; `exit(1)`. |

### Part 2: build options, runtime options and cleanup

- **Architecture at compile time.** `--with-vec-arch=avx|rvv` selects the
  backend. `libs/masa-core/configure.ac` added `-m64` unconditionally, which
  breaks RISC-V compilers. It now adds `-m64` only when the compiler accepts it.
- **Vector type and F strategy at runtime.** New `--vector` and `--f-strategy`
  options in `VectorialOpenMPAlignerParameters`. They are long options only, because
  `-v` is MASA's `--verbose`. `VectorialCPUBlockProcessor` is no longer a template:
  it creates a type-erased `FarrarBlockEngine` for the requested backend when
  `setSequences()` is first called (after the command line is parsed). The
  backend names are the same as in the standalone `farrar`.
- **Default vector type is now 32-bit** (`AvxInt32` / `RvvInt32M1`), because of
  the 16-bit limit described above. The default F strategy stays `lazy-f`.
- **One Farrar source.** The fixed Farrar moved to `Multi-Architecture/include/Farrar.hpp`
  and `Multi-Architecture/src/Farrar.cpp`, with the MASA part guarded by
  `FARRAR_WITH_MASA`. The build compiles it from there.
- **Removed from the top-level project:**
  - `src/Farrar.*`, `src/Backend.hpp`, `src/Buffer.hpp`, `src/constants.hpp`,
    `src/AVX/` and `src/RVV/`: duplicates of `Multi-Architecture/`.
  - `src/CLIParameters.*` and `src/execution.*`: the standalone CLI, not used by MASA.
  - The old `README`, `work.tmp/`, `autom4te.cache/`, `configure~`, objects and stale `.deps`.
- **masa-core fix.** `IllegalArgumentException::what()` returned a pointer into
  a destroyed local string, so every command-line error printed garbage. It now
  keeps the message in a member.

### Part 3: renames

- Classes and files: `MultiVecAligner` → `VectorialOpenMPAligner`,
  `MultiVecAlignerParameters` → `VectorialOpenMPAlignerParameters`,
  `MultiVecBlockProcessor` → `VectorialCPUBlockProcessor`. The statistics
  lines are now prefixed with `Vectorial-OpenMP`.
- Executable: `vectorial-masa` → `vectorial-openmp`.
- `Multi-Architecture/Sequences/` was removed. Its 150k, 500k and 1M pairs were
  moved into the top-level `Sequences/` (the other pairs were already there).
  The standalone README now points to `../Sequences/`.

### Part 4: adaptive precision and the SVE build

**Problem.** MASA carries absolute scores from block to block, so the `Int16`
types are wrong once a score gets near 32767. The 1M pair gives 32767 at the
wrong position instead of 88353. The adds wrap around (they don't saturate) and
`toLane()` clamps, so the wrong score is silent. The `Int32` types are correct
but have half the lanes.

**Idea.** Inside a block the scores only change by a few thousand, even when the
absolute score is 500000. So each block is run on relative scores: the lanes
hold `score - b`, and `b` is added back when the values leave the block.

**Steps, per block (R rows x C columns), decided before calling Farrar:**

1. **Lower bound `b` from the corner only.** The path corner -> diagonal into
   (1,1) -> gap along row 1 -> gap down reaches every cell of the block, so
   `H >= b = max(0, corner + mismatch + 2*gapOpen + (R+C-4)*gapExt)` everywhere
   (Farrar's convention: a gap of length k costs `gapOpen + (k-1)*gapExt`).
   The relative scores are then >= 0, so the SW floor `max(0, .)` of the lanes
   never binds where the real floor would not. The kernel is unchanged.
2. **Upper bound by scanning the boundary.** `bmax` is the largest H/E/F of the
   top row, left column and corner (R + C + 1 cells, ~0.2% of the block's
   work). A path gains at most `match` per diagonal step, so every value of the
   block is <= `bmax + match*min(R,C)`. The bests of the neighbour blocks are
   not used: a pruned block writes nothing, so `row[]`/`col[]` keep stale values
   of an older block while its grid score stays -INF, and the first row/column
   of a partition comes from another partition.
3. **Fit check.** If `(bmax - b) + match*min(R,C) <= 32767 - 128`, the block runs
   in int16 with bias `b`. Otherwise it runs in int32 with bias 0, exactly as
   before. Blocks with fewer than 2 rows or columns always use int32.
4. **Bias inside Farrar.** `Farrar::alignBlock(..., int bias = 0)` loads every
   boundary value as `toLane(v - bias)` (in `long long`, so `-INF - bias` cannot
   overflow) and writes every value back to MASA as `lane + bias` in `int`:
   `row[].h/f`, `col[].h/e` and the block best. `col[0].h` is copied from the
   raw input, so it is already absolute. With bias 0 Farrar behaves exactly as
   before.
5. **Exact special rows.** The clamp to `LOW` can only raise boundary values that
   are below `b + LOW`. With R, C >= 2, every H/E/F that leaves the block is
   computed from interior cells (relative H >= 0, so E, F >= gapOpen), which
   dominate any clamped value. So the rows and columns saved for stages 2-6
   hold the exact absolute scores.

**Code.**

- `libs/farrar` (`include/Farrar.hpp`, `src/Farrar.cpp`): the `bias` parameter.
- `src/VectorialCPUBlockProcessor.*`: `AdaptiveFarrarBlockEngine<B16, B32>`
  holds one int16 and one int32 Farrar per thread and applies steps 1-3. New
  types `AvxAdaptive`, `RvvAdaptiveM1`...`M8` (int16 and int32 with the same
  LMUL), and `SveInt16`, `SveInt32`, `SveAdaptive`.
- `src/VectorialOpenMPAligner.*`: prints the `ADAPTIVE PRECISION` statistics
  (int16 blocks, int32 blocks, largest bound seen).
- `configure.ac` / `Makefile.am`: `--with-vec-arch=sve` adds `-march=armv8.2-a+sve`
  and the SVE headers. The SVE build has not been run here (no Arm hardware).
- `.gitmodules`: `libs/farrar` tracks the `adaptive-precision` branch of Farrar,
  which has the `bias` parameter.

**Results (AVX2, 8 threads, WSL).** 10k-150k, full pipeline: `AvxInt16`,
`AvxInt32` and `AvxAdaptive` give the same stage-1 best, position and final
alignment score, and adaptive runs 100% of the blocks in int16:

| Pair | Score | Position |
|---|---|---|
| 10k | 5091 | (9418,9114) |
| 18k | 119 | (1666,1692) |
| 30k | 29221 | (29840,29756) |
| 50k | 52 | (41352,33438) |
| 150k | 18 | (41058,44353) |

1M pair, stage 1 only, `--ram-size=1G` (no disk writes). Our types: mean of 3
runs (standard deviation 0.1-1.2 s). Original MASA-OpenMP 1.0.1.1024 (scalar
`CPUBlockProcessor`, same `-O3`): 1 run.

| Block pruning | Version | Wall (s) | vs. original | Score | Position | int16 / int32 blocks |
|---|---|---|---|---|---|---|
| on | original | 293.35 | 1.0x | 88353 | (1072950,722725) | |
| on | `AvxInt16` | 24.39 | | **32767** (overflow) | (147035,870213) | |
| on | `AvxInt32` | 40.34 | 7.3x | 88353 | (1072950,722725) | |
| on | `AvxAdaptive` | 23.45 | 12.5x | 88353 | (1072950,722725) | 942002 / 59 |
| off | original | 332.99 | 1.0x | 88353 | (1072950,722725) | |
| off | `AvxInt16` | 25.86 | | **32767** (overflow) | (147035,870213) | |
| off | `AvxInt32` | 46.27 | 7.2x | 88353 | (1072950,722725) | |
| off | `AvxAdaptive` | 26.86 | 12.4x | 88353 | (1072950,722725) | 1068960 / 0 |

- Adaptive is 1.72x faster than `AvxInt32` and as fast as `AvxInt16`, with the
  correct score.
- The int32 blocks come only from pruning (stale boundaries): without pruning,
  every block runs in int16. Without pruning the largest bound on 30k is 8197;
  with pruning it reaches 26480 (30k) and 31412 (1M).
- Pruning skips only 11.9% of the blocks on the 1M pair, saving about 13% of
  the time.
- With `--disk-size=8G` the wall time is dominated by writing ~7.7 GB of special
  rows to disk (45-115 s depending on what ran before), so kernel timings use
  `--ram-size` only.

### Part 5: exact borders for pruned blocks, and Needleman-Wunsch in Farrar

**Pruned blocks.** MASA writes nothing to `row[]`/`col[]` for a pruned block,
so the blocks below and to its right read values of an older block. On the 1M
pair the 59 blocks that still needed int32 all had a real top row of 30k-86k
next to a corner and left column of ~0 left behind by a pruned neighbour (the
same block without pruning has a corner of 30719 and a bound of 6410).
`VectorialOpenMPAligner::fillPrunedBorders` now writes lower bounds for a
pruned block, never above the real values:

- corner of the next block: the pruned block's top-right input, as a computed
  block passes on;
- bottom row and right column: a gap from the input above / on the left;
- every H: also the corner floor (`cornerLowerBound`), and 0 in SW only.

The pruning only reads the block scores, so the same blocks are pruned.

| Pair (stage 1, pruning on) | int32 blocks before | after | largest bound before | after |
|---|---|---|---|---|
| 1M | 59 | 0 | 31412 | 7138 |
| 5M | 2303 | 0 | 32045 | 7173 |

**Needleman-Wunsch.** Stages 2 and 3 run NW blocks, which used to go to
MASA's scalar `CPUBlockProcessor`. Farrar now runs both recurrences with one
kernel. The only difference inside it is the floor of `H`: `0` for SW, `LOW`
for NW (it only stops the adds from wrapping). In the adaptive engine NW uses
`b = cornerLowerBound(...)` without the `max(0, .)`: every H of the block is
still `>= b`, so the lanes hold `H - b >= 0` and the same int16 test applies.

- `libs/farrar`: `AlignmentType` (`Local` / `Global`), `cornerLowerBound`,
  `alignBlock(..., bias, type)`, `setFStrategy`. The standalone `farrar` has a
  new `-a, --alignment=sw|nw` option (see its README).
- `src/VectorialCPUBlockProcessor.*`: SW and NW blocks both go to Farrar.
- `--nw-f-strategy` (default `prefix-scan-f`). In NW the cells below the
  alignment are reached through long vertical gaps, so lazy-F sweeps most of
  every column and is slower than the scalar code; in SW those cells hit the 0
  floor. Stage 2 on the 1M pair: scalar 2.40 s, lazy-f 3.41 s, prefix-scan-f
  0.87 s (mean of 3 runs). Stage 1 keeps `lazy-f`.

**Results.** Full pipeline (stages 1-6), `--ram-size=1G`, 1 run per line,
against the previous build (scalar NW): `alignment.00.bin` and the summary are
identical for 10k-1M, pruning on and off, `AvxAdaptive` and `AvxInt32` (28 of
28 pairs). Stages 2 and 3 in ms:

| Pair | Pruning | Vector | Stage 2 before | after | Stage 3 before | after |
|---|---|---|---|---|---|---|
| 500k | on | `AvxAdaptive` | 349 | 157 | 0.7 | 0.7 |
| 500k | on | `AvxInt32` | 374 | 231 | 0.7 | 0.7 |
| 1M | on | `AvxAdaptive` | 2328 | 906 | 286 | 173 |
| 1M | on | `AvxInt32` | 2395 | 1298 | 347 | 198 |
| 1M | off | `AvxAdaptive` | 2333 | 768 | 351 | 142 |
| 1M | off | `AvxInt32` | 2327 | 984 | 437 | 168 |

Stages 2 and 3 run in int16 for every block but one (10k, stage 3). On the
smaller pairs stage 2 takes a few tens of ms and single runs vary by up to 2x.
Stage-2 blocks are only 128 x 136 cells, which is the size MASA chooses for
these narrow partitions.
