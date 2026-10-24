# Vectorial MASA-OpenMP

MASA-OpenMP (Edans Sandes, UnB) aligns huge DNA sequences with Smith-Waterman by
splitting the DP matrix into blocks and processing the anti-diagonals of blocks
in parallel with OpenMP. This version replaces the scalar block kernel
(`CPUBlockProcessor`) with the **multi-architecture striped Farrar** implementation
from [`libs/farrar/`](libs/farrar) (git submodule tracking the `main`
branch of [Multi-Architecture--Farrar](https://github.com/Guts-Santana/Multi-Architecture--Farrar)), so every block is computed with
AVX2 or RISC-V Vector (RVV) instructions.

## Project layout

```
.
├── src/                                  MASA integration (only MASA-specific code)
│   ├── main.cpp                          creates parameters, block processor and aligner
│   ├── VectorialOpenMPAligner.*          OpenMP block scheduler (from MASA-OpenMP)
│   ├── VectorialOpenMPAlignerParameters.*  --vector / --f-strategy command line options
│   └── VectorialCPUBlockProcessor.*      MASA block  ->  Farrar engine (one per thread)
├── libs/farrar/                          Farrar project, git submodule (also builds the standalone `farrar`)
│   ├── include/                          Farrar.hpp, Backend.hpp, Buffer.hpp, AVX/, RVV/
│   └── src/Farrar.cpp                    the only Farrar source compiled into vectorial-openmp
├── Sequences/                            local test pairs (not in git): 10k ... 1M
└── libs/masa-core/                       MASA core library (stages, I/O, block grid)
```

There is a single copy of Farrar. The MASA-specific part (`alignBlock` and its
helpers) is compiled only when `FARRAR_WITH_MASA` is defined. The top-level
`Makefile.am` defines it; the standalone Farrar build does not, so
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
```

The RVV build adds `-march=rv64gcv -mabi=lp64d`. The AVX build adds `-mavx2 -malign-double`.

## Running

```bash
./vectorial-openmp [MASA options] [--vector=TYPE] [--f-strategy=TYPE] seq0.fasta seq1.fasta
```

| Option | Values | Default |
|---|---|---|
| `--vector` | AVX build: `AvxInt16`, `AvxInt32`<br>RVV build: `RvvInt16M1`, `RvvInt16M2`, `RvvInt16M4`, `RvvInt16M8`, `RvvInt32M1`, `RvvInt32M2`, `RvvInt32M4`, `RvvInt32M8` | `AvxInt32` / `RvvInt32M1` |
| `--f-strategy` | `lazy-f` (alias `lazy`), `prefix-scan-f` (alias `prefix`) | `lazy-f` |

`./vectorial-openmp --help` lists the vector types available in the current build.
An invalid value stops the program with a clear message. The chosen
configuration is written to the statistics file (`Vectorial-OpenMP vector type`,
`Vectorial-OpenMP F strategy`). All the usual MASA options (`--block-size`, `--grid-size`,
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
| 13 | NW requests (`recurrenceType != SMITH_WATERMAN`) went to Farrar, which always clamps scores at 0 (SW only). | They are handed to `CPUBlockProcessor`. |
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
