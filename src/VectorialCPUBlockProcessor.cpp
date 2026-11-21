#include "VectorialCPUBlockProcessor.hpp"
#include "VectorialOpenMPAlignerParameters.hpp"
#include "Backend.hpp"
#include "Farrar.hpp"

#include <omp.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>

template <typename Backend>
class FarrarBlockEngineImpl : public FarrarBlockEngine {
private:
    Farrar<Backend> farrar;
    const std::string swStrategy;
    const std::string nwStrategy;

public:
    FarrarBlockEngineImpl(const std::string& swStrategy, const std::string& nwStrategy,
                          const ScoringScheme& scoring)
        : farrar(swStrategy, scoring), swStrategy(swStrategy), nwStrategy(nwStrategy) {}

    virtual score_t alignBlock(const char* query_seg, int q_len,
                               const char* target_seg, int t_len,
                               cell_t* row, cell_t* col,
                               int abs_i, int abs_j,
                               AlignmentType type) override {
        // No rebase: the lanes hold the absolute scores (SW and NW alike)
        farrar.setFStrategy(type == AlignmentType::Global ? nwStrategy : swStrategy);
        return farrar.alignBlock(query_seg, q_len, target_seg, t_len, row, col,
                                 abs_i, abs_j, 0, type);
    }
};

/*
 * MASA scores (CPUBlockProcessor.hpp) in Farrar's convention. MASA's penalties
 * are positive and a gap of length k costs DNA_GAP_OPEN + k * DNA_GAP_EXT, while
 * Farrar's are negative and its gapOpen is the cost of the whole first gap cell.
 */
ScoringScheme masaScoring() {
    ScoringScheme scoring;
    scoring.match    = DNA_MATCH;
    scoring.mismatch = DNA_MISMATCH;
    scoring.gapOpen  = -DNA_GAP_FIRST;
    scoring.gapExt   = -DNA_GAP_EXT;
    return scoring;
}

template <typename Backend>
static std::unique_ptr<FarrarBlockEngine> makeEngine(const std::string& strategy,
                                                     const std::string& nwStrategy) {
    return std::unique_ptr<FarrarBlockEngine>(
            new FarrarBlockEngineImpl<Backend>(strategy, nwStrategy, masaScoring()));
}

/*
 * Adaptive precision: before each block, decide whether it can run in int16
 * lanes (after rebasing the scores by a bias b) or must run in int32.
 *
 * Lower bound b. Only the corner H(0,0) = col[0].h is trusted: it is the only
 * boundary cell from which a path reaches every cell (i,j) of the R x C block.
 * H(i,j) is the max over all paths, so any path, scored as if every diagonal
 * step were a mismatch, gives a floor. Two paths are used:
 *   gaps: corner -> one diagonal step -> gap along row 1 -> gap down the column
 *         H >= corner + mismatch + 2*gapOpen + (R+C-4)*gapExt   (worst (R,C))
 *   diag: corner -> min(i,j) diagonal steps -> one gap of |i-j| cells
 *         H >= corner + mismatch*min(R,C) + gapOpen + max(|R-C|-1,0)*gapExt
 * (Farrar's convention: a gap of length k costs gapOpen + (k-1)*gapExt).
 * The worst cell of each path is taken (one b for the whole block) and the
 * higher of the two floors is kept. With MASA's scores the diagonal one wins
 * (3 per diagonal step < 2*2 for a row plus a column of gap): c - 3077
 * instead of c - 4101 for 1024 x 1024 blocks.
 * The floor is cornerLowerBound (Farrar.hpp), valid for SW and NW alike.
 *   SW: b = max(0, floor). Every H of the block is >= b, so the lanes hold
 *       H - b >= 0 and the SW floor max(0, .) of the lanes never binds where
 *       the real floor would not: the kernel stays exact unchanged.
 *   NW: b = floor, with no 0 (NW scores go below 0). The lanes still hold
 *       H - b >= 0, and NW's lane floor (LOW) never binds either.
 *
 * Upper bound. The boundary is scanned (R + C + 1 cells, ~0.2% of the block
 * work for 1024 x 1024). The scan is used instead of the bests of the
 * neighbour blocks because a pruned block has no score (its grid score stays
 * -INF; VectorialOpenMPAligner::fillPrunedBorders writes only lower bounds on
 * its borders), and the first row/column of a partition comes from another
 * partition.
 * A path entering the block from a boundary cell gains at most match per
 * diagonal step, so every value of the block is <= bmax + match*min(R,C).
 *
 * Exactness of the values written back to MASA (special rows/columns): the
 * clamp to LOW can only raise boundary values that are < b + LOW. For R, C >= 2
 * the H, E and F that leave the block are computed from interior cells
 * (relative H >= 0, so E, F >= gapOpen), which dominate any clamped value, so
 * they are exact. Blocks with R < 2 or C < 2 always use int32, with no rebase
 * (NW included: int32 lanes hold any score above LOW = INT32_MIN / 4).
 *
 * The bias is added back in int by Farrar::alignBlock, never in the lanes.
 */
template <typename Backend16, typename Backend32>
class AdaptiveFarrarBlockEngine : public FarrarBlockEngine {
private:
    // Headroom left below INT16_MAX for the kernel's intermediate values
    static constexpr long long MARGIN = 128;
    static constexpr long long LIMIT = INT16_MAX - MARGIN;

    Farrar<Backend16> farrar16;
    Farrar<Backend32> farrar32;
    const ScoringScheme scoring;
    const std::string swStrategy;
    const std::string nwStrategy;

    long long int16Blocks = 0;
    long long int32Blocks = 0;
    long long peakInt16Bound = 0;

public:
    AdaptiveFarrarBlockEngine(const std::string& swStrategy, const std::string& nwStrategy,
                              const ScoringScheme& scoring)
        : farrar16(swStrategy, scoring), farrar32(swStrategy, scoring), scoring(scoring),
          swStrategy(swStrategy), nwStrategy(nwStrategy) {}

    virtual score_t alignBlock(const char* query_seg, int q_len,
                               const char* target_seg, int t_len,
                               cell_t* row, cell_t* col,
                               int abs_i, int abs_j,
                               AlignmentType type) override {
        const long long R = q_len;
        const long long C = t_len;

        const std::string& strategy = (type == AlignmentType::Global) ? nwStrategy : swStrategy;
        farrar16.setFStrategy(strategy);
        farrar32.setFStrategy(strategy);

        if (R >= 2 && C >= 2) {
            const long long corner = col[0].h;

            // Largest value entering the block (H, E and F of the boundary)
            long long bmax = corner;
            for (int i = 1; i <= q_len; ++i) {
                bmax = std::max<long long>(bmax, std::max(col[i].h, col[i].e));
            }
            for (int j = 0; j < t_len; ++j) {
                bmax = std::max<long long>(bmax, std::max(row[j].h, row[j].f));
            }

            // Both corner paths (see cornerLowerBound), the higher floor kept;
            // only SW has the extra floor at 0
            long long b = cornerLowerBound(corner, R, C, scoring);
            if (type == AlignmentType::Local)
                b = std::max(b, 0LL);
            const long long bound = std::max(bmax - b, 0LL) + scoring.match * std::min(R, C);

            if (bound <= LIMIT) {
                ++int16Blocks;
                peakInt16Bound = std::max(peakInt16Bound, bound);
                return farrar16.alignBlock(query_seg, q_len, target_seg, t_len,
                                           row, col, abs_i, abs_j, static_cast<int>(b), type);
            }
        }

        ++int32Blocks;
        return farrar32.alignBlock(query_seg, q_len, target_seg, t_len,
                                   row, col, abs_i, abs_j, 0, type);
    }

    virtual bool isAdaptive() const override { return true; }
    virtual long long getInt16Blocks() const override { return int16Blocks; }
    virtual long long getInt32Blocks() const override { return int32Blocks; }
    virtual long long getPeakInt16Bound() const override { return peakInt16Bound; }
};

template <typename Backend16, typename Backend32>
static std::unique_ptr<FarrarBlockEngine> makeAdaptiveEngine(const std::string& strategy,
                                                             const std::string& nwStrategy) {
    return std::unique_ptr<FarrarBlockEngine>(
            new AdaptiveFarrarBlockEngine<Backend16, Backend32>(strategy, nwStrategy, masaScoring()));
}

const std::vector<std::string>& VectorialCPUBlockProcessor::supportedVectors() {
    static const std::vector<std::string> vectors = {
#if defined(USE_AVX)
        "AvxInt16", "AvxInt32", "AvxAdaptive"
#elif defined(USE_RVV)
        "RvvInt16M1", "RvvInt16M2", "RvvInt16M4", "RvvInt16M8",
        "RvvInt32M1", "RvvInt32M2", "RvvInt32M4", "RvvInt32M8",
        "RvvAdaptiveM1", "RvvAdaptiveM2", "RvvAdaptiveM4", "RvvAdaptiveM8"
#elif defined(USE_SVE)
        "SveInt16", "SveInt32", "SveAdaptive"
#endif
    };
    return vectors;
}

bool VectorialCPUBlockProcessor::isSupportedVector(const std::string& vector) {
    for (const std::string& v : supportedVectors()) {
        if (v == vector) return true;
    }
    return false;
}

std::string VectorialCPUBlockProcessor::defaultVector() {
    // 32-bit lanes: MASA carries the scores across blocks, so for long
    // sequences they quickly overflow int16 (max 32767)
#if defined(USE_AVX)
    return "AvxInt32";
#elif defined(USE_SVE)
    return "SveInt32";
#else
    return "RvvInt32M1";
#endif
}

bool VectorialCPUBlockProcessor::isSupportedFStrategy(const std::string& strategy) {
    return strategy == "lazy-f" || strategy == "lazy"
        || strategy == "prefix-scan-f" || strategy == "prefix";
}

std::string VectorialCPUBlockProcessor::normalizeFStrategy(const std::string& strategy) {
    return (strategy == "prefix" || strategy == "prefix-scan-f") ? "prefix-scan-f" : "lazy-f";
}

std::string VectorialCPUBlockProcessor::defaultFStrategy() {
    return "lazy-f";
}

std::string VectorialCPUBlockProcessor::defaultNwFStrategy() {
    // In NW the cells below the alignment are reached through long vertical
    // gaps, so lazy-f sweeps most of every column: prefix-scan-f is ~2.8x faster
    // in stage 2 on the 1M pair (lazy-f is the faster one for SW, in stage 1)
    return "prefix-scan-f";
}

std::unique_ptr<FarrarBlockEngine> VectorialCPUBlockProcessor::createEngine(
        const std::string& vector, const std::string& strategy, const std::string& nwStrategy) {
#if defined(USE_AVX)
    if (vector == "AvxInt16") return makeEngine<AvxInt16>(strategy, nwStrategy);
    if (vector == "AvxInt32") return makeEngine<AvxInt32>(strategy, nwStrategy);
    if (vector == "AvxAdaptive") return makeAdaptiveEngine<AvxInt16, AvxInt32>(strategy, nwStrategy);
#elif defined(USE_RVV)
    if (vector == "RvvInt16M1") return makeEngine<RvvInt16M1>(strategy, nwStrategy);
    if (vector == "RvvInt16M2") return makeEngine<RvvInt16M2>(strategy, nwStrategy);
    if (vector == "RvvInt16M4") return makeEngine<RvvInt16M4>(strategy, nwStrategy);
    if (vector == "RvvInt16M8") return makeEngine<RvvInt16M8>(strategy, nwStrategy);
    if (vector == "RvvInt32M1") return makeEngine<RvvInt32M1>(strategy, nwStrategy);
    if (vector == "RvvInt32M2") return makeEngine<RvvInt32M2>(strategy, nwStrategy);
    if (vector == "RvvInt32M4") return makeEngine<RvvInt32M4>(strategy, nwStrategy);
    if (vector == "RvvInt32M8") return makeEngine<RvvInt32M8>(strategy, nwStrategy);
    // Same LMUL for both: int16 then has twice the lanes of int32
    if (vector == "RvvAdaptiveM1") return makeAdaptiveEngine<RvvInt16M1, RvvInt32M1>(strategy, nwStrategy);
    if (vector == "RvvAdaptiveM2") return makeAdaptiveEngine<RvvInt16M2, RvvInt32M2>(strategy, nwStrategy);
    if (vector == "RvvAdaptiveM4") return makeAdaptiveEngine<RvvInt16M4, RvvInt32M4>(strategy, nwStrategy);
    if (vector == "RvvAdaptiveM8") return makeAdaptiveEngine<RvvInt16M8, RvvInt32M8>(strategy, nwStrategy);
#elif defined(USE_SVE)
    if (vector == "SveInt16") return makeEngine<SveInt16>(strategy, nwStrategy);
    if (vector == "SveInt32") return makeEngine<SveInt32>(strategy, nwStrategy);
    if (vector == "SveAdaptive") return makeAdaptiveEngine<SveInt16, SveInt32>(strategy, nwStrategy);
#endif
    return nullptr;
}

VectorialCPUBlockProcessor::VectorialCPUBlockProcessor(const VectorialOpenMPAlignerParameters* params)
    : params(params), full_seq0(nullptr), full_seq1(nullptr)
{
}

void VectorialCPUBlockProcessor::setSequences(const char* seq0, const char* seq1, int seq0_len, int seq1_len) {
    this->full_seq0 = seq0;
    this->full_seq1 = seq1;
    cpu_processor.setSequences(seq0, seq1, seq0_len, seq1_len);

    if (farrar_engines.empty()) {
        int n_threads = omp_get_max_threads();
        for (int t = 0; t < n_threads; ++t) {
            std::unique_ptr<FarrarBlockEngine> engine =
                createEngine(params->getVector(), params->getFStrategy(),
                             params->getNwFStrategy());
            if (!engine) {
                std::cerr << "ERROR: vector type '" << params->getVector()
                          << "' is not available in this build.\n";
                exit(1);
            }
            farrar_engines.push_back(std::move(engine));
        }
    }
}

void VectorialCPUBlockProcessor::unsetSequences() {
    this->full_seq0 = nullptr;
    this->full_seq1 = nullptr;
    cpu_processor.unsetSequences();
}

score_t VectorialCPUBlockProcessor::processBlock(cell_t *row, cell_t *col,
                                             const int i0, const int j0,
                                             const int i1, const int j1,
                                             const int recurrenceType)
{
    if (!full_seq0 || !full_seq1) {
        std::cerr << "ERROR: MASA called processBlock before setSequences!\n";
        exit(1);
    }

    // Farrar runs both recurrences; anything else goes to MASA's scalar code
    if (recurrenceType != SMITH_WATERMAN && recurrenceType != NEEDLEMAN_WUNSCH) {
        return cpu_processor.processBlock(row, col, i0, j0, i1, j1, recurrenceType);
    }

    // i runs over seq0 (block rows), j over seq1 (block columns)
    int block_height = i1 - i0;
    int block_width = j1 - j0;

    const char* query_seg = full_seq0 + i0;
    const char* target_seg = full_seq1 + j0;

    return farrar_engines[omp_get_thread_num()]->alignBlock(
            query_seg, block_height,
            target_seg, block_width,
            row, col,
            i0, j0,
            toAlignmentType(recurrenceType));
}

void VectorialCPUBlockProcessor::printAdaptiveStatistics(FILE* file) const {
    long long int16Blocks = 0;
    long long int32Blocks = 0;
    long long peakBound = 0;
    bool adaptive = false;
    for (const std::unique_ptr<FarrarBlockEngine>& engine : farrar_engines) {
        if (!engine->isAdaptive()) continue;
        adaptive = true;
        int16Blocks += engine->getInt16Blocks();
        int32Blocks += engine->getInt32Blocks();
        peakBound = std::max(peakBound, engine->getPeakInt16Bound());
    }
    if (!adaptive) return;

    long long total = int16Blocks + int32Blocks;
    fprintf(file, "\n===== ADAPTIVE PRECISION =====\n");
    fprintf(file, "   int16 blocks: %lld (%.2f%%)\n", int16Blocks,
            total ? 100.0 * int16Blocks / total : 0.0);
    fprintf(file, "   int32 blocks: %lld\n", int32Blocks);
    fprintf(file, "   Peak int16 bound: %lld (limit %d)\n", peakBound, INT16_MAX - 128);
    fflush(file);
}
