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

public:
    FarrarBlockEngineImpl(const std::string& strategy, const ScoringScheme& scoring)
        : farrar(strategy, scoring) {}

    virtual score_t alignBlock(const char* query_seg, int q_len,
                               const char* target_seg, int t_len,
                               cell_t* row, cell_t* col,
                               int abs_i, int abs_j) override {
        return farrar.alignBlock(query_seg, q_len, target_seg, t_len, row, col, abs_i, abs_j);
    }
};

/*
 * MASA scores (CPUBlockProcessor.hpp) in Farrar's convention. MASA's penalties
 * are positive and a gap of length k costs DNA_GAP_OPEN + k * DNA_GAP_EXT, while
 * Farrar's are negative and its gapOpen is the cost of the whole first gap cell.
 */
static ScoringScheme masaScoring() {
    ScoringScheme scoring;
    scoring.match    = DNA_MATCH;
    scoring.mismatch = DNA_MISMATCH;
    scoring.gapOpen  = -DNA_GAP_FIRST;
    scoring.gapExt   = -DNA_GAP_EXT;
    return scoring;
}

template <typename Backend>
static std::unique_ptr<FarrarBlockEngine> makeEngine(const std::string& strategy) {
    return std::unique_ptr<FarrarBlockEngine>(new FarrarBlockEngineImpl<Backend>(strategy, masaScoring()));
}

/*
 * Adaptive precision: before each block, decide whether it can run in int16
 * lanes (after rebasing the scores by a bias b) or must run in int32.
 *
 * Lower bound b. Only the corner H(0,0) = col[0].h is trusted. The path
 *   corner -> diagonal into (1,1) -> horizontal gap along row 1 -> vertical gap
 * reaches every cell (i,j) of the R x C block, and the worst cell is (R,C):
 *   H(i,j) >= corner + mismatch + 2*gapOpen + (R+C-4)*gapExt
 * (Farrar's convention: a gap of length k costs gapOpen + (k-1)*gapExt).
 * With b = max(0, that bound), every H of the block is >= b, so the lanes
 * hold H - b >= 0 and the SW floor max(0, .) of the lanes never binds where
 * the real floor would not: the kernel stays exact unchanged.
 *
 * Upper bound. The boundary is scanned (R + C + 1 cells, ~0.2% of the block
 * work for 1024 x 1024). The scan is used instead of the bests of the
 * neighbour blocks because MASA does not write anything for a pruned block:
 * row[]/col[] keep stale values of an older block while its grid score stays
 * -INF, and the first row/column of a partition comes from another partition.
 * A path entering the block from a boundary cell gains at most match per
 * diagonal step, so every value of the block is <= bmax + match*min(R,C).
 *
 * Exactness of the values written back to MASA (special rows/columns): the
 * clamp to LOW can only raise boundary values that are < b + LOW. For R, C >= 2
 * the H, E and F that leave the block are computed from interior cells
 * (relative H >= 0, so E, F >= gapOpen), which dominate any clamped value, so
 * they are exact. Blocks with R < 2 or C < 2 always use int32.
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

    long long int16Blocks = 0;
    long long int32Blocks = 0;
    long long peakInt16Bound = 0;

public:
    AdaptiveFarrarBlockEngine(const std::string& strategy, const ScoringScheme& scoring)
        : farrar16(strategy, scoring), farrar32(strategy, scoring), scoring(scoring) {}

    virtual score_t alignBlock(const char* query_seg, int q_len,
                               const char* target_seg, int t_len,
                               cell_t* row, cell_t* col,
                               int abs_i, int abs_j) override {
        const long long R = q_len;
        const long long C = t_len;

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

            const long long lower = corner + std::min(scoring.mismatch, scoring.match)
                                  + 2LL * scoring.gapOpen + (R + C - 4) * scoring.gapExt;
            const long long b = std::max(0LL, lower);
            const long long bound = std::max(bmax - b, 0LL) + scoring.match * std::min(R, C);

            if (bound <= LIMIT) {
                ++int16Blocks;
                peakInt16Bound = std::max(peakInt16Bound, bound);
                return farrar16.alignBlock(query_seg, q_len, target_seg, t_len,
                                           row, col, abs_i, abs_j, static_cast<int>(b));
            }
        }

        ++int32Blocks;
        return farrar32.alignBlock(query_seg, q_len, target_seg, t_len,
                                   row, col, abs_i, abs_j, 0);
    }

    virtual bool isAdaptive() const override { return true; }
    virtual long long getInt16Blocks() const override { return int16Blocks; }
    virtual long long getInt32Blocks() const override { return int32Blocks; }
    virtual long long getPeakInt16Bound() const override { return peakInt16Bound; }
};

template <typename Backend16, typename Backend32>
static std::unique_ptr<FarrarBlockEngine> makeAdaptiveEngine(const std::string& strategy) {
    return std::unique_ptr<FarrarBlockEngine>(
            new AdaptiveFarrarBlockEngine<Backend16, Backend32>(strategy, masaScoring()));
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

std::unique_ptr<FarrarBlockEngine> VectorialCPUBlockProcessor::createEngine(
        const std::string& vector, const std::string& strategy) {
#if defined(USE_AVX)
    if (vector == "AvxInt16") return makeEngine<AvxInt16>(strategy);
    if (vector == "AvxInt32") return makeEngine<AvxInt32>(strategy);
    if (vector == "AvxAdaptive") return makeAdaptiveEngine<AvxInt16, AvxInt32>(strategy);
#elif defined(USE_RVV)
    if (vector == "RvvInt16M1") return makeEngine<RvvInt16M1>(strategy);
    if (vector == "RvvInt16M2") return makeEngine<RvvInt16M2>(strategy);
    if (vector == "RvvInt16M4") return makeEngine<RvvInt16M4>(strategy);
    if (vector == "RvvInt16M8") return makeEngine<RvvInt16M8>(strategy);
    if (vector == "RvvInt32M1") return makeEngine<RvvInt32M1>(strategy);
    if (vector == "RvvInt32M2") return makeEngine<RvvInt32M2>(strategy);
    if (vector == "RvvInt32M4") return makeEngine<RvvInt32M4>(strategy);
    if (vector == "RvvInt32M8") return makeEngine<RvvInt32M8>(strategy);
    // Same LMUL for both: int16 then has twice the lanes of int32
    if (vector == "RvvAdaptiveM1") return makeAdaptiveEngine<RvvInt16M1, RvvInt32M1>(strategy);
    if (vector == "RvvAdaptiveM2") return makeAdaptiveEngine<RvvInt16M2, RvvInt32M2>(strategy);
    if (vector == "RvvAdaptiveM4") return makeAdaptiveEngine<RvvInt16M4, RvvInt32M4>(strategy);
    if (vector == "RvvAdaptiveM8") return makeAdaptiveEngine<RvvInt16M8, RvvInt32M8>(strategy);
#elif defined(USE_SVE)
    if (vector == "SveInt16") return makeEngine<SveInt16>(strategy);
    if (vector == "SveInt32") return makeEngine<SveInt32>(strategy);
    if (vector == "SveAdaptive") return makeAdaptiveEngine<SveInt16, SveInt32>(strategy);
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
                createEngine(params->getVector(), params->getFStrategy());
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

    if (recurrenceType != SMITH_WATERMAN) {
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
            i0, j0);
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
    fprintf(file, "   Peak int16 bound: %lld (limit %d)\n", peakBound, INT16_MAX);
    fflush(file);
}
