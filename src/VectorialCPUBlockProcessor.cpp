#include "VectorialCPUBlockProcessor.hpp"
#include "VectorialOpenMPAlignerParameters.hpp"
#include "Backend.hpp"
#include "Farrar.hpp"

#include <omp.h>
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

const std::vector<std::string>& VectorialCPUBlockProcessor::supportedVectors() {
    static const std::vector<std::string> vectors = {
#if defined(USE_AVX)
        "AvxInt16", "AvxInt32"
#elif defined(USE_RVV)
        "RvvInt16M1", "RvvInt16M2", "RvvInt16M4", "RvvInt16M8",
        "RvvInt32M1", "RvvInt32M2", "RvvInt32M4", "RvvInt32M8"
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
#elif defined(USE_RVV)
    if (vector == "RvvInt16M1") return makeEngine<RvvInt16M1>(strategy);
    if (vector == "RvvInt16M2") return makeEngine<RvvInt16M2>(strategy);
    if (vector == "RvvInt16M4") return makeEngine<RvvInt16M4>(strategy);
    if (vector == "RvvInt16M8") return makeEngine<RvvInt16M8>(strategy);
    if (vector == "RvvInt32M1") return makeEngine<RvvInt32M1>(strategy);
    if (vector == "RvvInt32M2") return makeEngine<RvvInt32M2>(strategy);
    if (vector == "RvvInt32M4") return makeEngine<RvvInt32M4>(strategy);
    if (vector == "RvvInt32M8") return makeEngine<RvvInt32M8>(strategy);
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
