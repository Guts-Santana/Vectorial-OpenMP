#ifndef VECTORIAL_CPU_BLOCK_PROCESSOR
#define VECTORIAL_CPU_BLOCK_PROCESSOR

#include "AbstractBlockProcessor.hpp"
#include "CPUBlockProcessor.hpp"
#include "Farrar.hpp"
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class VectorialOpenMPAlignerParameters;

/**
 * MASA scores (CPUBlockProcessor.hpp) in Farrar's convention. The block floor
 * is cornerLowerBound(corner, R, C, masaScoring()) (Farrar.hpp).
 */
ScoringScheme masaScoring();

/** MASA's recurrence type (SMITH_WATERMAN / NEEDLEMAN_WUNSCH) as Farrar's */
inline AlignmentType toAlignmentType(int recurrenceType) {
    return (recurrenceType == NEEDLEMAN_WUNSCH) ? AlignmentType::Global : AlignmentType::Local;
}

/**
 * Type-erased Farrar engine, so the vector backend (AvxInt16, RvvInt32M4, ...)
 * can be chosen at runtime with --vector.
 */
class FarrarBlockEngine {
public:
    virtual ~FarrarBlockEngine() = default;
    virtual score_t alignBlock(const char* query_seg, int q_len,
                               const char* target_seg, int t_len,
                               cell_t* row, cell_t* col,
                               int abs_i, int abs_j,
                               AlignmentType type) = 0;

    /* Only the adaptive engines count anything */
    virtual bool isAdaptive() const { return false; }
    virtual long long getInt16Blocks() const { return 0; }
    virtual long long getInt32Blocks() const { return 0; }
    virtual long long getPeakInt16Bound() const { return 0; }
};

class VectorialCPUBlockProcessor : public AbstractBlockProcessor {
private:
    const VectorialOpenMPAlignerParameters* params;

    // One engine per OpenMP thread: Farrar keeps its buffers as member state,
    // and VectorialOpenMPAligner calls processBlock from inside an omp parallel for.
    // Created on the first setSequences(), after the command line was parsed.
    std::vector<std::unique_ptr<FarrarBlockEngine>> farrar_engines;

    // Fallback for a recurrence other than SW / NW (MASA has none today)
    CPUBlockProcessor cpu_processor;

    const char* full_seq0;
    const char* full_seq1;

public:
    VectorialCPUBlockProcessor(const VectorialOpenMPAlignerParameters* params);
    virtual ~VectorialCPUBlockProcessor() = default;

    virtual void setSequences(const char* seq0, const char* seq1, int seq0_len, int seq1_len) override;
    virtual void unsetSequences() override;

    virtual score_t processBlock(cell_t *row, cell_t *col,
                                 const int i0, const int j0,
                                 const int i1, const int j1,
                                 const int recurrenceType) override;

    /* Options available in this build (depends on --with-vec-arch) */
    static const std::vector<std::string>& supportedVectors();
    static bool isSupportedVector(const std::string& vector);
    static std::string defaultVector();

    static bool isSupportedFStrategy(const std::string& strategy);
    static std::string normalizeFStrategy(const std::string& strategy);
    static std::string defaultFStrategy();
    static std::string defaultNwFStrategy();

    /* strategy is used for the SW blocks, nwStrategy for the NW ones */
    static std::unique_ptr<FarrarBlockEngine> createEngine(const std::string& vector,
                                                           const std::string& strategy,
                                                           const std::string& nwStrategy);

    /* Adaptive-precision statistics, summed over the thread engines */
    void printAdaptiveStatistics(FILE* file) const;
};

#endif /* VECTORIAL_CPU_BLOCK_PROCESSOR */
