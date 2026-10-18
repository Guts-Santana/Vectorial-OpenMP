/*
 * Multi-Architecture Farrar
 * Copyright (C) 2026 Gustavo Santana Lima
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */


#ifndef FARRAR_HPP
#define FARRAR_HPP

#include <iostream>
#include <string>
#include <algorithm>
#include <iomanip>
#include "constants.hpp"
#include <vector>
#include <limits>

#include "Buffer.hpp"
#include "Backend.hpp"

#ifdef FARRAR_WITH_MASA
// cell_t / score_t / INF used by the MASA block interface (alignBlock)
#include "libmasa/libmasaTypes.hpp"
#endif

template <typename Backend>
class Farrar {
protected:
    using VecType    = typename Backend::VecType;
    using Ops        = typename Backend::Ops;
    using Traits     = typename Backend::Traits;
    using BufferType = Buffer<Backend>;


    private:
        Ops ops;

    int gap_open = GAP_OPEN;
    int gap_ext = GAP_EXT;
    int match = MATCH;
    int mismatch = MISMATCH;

    size_t VL;
    size_t segLen;
    int maxScore;
    std::string s0;
    std::string s1;
    std::string ftype;

    using PropagateF = void (Farrar<Backend>::*)(VecType&, VecType&, typename Traits::ElemType);
    PropagateF propagateF;

    BufferType* pvHStore  = nullptr;
    BufferType* pvHLoad   = nullptr;
    BufferType* pvE       = nullptr;
    BufferType* vProfile  = nullptr;
    // True F of every cell of the current column (only used by alignBlock)
    BufferType* pvF       = nullptr;

    // "Minus infinity" that still leaves headroom for the non-saturating adds
    static constexpr typename Traits::ElemType LOW =
        std::numeric_limits<typename Traits::ElemType>::min() / 4;

    // MASA boundaries are int and may contain -INF (-999999999): clamp into lane range
    static typename Traits::ElemType toLane(int v)
    {
        constexpr int HIGH = std::numeric_limits<typename Traits::ElemType>::max();
        return static_cast<typename Traits::ElemType>(v < LOW ? LOW : (v > HIGH ? HIGH : v));
    }

    static constexpr std::array<char, 5> alphabet = {
        'A', 'C', 'G', 'T', 'N'
    };

    public:
    Farrar(std::string s0, std::string s1, std::string ftype) : maxScore(0), s0(s0), s1(s1), ftype(ftype)
    {
        VL = Traits::maxVL();
        ops.setVL(VL);
        segLen = (s0.length() + VL - 1) / VL;

        if (ftype == "prefix-scan-f" || ftype == "prefix")
        {
            propagateF = &Farrar<Backend>::propagatePrefixScanF;
        }
        else
        {
            propagateF = &Farrar<Backend>::propagateLazyF;
        }
    }


    ~Farrar()
    {
        clearData();
    }

    void setSequences(std::string s0, std::string s1);

    void buildProfile();

    void initMatrices();

    VecType processColumn(int column);

    void processPrimaryPass(int column, VecType &vF, VecType &vMax);

    void propagatePrefixScanF(VecType &vF, VecType &vMax, typename Traits::ElemType top_f);

    void propagateLazyF(VecType &vF, VecType &vMax, typename Traits::ElemType top_f);

    int obtainScore();

    void clearData();

    inline int charToIndex(char c)
    {
        switch (c)
        {
        case 'A':
        case 'a':
            return 0;

        case 'C':
        case 'c':
            return 1;

        case 'G':
        case 'g':
            return 2;

        case 'T':
        case 't':
            return 3;

        case 'N':
        case 'n':
            return 4;

        default:
            std::cerr << "Invalid character: "
                      << (int)(unsigned char)c
                      << " (0x"
                      << std::hex << (int)(unsigned char)c
                      << std::dec << ")\n";
            return -1;
        }
    }


    //Implemented to MASA
    Farrar(std::string ftype) : segLen(0), maxScore(0), ftype(ftype)
    {
        VL = Traits::maxVL();
        ops.setVL(VL);
        
        if (ftype == "prefix-scan-f" || ftype == "prefix") {
            propagateF = &Farrar<Backend>::propagatePrefixScanF;
        } else {
            propagateF = &Farrar<Backend>::propagateLazyF;
        }
    }

#ifdef FARRAR_WITH_MASA
    score_t alignBlock(
        const char* query_seg, int q_len,      
        const char* target_seg, int t_len,     
        cell_t* row,                  
        cell_t* col,                  
        int abs_i, int abs_j                  
    );

    void processPrimaryPass(
        char target_char, 
        typename Traits::ElemType diag_h, 
        typename Traits::ElemType top_f, 
        VecType &vF, 
        VecType &vMax
    );

    void initializeStripedLeftColumn(cell_t* col_buffer, int q_len);
    void unstripeRightColumnE(cell_t* col_buffer, int q_len);
    void unstripeRightColumnH(cell_t* col_buffer, int q_len);
    
    // Build a temporary profile specifically for a block's query segment
    void buildBlockProfile(const char* query_seg, int q_len);
#endif

};




enum class RvvType {
    vint16m1, vint16m2, vint16m4, vint16m8,
    vint32m1, vint32m2, vint32m4, vint32m8,
    Unknown
};

inline RvvType parseRvvType(const std::string& str) {
    if (str == "vint16m1_t") return RvvType::vint16m1;
    if (str == "vint16m2_t") return RvvType::vint16m2;
    if (str == "vint16m4_t") return RvvType::vint16m4;
    if (str == "vint16m8_t") return RvvType::vint16m8;
    if (str == "vint32m1_t") return RvvType::vint32m1;
    if (str == "vint32m2_t") return RvvType::vint32m2;
    if (str == "vint32m4_t") return RvvType::vint32m4;
    if (str == "vint32m8_t") return RvvType::vint32m8;
    return RvvType::Unknown;
}

#endif
