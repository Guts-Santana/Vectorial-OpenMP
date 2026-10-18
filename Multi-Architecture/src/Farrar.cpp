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


#include "Farrar.hpp"

template <typename Backend>
void Farrar<Backend>::setSequences(std::string s0, std::string s1)
{
    this->s0 = s0;
    this->s1 = s1;
}


template <typename Backend>
int Farrar<Backend>::obtainScore()
{
    initMatrices();
    buildProfile();
    VecType vMaxBlock = Traits::set(0, VL);
    for (int i = 0; i < s1.length(); i++)
    {
        vMaxBlock = ops.max(vMaxBlock, processColumn(i));
    }
    maxScore = static_cast<int>(ops.maxValue(vMaxBlock));
    return maxScore;
}

template <typename Backend>
void Farrar<Backend>::buildProfile()
{

    size_t total_blocks = alphabet.size() * segLen;

    posix_memalign((void **)&vProfile, 64, total_blocks * sizeof(BufferType));

    for (size_t k = 0; k < total_blocks; ++k)
    {
        new (&vProfile[k]) BufferType(VL, 0);
    }

    for (char residue : alphabet)
    {
        int resIdx = charToIndex(residue);
        if (resIdx == -1)
            continue;

        size_t baseOffset = static_cast<size_t>(resIdx) * segLen;

        for (size_t i = 0; i < segLen; i++)
        {
            for (size_t j = 0; j < VL; j++)
            {
                size_t idx = j * segLen + i;
                vProfile[baseOffset + i][j] = (idx < s0.length() && s0[idx] == residue) ? match : mismatch;
            }
        }
    }
}

template <typename Backend>
void Farrar<Backend>::initMatrices()
{
    segLen = (s0.length() + VL - 1) / VL;

    clearData();

    posix_memalign((void **)&pvHStore, 64, segLen * sizeof(BufferType));
    posix_memalign((void **)&pvHLoad, 64, segLen * sizeof(BufferType));
    posix_memalign((void **)&pvE, 64, segLen * sizeof(BufferType));

    for (size_t j = 0; j < segLen; ++j)
    {
        new (&pvHStore[j]) BufferType(VL, 0);
        new (&pvHLoad[j]) BufferType(VL, 0);
        new (&pvE[j]) BufferType(VL, 0);
    }
}

template <typename Backend>
typename Farrar<Backend>::VecType
Farrar<Backend>::processColumn(int column)
{
    VecType vF = Traits::set(0, VL);
    VecType vMax = Traits::set(0, VL);

    processPrimaryPass(column, vF, vMax);
    (this->*propagateF)(vF, vMax, 0);

    return vMax;
}

template <typename Backend>
void Farrar<Backend>::processPrimaryPass(int column, VecType &vF, VecType &vMax)
{
    VecType vE;
    VecType vH = pvHStore[segLen - 1].load();
    vH = ops.shift(vH, 0);

    std::swap(pvHStore, pvHLoad);

    int profileIndex = charToIndex(s1[column]);
    int baseIndex = profileIndex * segLen;

    for (int j = 0; j < segLen; j++)
    {
        if (profileIndex == -1)
        {
            vH = ops.add(vH, mismatch);
        }
        else
        {
            VecType vProfileTemp = vProfile[baseIndex + j].load();
            vH = ops.add(vH, vProfileTemp);
        }

        vE = pvE[j].load();

        vH = ops.max(vH, vE);
        vH = ops.max(vH, vF);
        vH = ops.max(vH, 0);

        vMax = ops.max(vMax, vH);

        pvHStore[j].store(vH);

        vH = ops.add(vH, gap_open);
        vE = ops.add(vE, gap_ext);
        vE = ops.max(vE, vH);
        pvE[j].store(vE);

        vF = ops.add(vF, gap_ext);
        vF = ops.max(vF, vH);
        vH = pvHLoad[j].load();
    }
}

template <typename Backend>
void Farrar<Backend>::clearData()
{
    if (pvHStore)
    {
        free(pvHStore);
        pvHStore = nullptr;

        free(pvHLoad);
        pvHLoad = nullptr;

        free(pvE);
        pvE = nullptr;

        free(vProfile);
        vProfile = nullptr;
    }
    if (pvF)
    {
        free(pvF);
        pvF = nullptr;
    }
}


#ifdef FARRAR_WITH_MASA

template <typename Backend>
void Farrar<Backend>::initializeStripedLeftColumn(cell_t* col_buffer, int q_len) 
{
    // 1. Reset all lanes (important for the padding area at the end of the query)
    for (size_t idx = 0; idx < segLen; ++idx) {
        pvHStore[idx].store(Traits::set(0, VL));
        pvE[idx].store(Traits::set(LOW, VL));
    }

    // 2. Interleave MASA's left boundary into the striped SIMD buffers.
    //    MASA keeps E[i][j0-1]; Farrar's pvE keeps the E already prepared for
    //    the current column: E[i][j0] = max(H[i][j0-1] + open, E[i][j0-1] + ext)
    for (int i = 0; i < q_len; ++i) 
    {
        int lane = i / segLen;
        int idx  = i % segLen;

        int h = toLane(col_buffer[i + 1].h);
        int e = toLane(col_buffer[i + 1].e);
        pvHStore[idx][lane] = h;
        pvE[idx][lane]      = toLane(std::max(h + gap_open, e + gap_ext));
    }
}

template <typename Backend>
void Farrar<Backend>::unstripeRightColumnE(cell_t* col_buffer, int q_len) 
{
    // Must be called right before the last column is processed: at that point
    // pvE holds exactly E[i][j1-1], which is what MASA expects in col[].e
    for (int i = 0; i < q_len; ++i) 
    {
        col_buffer[i + 1].e = pvE[i % segLen][i / segLen];
    }
}

template <typename Backend>
void Farrar<Backend>::unstripeRightColumnH(cell_t* col_buffer, int q_len) 
{
    for (int i = 0; i < q_len; ++i) 
    {
        col_buffer[i + 1].h = pvHStore[i % segLen][i / segLen];
    }
}

template <typename Backend>
void Farrar<Backend>::buildBlockProfile(const char* query_seg, int q_len) 
{
    // Padding rows (i >= q_len) get LOW, otherwise their diagonal (a cell of the
    // left boundary column, outside the block) could beat the block's best score
    for (int i = 0; i < alphabet.size(); ++i) {
        for (size_t s = 0; s < segLen; ++s) {
            vProfile[i * segLen + s].store(Traits::set(LOW, VL));
        }
    }

    for (int i = 0; i < q_len; ++i) {
        int char_idx = charToIndex(query_seg[i]);
        int lane = i / segLen;
        int idx  = i % segLen;

        // Direct scalar write to the buffer
        for (int c = 0; c < alphabet.size(); ++c) {
            vProfile[c * segLen + idx][lane] = (c == char_idx) ? match : mismatch;
        }
    }
}

template <typename Backend>
score_t Farrar<Backend>::alignBlock(
    const char* query_seg, int q_len,
    const char* target_seg, int t_len,
    cell_t* row,
    cell_t* col,
    int abs_i, int abs_j)
{
    size_t newSegLen = (q_len + VL - 1) / VL;

    // Allocate memory ONLY if pointers are null or size changed
    if (newSegLen != segLen || !pvHStore) {
        if (pvHStore) { free(pvHStore); pvHStore = nullptr; }
        if (pvHLoad)  { free(pvHLoad);  pvHLoad  = nullptr; }
        if (pvE)      { free(pvE);      pvE      = nullptr; }
        if (pvF)      { free(pvF);      pvF      = nullptr; }
        if (vProfile) { free(vProfile); vProfile = nullptr; }

        segLen = newSegLen;

        posix_memalign((void **)&pvHStore, 64, segLen * sizeof(BufferType));
        posix_memalign((void **)&pvHLoad,  64, segLen * sizeof(BufferType));
        posix_memalign((void **)&pvE,      64, segLen * sizeof(BufferType));
        posix_memalign((void **)&pvF,      64, segLen * sizeof(BufferType));
        posix_memalign((void **)&vProfile, 64, alphabet.size() * segLen * sizeof(BufferType));

        for (size_t j = 0; j < segLen; ++j) {
            new (&pvHStore[j]) BufferType(VL, 0);
            new (&pvHLoad[j])  BufferType(VL, 0);
            new (&pvE[j])      BufferType(VL, 0);
            new (&pvF[j])      BufferType(VL, 0);
        }
        for (size_t k = 0; k < alphabet.size() * segLen; ++k) {
            new (&vProfile[k]) BufferType(VL, 0);
        }
    }

    buildBlockProfile(query_seg, q_len);
    initializeStripedLeftColumn(col, q_len);

    score_t block_best;
    block_best.score = -INF;
    block_best.i = -1;
    block_best.j = -1;

    int last_lane = (q_len - 1) / segLen;
    int last_idx  = (q_len - 1) % segLen;

    // H[i0-1][j-1]: diagonal of the first row of column j
    int diag_h = toLane(col[0].h);
    // H[i0-1][j1-1] is the diagonal cell of the right neighbour block
    int corner_h = row[t_len - 1].h;

    for (int j = 0; j < t_len; ++j) 
    {
        int top_h = toLane(row[j].h);
        int top_f = toLane(row[j].f);
        // F entering the first row: F[i0][j] = max(H[i0-1][j] + open, F[i0-1][j] + ext)
        typename Traits::ElemType first_f = toLane(std::max(top_h + gap_open, top_f + gap_ext));

        VecType vF = Traits::set(LOW, VL);
        VecType vMax = Traits::set(0, VL);

        // pvE currently holds E[*][j1-1]: this is MASA's right column E
        if (j == t_len - 1) unstripeRightColumnE(col, q_len);

        processPrimaryPass(target_seg[j], diag_h, first_f, vF, vMax);
        (this->*propagateF)(vF, vMax, first_f);

        diag_h = top_h;

        // Overwrite row boundary for MASA (last row of the block)
        row[j].h = pvHStore[last_idx][last_lane];
        row[j].f = pvF[last_idx][last_lane];

        int col_max = ops.maxValue(vMax);
        if (col_max > block_best.score) {
            block_best.score = col_max;
            block_best.j = abs_j + j;
            // Locate the row of the maximum (first one, padding rows excluded)
            for (int i = 0; i < q_len; ++i) {
                if (pvHStore[i % segLen][i / segLen] == col_max) {
                    block_best.i = abs_i + i;
                    break;
                }
            }
        }
    }

    // Overwrite column boundary for MASA
    unstripeRightColumnH(col, q_len);
    col[0].h = corner_h;

    return block_best;
}

template <typename Backend>
void Farrar<Backend>::processPrimaryPass(
    char target_char, 
    typename Traits::ElemType diag_h, 
    typename Traits::ElemType top_f, 
    VecType &vF, 
    VecType &vMax)
{
    VecType vE;
    VecType vH = pvHStore[segLen - 1].load();
    vH = ops.shift(vH, diag_h);

    std::swap(pvHStore, pvHLoad);

    // Direct character lookup without indexing s1
    int profileIndex = charToIndex(target_char);
    int baseIndex = profileIndex * segLen;

    for (int j = 0; j < segLen; j++)
    {
        if (profileIndex == -1)
        {
            vH = ops.add(vH, mismatch);
        }
        else
        {
            VecType vProfileTemp = vProfile[baseIndex + j].load();
            vH = ops.add(vH, vProfileTemp);
        }

        vE = pvE[j].load();

        // F of this segment computed so far (cross-lane part is fixed by propagateF)
        pvF[j].store(vF);

        vH = ops.max(vH, vE);
        vH = ops.max(vH, vF);
        vH = ops.max(vH, 0); // Smith-Waterman floor at 0

        vMax = ops.max(vMax, vH);

        pvHStore[j].store(vH);

        vH = ops.add(vH, gap_open);
        vE = ops.add(vE, gap_ext);
        vE = ops.max(vE, vH);
        pvE[j].store(vE);

        vF = ops.add(vF, gap_ext);
        vF = ops.max(vF, vH);
        vH = pvHLoad[j].load();
    }
}

#endif // FARRAR_WITH_MASA

template <typename Backend>
void Farrar<Backend>::propagatePrefixScanF(VecType &vF, VecType &vMax, typename Traits::ElemType top_f)
{
    int accumulatedDecay;
    vF = ops.shift(vF, top_f);
    VecType vH, vShift;

    for (size_t offset = 1; offset < VL; offset <<= 1)
    {
        vShift = ops.slideup(vF, offset);

        // Consecutive lanes are segLen rows apart
        accumulatedDecay = std::max<int>(offset * segLen * gap_ext, LOW);
        vShift = ops.add(vShift, accumulatedDecay);

        // AVX slideup fills lanes < offset with 0 (a fake F of 0): push them to LOW
        VecType vFillMask = ops.add(ops.slideup(Traits::set(-LOW, VL), offset), LOW);
        vShift = ops.add(vShift, vFillMask);

        vF = ops.max(vF, vShift);
    }

    for (size_t j = 0; j < segLen; j++)
    {
        vH = pvHStore[j].load();
        vH = ops.max(vH, vF);
        pvHStore[j].store(vH);
        vMax = ops.max(vMax, vH);
        if (pvF)
        {
            pvF[j].store(ops.max(pvF[j].load(), vF));
            pvE[j].store(ops.max(pvE[j].load(), ops.add(vH, gap_open)));
        }
        vF = ops.add(vF, gap_ext);
    }
}

template <typename Backend>
void Farrar<Backend>::propagateLazyF(VecType &vF, VecType &vMax, typename Traits::ElemType top_f)
{
    vF = ops.shift(vF, top_f);
    size_t j = 0;
    VecType vHStore = pvHStore[j].load();
    // Segment 0 has no in-lane F from the primary pass: always merge it
    if (pvF) pvF[0].store(ops.max(pvF[0].load(), vF));
    while (ops.anyBiggerElement(vF, ops.add(vHStore, gap_open)))
    {
        vHStore = pvHStore[j].load();
        vHStore = ops.max(vHStore, vF);
        pvHStore[j].store(vHStore);
        vMax = ops.max(vMax, vHStore);
        if (pvF)
        {
            pvF[j].store(ops.max(pvF[j].load(), vF));
            // H may have grown: the E prepared for the next column must follow
            pvE[j].store(ops.max(pvE[j].load(), ops.add(vHStore, gap_open)));
        }

        j++;
        vF = ops.add(vF, gap_ext);

        if (j >= segLen)
        {
            // Lane k continues into lane k+1; nothing flows from the bottom row back into row 0
            vF = ops.shift(vF, LOW);
            j = 0;
            if (pvF) pvF[0].store(ops.max(pvF[0].load(), vF));
        }
    }
}



#if defined(USE_RVV)

template class Farrar<RvvInt16M1>;
template class Farrar<RvvInt16M2>;
template class Farrar<RvvInt16M4>;
template class Farrar<RvvInt16M8>;

template class Farrar<RvvInt32M1>;
template class Farrar<RvvInt32M2>;
template class Farrar<RvvInt32M4>;
template class Farrar<RvvInt32M8>;

#endif

#if defined(USE_AVX)

template class Farrar<AvxInt16>;
template class Farrar<AvxInt32>;

#endif