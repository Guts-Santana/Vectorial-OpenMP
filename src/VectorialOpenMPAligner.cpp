/*******************************************************************************
 *
 * Copyright (c) 2010-2015   Edans Sandes
 *
 * This file is part of MASA-OpenMP.
 * 
 * MASA-OpenMP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MASA-OpenMP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MASA-OpenMP.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "VectorialOpenMPAligner.hpp"
#include "VectorialOpenMPAlignerParameters.hpp"
#include "VectorialCPUBlockProcessor.hpp"
#include <omp.h>

VectorialOpenMPAligner::VectorialOpenMPAligner(AbstractBlockProcessor* blockProcessor, BlockAlignerParameters* params)
		: AbstractBlockAligner(blockProcessor, params), vectorialProcessor(blockProcessor) {
	setPreferredSizes(0, omp_get_max_threads()*2);
}

VectorialOpenMPAligner::~VectorialOpenMPAligner() {
	// TODO Auto-generated destructor stub
}

void VectorialOpenMPAligner::scheduleBlocks(int grid_width, int grid_height) {
	for (int d = 0; (d < grid_height+grid_width+1) && mustContinue(); d++) {
		int bound = (d+1 < grid_width) ? d+1 : grid_width;
		#pragma omp parallel for schedule(dynamic,1)
		for (int bx=0; bx < bound; bx++) {
				int by=d-bx;
				if (by >= grid_height || !mustContinue()) continue;
				AbstractBlockAligner::alignBlock(bx, by);
		}
	}
}

void VectorialOpenMPAligner::alignBlock(int bx, int by, int i0, int j0, int i1, int j1) {
	/* The blocks in the top-most positions must read the first row */
	if (by == 0) {
		/* Receives the first row chunk [j0..j1] from the MASA-core */
		receiveFirstRow(row[bx], j1 - j0);
	}

	/* The blocks in the left-most positions must read the first column */
	if (bx == 0) {
		/* Puts the H[i-1][j-1] dependency in the first cell of the column. */
		col[by][0] = getFirstColumnTail();

		/* Receives the first column chunk from the MASA-core */
		receiveFirstColumn(col[by]+1, i1 - i0);
	}

	/*
	 * The processBlock function executes the SW/NW recurrence function
	 * for all the cells of this block.
	 * See the input/output parameters description in the function implementation.
	 */
	if (!mustContinue()) return; // MASA-core is telling to stop
	if (!AbstractBlockAligner::processBlock(bx, by, i0, j0, i1, j1)) {
		/* Pruned: MASA writes nothing, so write lower bounds for the next blocks */
		fillPrunedBorders(row[bx], col[by], i1 - i0, j1 - j0,
				toAlignmentType(getRecurrenceType()));
	}

	/*
	 * Dispatch special rows with a minimum defined distance (getSpecialRowInterval()).
	 */
	if (isSpecialRow(by)) {
		#pragma omp critical
		{
			if (bx == 0) {
				/* The left-most blocks must send an extra cell to represent the
				 * very first column of the matrix. */
				cell_t first_cell = getFirstColumnTail();
				first_cell.f = -INF;
				dispatchRow(i1, &first_cell, 1);
			}
			dispatchRow(i1, row[bx], j1-j0); // Dispatch the special row.
		}
	}

	/*
	 * The right-most blocks must dispatch the last column to MASA.
	 */
	if (isSpecialColumn(bx)) {
		if (by == 0) {
			/* The left-most blocks must send an extra cell to represent the
			 * very first column of the matrix. */
			cell_t first_cell = getFirstRowTail();
			first_cell.f = -INF;
			dispatchColumn(j1, &first_cell, 1);
		}
		dispatchColumn(j1, col[by]+1, i1-i0);
	}

}

/*
 * Borders left by a pruned block of R x C cells (SW or NW).
 *
 * MASA does not touch row[]/col[] for a pruned block, so the blocks below and to
 * its right would read values of an older block: a corner near 0 next to a top
 * row of 80000, which forces the adaptive engine into int32 (and is not even
 * guaranteed to be below the real value). Here every value written is a lower
 * bound of the real one, given inputs that are lower bounds themselves:
 *   - corner of the next block (col[0]): the top-right input row[C-1].h, the
 *     same value a computed block passes on;
 *   - bottom row: a gap down from the input above, R cells;
 *                 a gap along the row from the bottom-left input col[R].h;
 *   - right column: a gap along the row from the input on the left, C cells;
 *                   a gap down from the top-right input row[C-1].h;
 *   - every H: also the corner floor (cornerLowerBound) and, in SW only, 0.
 * The pruning itself only reads the block scores, so it does not change, and a
 * value that is never above the real one cannot make an alignment look better.
 */
void VectorialOpenMPAligner::fillPrunedBorders(cell_t* blockRow, cell_t* blockCol, int R, int C,
		AlignmentType type) {
	const ScoringScheme scoring = masaScoring();
	const long long go = scoring.gapOpen; // first gap cell
	const long long ge = scoring.gapExt;  // each extra gap cell
	const long long NEG = -INF;

	const long long corner = blockCol[0].h;
	const long long topRight = blockRow[C - 1].h;    // H(i0-1, j1-1)
	const long long bottomLeft = blockCol[R].h;      // H(i1-1, j0-1)
	long long floor = std::max(NEG, cornerLowerBound(corner, R, C, scoring));
	if (type == AlignmentType::Local)
		floor = std::max(floor, 0LL);

	/* Bottom row: F from the gap down, H also from the gap along the row */
	for (int j = 0; j < C; ++j) {
		const long long f = std::max(NEG, blockRow[j].h + go + (R - 1) * ge);
		const long long e = bottomLeft + go + j * ge;
		blockRow[j].h = static_cast<int>(std::max({floor, f, e}));
		blockRow[j].f = static_cast<int>(f);
	}

	/* Right column: E from the gap along the row, H also from the gap down */
	for (int i = 1; i <= R; ++i) {
		const long long e = std::max(NEG, blockCol[i].h + go + (C - 1) * ge);
		const long long f = topRight + go + (i - 1) * ge;
		blockCol[i].h = static_cast<int>(std::max({floor, e, f}));
		blockCol[i].e = static_cast<int>(e);
	}

	blockCol[0].h = static_cast<int>(topRight);
}

/*
 * Changes statistics in a safe way.
 */
void VectorialOpenMPAligner::increaseBlockStat(const bool pruned) {
	#pragma omp critical
	{
		AbstractBlockAligner::increaseBlockStat(pruned);
	}
}

/**
 * Prints the runtime parameters.
 * @param file handler to print out the statistics.
 * @see IAligner::printInitialStatistics
 */
void VectorialOpenMPAligner::printInitialStatistics(FILE* file) {
	AbstractBlockAligner::printInitialStatistics(file);
	fprintf(file, "Vectorial-OpenMP max threads: %d\n", omp_get_max_threads());
	const VectorialOpenMPAlignerParameters* params = dynamic_cast<const VectorialOpenMPAlignerParameters*>(getParameters());
	if (params != NULL) {
		fprintf(file, "Vectorial-OpenMP vector type: %s\n", params->getVector().c_str());
		fprintf(file, "Vectorial-OpenMP F strategy: %s\n", params->getFStrategy().c_str());
		fprintf(file, "Vectorial-OpenMP NW F strategy: %s\n", params->getNwFStrategy().c_str());
	}
}

/**
 * Prints the pruning statistics plus, for the adaptive vector types, how many
 * blocks ran in int16 and in int32.
 * @see IAligner::printStatistics
 */
void VectorialOpenMPAligner::printStatistics(FILE* file) {
	AbstractBlockAligner::printStatistics(file);
	const VectorialCPUBlockProcessor* processor = dynamic_cast<const VectorialCPUBlockProcessor*>(vectorialProcessor);
	if (processor != NULL) {
		processor->printAdaptiveStatistics(file);
	}
}
