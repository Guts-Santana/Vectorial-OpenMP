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

#ifndef VECTORIAL_OPENMP_ALIGNER
#define VECTORIAL_OPENMP_ALIGNER

#include "libmasa/libmasa.hpp"

class VectorialOpenMPAligner : public AbstractBlockAligner {
public:
	VectorialOpenMPAligner(AbstractBlockProcessor* blockProcessor = NULL, BlockAlignerParameters* params = NULL);
	virtual ~VectorialOpenMPAligner();
	virtual void printInitialStatistics(FILE* file);

protected:
	virtual void scheduleBlocks(int grid_width, int grid_height);
	virtual void alignBlock(int bx, int by, int i0, int j0, int i1, int j1);

	virtual void increaseBlockStat(const bool pruned);
};

#endif /* VECTORIAL_OPENMP_ALIGNER */
