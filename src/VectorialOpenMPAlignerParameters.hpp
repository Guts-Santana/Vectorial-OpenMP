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

#ifndef VECTORIAL_OPENMP_ALIGNER_PARAMETERS
#define VECTORIAL_OPENMP_ALIGNER_PARAMETERS

#include "libmasa/libmasa.hpp"
#include <string>

/**
 * Block aligner parameters plus the Farrar specific ones:
 *   --vector=TYPE        vector backend (AvxInt16, AvxInt32, RvvInt16M1 ... RvvInt32M8)
 *   --f-strategy=TYPE    F propagation strategy (lazy-f, prefix-scan-f)
 */
class VectorialOpenMPAlignerParameters : public BlockAlignerParameters {
public:
	VectorialOpenMPAlignerParameters();
	virtual ~VectorialOpenMPAlignerParameters();

	virtual int processArgument(int argc, char** argv);
	virtual void printUsage() const;

	const std::string& getVector() const;
	const std::string& getFStrategy() const;

private:
	std::string vector;
	std::string fStrategy;
};

#endif /* VECTORIAL_OPENMP_ALIGNER_PARAMETERS */
