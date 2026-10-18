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

#include "VectorialOpenMPAlignerParameters.hpp"
#include "VectorialCPUBlockProcessor.hpp"

#include <getopt.h>
#include <sstream>

#define USAGE_HEADER	"Vectorial (Farrar) Parameters"

/**
 * getopt arguments (only long options: -v is already MASA's --verbose)
 */
#define ARG_VECTOR       0x2001
#define ARG_F_STRATEGY   0x2002

static struct option long_options[] = {
        {"vector",       required_argument,      0, ARG_VECTOR},
        {"f-strategy",   required_argument,      0, ARG_F_STRATEGY},
        {0, 0, 0, 0}
    };

VectorialOpenMPAlignerParameters::VectorialOpenMPAlignerParameters()
	: vector(VectorialCPUBlockProcessor::defaultVector()),
	  fStrategy(VectorialCPUBlockProcessor::defaultFStrategy())
{
}

VectorialOpenMPAlignerParameters::~VectorialOpenMPAlignerParameters()
{
}

void VectorialOpenMPAlignerParameters::printUsage() const {
	BlockAlignerParameters::printUsage();

	std::stringstream usage;
	usage << "--vector=TYPE                Vector type used by Farrar. Default: "
	      << VectorialCPUBlockProcessor::defaultVector() << ".\n"
	      << "                                Available in this build:";
	for (const std::string& v : VectorialCPUBlockProcessor::supportedVectors()) {
		usage << " " << v;
	}
	usage << ".\n"
	      << "--f-strategy=TYPE            F propagation strategy: lazy-f or prefix-scan-f.\n"
	      << "                                Default: " << VectorialCPUBlockProcessor::defaultFStrategy() << ".";
	AbstractAlignerParameters::printFormattedUsage(USAGE_HEADER, usage.str().c_str());
}

int VectorialOpenMPAlignerParameters::processArgument(int argc, char** argv) {
	int ret = AbstractAlignerParameters::callGetOpt(argc, argv, long_options);
	switch ( ret ) {
		case ARG_VECTOR:
			if (!VectorialCPUBlockProcessor::isSupportedVector(optarg)) {
				std::stringstream out;
				out << "Unsupported vector type '" << optarg << "'. Available:";
				for (const std::string& v : VectorialCPUBlockProcessor::supportedVectors()) {
					out << " " << v;
				}
				setLastError(out.str().c_str());
				return ARGUMENT_ERROR;
			}
			vector = optarg;
			return ARGUMENT_OK;
		case ARG_F_STRATEGY:
			if (!VectorialCPUBlockProcessor::isSupportedFStrategy(optarg)) {
				setLastError("F strategy must be lazy-f or prefix-scan-f.");
				return ARGUMENT_ERROR;
			}
			fStrategy = VectorialCPUBlockProcessor::normalizeFStrategy(optarg);
			return ARGUMENT_OK;
		case ARGUMENT_ERROR_NOT_FOUND:
			// Not one of ours: let the block aligner parameters try it
			return BlockAlignerParameters::processArgument(argc, argv);
		default:
			return ret;
	}
}

const std::string& VectorialOpenMPAlignerParameters::getVector() const {
	return vector;
}

const std::string& VectorialOpenMPAlignerParameters::getFStrategy() const {
	return fStrategy;
}
