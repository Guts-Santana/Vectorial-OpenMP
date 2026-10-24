AUTOMAKE_OPTIONS = subdir-objects
SUBDIRS = $(LIBMASA_PATH)

bin_PROGRAMS = vectorial-openmp

# Farrar (multi-architecture) lives in its own project, shared with the
# standalone benchmark: only its headers and src/Farrar.cpp are used here
FARRAR_PATH = libs/farrar

# Global Include Paths
AM_CPPFLAGS = \
    -I$(top_srcdir)/src \
    -I$(top_srcdir)/$(FARRAR_PATH)/include \
    -I$(top_srcdir)/$(LIBMASA_PATH)/src \
    -I$(top_srcdir)/$(LIBMASA_PATH)/src/libmasa/processors \
    -DFARRAR_WITH_MASA

# Base Compiler & Linker Flags
vectorial_openmp_CXXFLAGS = -std=gnu++17 -fno-strict-aliasing
vectorial_openmp_LDADD    = $(LIBMASA_PATH)/libmasa.a -lpthread -lgomp

# Architecture-Specific Flags (./configure --with-vec-arch=avx|rvv)
if USE_AVX
vectorial_openmp_CXXFLAGS += -mavx2 -malign-double
endif

if USE_RVV
vectorial_openmp_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

# Source Files
vectorial_openmp_SOURCES = \
    src/main.cpp \
    src/VectorialOpenMPAligner.cpp \
    src/VectorialOpenMPAlignerParameters.cpp \
    src/VectorialCPUBlockProcessor.cpp \
    $(FARRAR_PATH)/src/Farrar.cpp

# Headers (Ensures they are included in source distributions)
noinst_HEADERS = \
    src/VectorialOpenMPAligner.hpp \
    src/VectorialOpenMPAlignerParameters.hpp \
    src/VectorialCPUBlockProcessor.hpp \
    $(FARRAR_PATH)/include/Backend.hpp \
    $(FARRAR_PATH)/include/Buffer.hpp \
    $(FARRAR_PATH)/include/Farrar.hpp \
    $(FARRAR_PATH)/include/AVX/AvxOps.hpp \
    $(FARRAR_PATH)/include/AVX/AvxTraits.hpp \
    $(FARRAR_PATH)/include/RVV/RvvOps.hpp \
    $(FARRAR_PATH)/include/RVV/RvvTraits.hpp

dist_doc_DATA = README.md

# Symlink helper targets for distribution builds
dist-clone-link:
	@if test -L $(LIBMASA_PATH); then \
		mv $(LIBMASA_PATH) $(LIBMASA_PATH).link && \
		cp -r -L $(LIBMASA_PATH).link $(LIBMASA_PATH); \
	fi

dist-unclone-link:
	@if test -L $(LIBMASA_PATH).link; then \
		rm -rf $(LIBMASA_PATH) && \
		mv $(LIBMASA_PATH).link $(LIBMASA_PATH); \
	fi

dist-link: dist-clone-link dist dist-unclone-link