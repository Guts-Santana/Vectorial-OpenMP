# Builds vectorial-openmp here, so that the object files go to bin/ and not
# next to the sources in src/ (or inside the Farrar submodule).
# The sources are listed by name only (so automake puts the objects here) and
# GNU make finds them through vpath; -Wno-portability silences the warning
# about vpath being a GNU make extension.
AUTOMAKE_OPTIONS = -Wno-portability

bin_PROGRAMS = vectorial-openmp

# Farrar (multi-architecture) lives in its own project, shared with the
# standalone benchmark: only its headers and src/Farrar.cpp are used here
FARRAR_PATH = $(top_srcdir)/libs/farrar

# Global Include Paths
AM_CPPFLAGS = \
    -I$(top_srcdir)/src \
    -I$(top_builddir)/src \
    -I$(FARRAR_PATH)/include \
    -I$(top_srcdir)/$(LIBMASA_PATH)/src \
    -I$(top_srcdir)/$(LIBMASA_PATH)/src/libmasa/processors \
    -DFARRAR_WITH_MASA

# Base Compiler & Linker Flags (AM_CXXFLAGS, not per-target flags: per-target
# rules name the source file directly and would not go through vpath)
AM_CXXFLAGS = -std=gnu++17 -fno-strict-aliasing
vectorial_openmp_LDADD    = $(top_builddir)/$(LIBMASA_PATH)/libmasa.a -lpthread -lgomp

# Architecture-Specific Flags (./configure --with-vec-arch=avx|rvv|sve)
if USE_AVX
AM_CXXFLAGS += -mavx2 -malign-double
endif

if USE_RVV
AM_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

if USE_SVE
AM_CXXFLAGS += -march=armv8.2-a+sve
endif

# Source Files (in src/ and libs/farrar/src/)
vpath %.cpp $(top_srcdir)/src $(FARRAR_PATH)/src

vectorial_openmp_SOURCES = \
    main.cpp \
    VectorialOpenMPAligner.cpp \
    VectorialOpenMPAlignerParameters.cpp \
    VectorialCPUBlockProcessor.cpp \
    Farrar.cpp
