AC_PREREQ([2.69])
AC_INIT([MASA-OpenMP], [1.0.1.1024], [edans.sandes@gmail.com], [masa-openmp], [https://github.com/edanssandes/MASA-OpenMP])

AC_CONFIG_AUX_DIR([admin])
AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([Makefile])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

# Enable C++ compiler checks
AC_PROG_CXX
AC_PROG_CC

# MASA-Core Subdirectory Setup
AC_CONFIG_SUBDIRS([libs/masa-core])
LIBMASA_PATH="libs/masa-core"

if test ! -d "$srcdir/$LIBMASA_PATH/src"; then
    AC_MSG_ERROR([MASA-Core source directory was not found at $LIBMASA_PATH/src.])
fi

# Checks for header files
AC_CHECK_HEADERS([omp.h], [], [
    AC_MSG_ERROR([Missing required header file <omp.h>.])
])

# Debug vs Optimization flags
OPTIMIZATION="-O3"
DEBUG=""
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [enable GDB debug build])],
    [DEBUG="-ggdb"; OPTIMIZATION="-O0"]
)

# Vector Architecture Selection
AC_ARG_WITH([vec-arch],
    [AS_HELP_STRING([--with-vec-arch=ARCH], [select vector architecture: avx or rvv [default=avx]])],
    [VEC_ARCH="$withval"],
    [VEC_ARCH="avx"]
)

case "$VEC_ARCH" in
    avx)
        AC_MSG_NOTICE([Vector architecture: AVX2])
        AM_CONDITIONAL([USE_AVX], [true])
        AM_CONDITIONAL([USE_RVV], [false])
        AC_DEFINE([USE_AVX], [1], [Use AVX backend])
        
        # Enable 32/64 bit toggle specifically for x86
        AC_ARG_ENABLE([32bit],
            [AS_HELP_STRING([--enable-32bit], [force compilation in 32-bit mode for x86])],
            [PLATFORM="-m32"],
            [PLATFORM="-m64"]
        )
        ;;

    rvv)
        AC_MSG_NOTICE([Vector architecture: RISC-V Vector (RVV)])
        AM_CONDITIONAL([USE_AVX], [false])
        AM_CONDITIONAL([USE_RVV], [true])
        AC_DEFINE([USE_RVV], [1], [Use RVV backend])
        
        # Do NOT pass -m64 to RVV cross-compilers
        PLATFORM=""
        ;;

    *)
        AC_MSG_ERROR([
            Invalid vector architecture '$VEC_ARCH'.
            Use --with-vec-arch=avx or --with-vec-arch=rvv.
        ])
        ;;
esac

# Check OpenMP Support
m4_include([m4/ax_check_compile_flag.m4])
AX_CHECK_COMPILE_FLAG([-fopenmp], [OPENMP_CXXFLAGS="-fopenmp"],
    [AC_MSG_ERROR([Compiler does not support OpenMP (-fopenmp).])]
)

# Consolidate flags
CXXFLAGS="$CXXFLAGS -DUNIX $OPENMP_CXXFLAGS $PLATFORM $OPTIMIZATION $DEBUG -I\$(top_srcdir)/$LIBMASA_PATH/src"

AC_SUBST([LIBMASA_PATH])
AC_SUBST([CXXFLAGS])

AC_OUTPUT