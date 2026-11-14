
# libmasa first, then bin/ (vectorial-openmp is built there, so the object
# files stay out of src/)
SUBDIRS = $(LIBMASA_PATH) bin

FARRAR_PATH = libs/farrar

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
    $(FARRAR_PATH)/include/RVV/RvvTraits.hpp \
    $(FARRAR_PATH)/include/SVE/SveOps.hpp \
    $(FARRAR_PATH)/include/SVE/SveTraits.hpp

dist_doc_DATA = README.md

# Keep ./vectorial-openmp working from the top directory
all-local:
	@$(LN_S) -f bin/vectorial-openmp vectorial-openmp

clean-local:
	-rm -f vectorial-openmp

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