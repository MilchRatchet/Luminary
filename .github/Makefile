SOURCES := $(shell find .. -name '*.c' -not -path "*/zlib/*")
OBJS := $(patsubst %.c, %.o, $(SOURCES))

CONFIG_EXAMPLE_TEXT := \#ifndef CONFIG_H\n\#define CONFIG_H\n\#define LUMINARY_COMPILER "" \n\#define LUMINARY_OS "" \n\#define LUMINARY_VERSION_DATE "" \n\#define LUMINARY_BRANCH_NAME "" \n\#define LUMINARY_CUDA_VERSION "" \n\#define LUMINARY_OPTIX_VERSION "" \n\#define LUMINARY_VERSION_HASH 0 \n\#endif

compile-gcc:
	echo "$(CONFIG_EXAMPLE_TEXT)" > ../include/config.h
	gcc -c -I../include -I../include/UI -I/usr/include/SDL2 -lSDL2 -lSDL2_ttf -march=skylake -O3 -std=c11 -fPIC -Wall -Wno-unused-result -Wno-format -Wno-maybe-uninitialized -Wno-incompatible-pointer-types -Wno-strict-aliasing $(SOURCES)

compile-clang:
	echo "$(CONFIG_EXAMPLE_TEXT)" > ../include/config.h
	clang -c -I../include -I../include/UI -I/usr/include/SDL2 -lSDL2 -lSDL2_ttf -march=skylake -O3 -std=c11 -fPIC -Wall -Wno-unused-result -Wno-format -Wno-maybe-uninitialized -Wno-incompatible-pointer-types -Wno-strict-aliasing $(SOURCES)
