SOURCES := $(shell find .. -name '*.c' -not -path "*/zlib/*")

compile-gcc:
	gcc -c -I../Luminary/lib -I../Luminary/lib/UI -I../Luminary/lib/SDL -msse4.1 -std=c17 -Wall $(SOURCES)

compile-clang:
	clang -c -I../Luminary/lib -I../Luminary/lib/UI -I../Luminary/lib/SDL -msse4.1 -std=c17 -Wall $(SOURCES)
