#ifndef LUMINARY_STRUCT_INTERLEAVING_H
#define LUMINARY_STRUCT_INTERLEAVING_H

#include "utils.h"

void struct_triangles_interleave(Triangle* dst, Triangle* src, uint32_t count);
void struct_triangles_deinterleave(Triangle* dst, Triangle* src, uint32_t count);

#endif /* LUMINARY_STRUCT_INTERLEAVING_H */
