#ifndef STRUCT_INTERLEAVING_H
#define STRUCT_INTERLEAVING_H

#include "structs.h"

void struct_triangles_interleave(Triangle* dst, Triangle* src, uint32_t count);
void struct_triangles_deinterleave(Triangle* dst, Triangle* src, uint32_t count);

#endif /* STRUCT_INTERLEAVING_H */
