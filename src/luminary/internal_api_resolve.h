#ifndef LUMINARY_INTERNAL_API_RESOLVE_H
#define LUMINARY_INTERNAL_API_RESOLVE_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

////////////////////////////////////////////////////////////////////
// Rename API definitions into internal definitions
////////////////////////////////////////////////////////////////////

typedef LuminaryVec3 vec3;
typedef LuminaryRGBF RGBF;

typedef LuminaryCamera Camera;
typedef LuminaryApertureShape ApertureShape;
typedef LuminaryFilter Filter;
typedef LuminaryToneMap ToneMap;

typedef LuminaryOcean Ocean;
typedef LuminaryJerlovWaterType JerlovWaterType;

typedef LuminaryQueue Queue;
typedef LuminaryRingBuffer RingBuffer;

typedef LuminaryHost Host;

#endif /* LUMINARY_INTERNAL_API_RESOLVE */
