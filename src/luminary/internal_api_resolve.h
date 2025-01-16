#ifndef LUMINARY_INTERNAL_API_RESOLVE_H
#define LUMINARY_INTERNAL_API_RESOLVE_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

////////////////////////////////////////////////////////////////////
// Rename API definitions into internal definitions
////////////////////////////////////////////////////////////////////

typedef LuminaryVec3 vec3;
typedef LuminaryRGBF RGBF;
typedef LuminaryRGBAF RGBAF;
typedef LuminaryARGB8 ARGB8;

typedef LuminaryRendererSettings RendererSettings;
typedef LuminaryShadingMode ShadingMode;

typedef LuminaryOutputProperties OutputProperties;
typedef LuminaryOutputRequestProperties OutputRequestProperties;

typedef LuminaryCamera Camera;
typedef LuminaryApertureShape ApertureShape;
typedef LuminaryFilter Filter;
typedef LuminaryToneMap ToneMap;

typedef LuminaryOcean Ocean;
typedef LuminaryJerlovWaterType JerlovWaterType;

typedef LuminarySky Sky;
typedef LuminarySkyMode SkyMode;

typedef LuminaryCloud Cloud;
typedef LuminaryCloudLayer CloudLayer;

typedef LuminaryFog Fog;

typedef LuminaryParticles Particles;

typedef LuminaryMaterialBaseSubstrate MaterialBaseSubstrate;
typedef LuminaryMaterial Material;

typedef LuminaryQueue Queue;
typedef LuminaryRingBuffer RingBuffer;

typedef LuminaryHost Host;

typedef LuminaryWallTime WallTime;

typedef LuminaryPath Path;
typedef LuminaryImage Image;

#endif /* LUMINARY_INTERNAL_API_RESOLVE */
