#ifndef LUMINARY_INTERNAL_API_RESOLVE_H
#define LUMINARY_INTERNAL_API_RESOLVE_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

////////////////////////////////////////////////////////////////////
// Rename API definitions into internal definitions
////////////////////////////////////////////////////////////////////

typedef LuminaryVec3 vec3;
typedef LuminaryRGBF RGBF;

typedef LuminaryRendererSettings RendererSettings;
typedef LuminaryShadingMode ShadingMode;

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

typedef LuminaryToy Toy;

typedef LuminaryMaterial Material;
typedef LuminaryMaterialFlags MaterialFlags;

typedef LuminaryQueue Queue;
typedef LuminaryRingBuffer RingBuffer;

typedef LuminaryHost Host;

typedef LuminaryWallTime WallTime;

typedef LuminaryPath Path;

#endif /* LUMINARY_INTERNAL_API_RESOLVE */
