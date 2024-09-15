#ifndef OPTIXRT_PARTICLE_H
#define OPTIXRT_PARTICLE_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void optixrt_particle_init(RaytraceInstance* instance);
void optixrt_particle_clear(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* OPTIXRT_PARTICLE_H */
