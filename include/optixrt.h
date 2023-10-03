#ifndef OPTIXRT_H
#define OPTIXRT_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void optixrt_compile_kernel(RaytraceInstance* instance);
void optixrt_init(RaytraceInstance* instance);
void optixrt_update_params(RaytraceInstance* instance);
void optixrt_execute(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* OPTIXRT_H */
