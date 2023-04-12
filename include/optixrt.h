#ifndef OPTIXRT_H
#define OPTIXRT_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void optixrt_build_bvh(RaytraceInstance* instance);
void optixrt_compile_kernels(RaytraceInstance* instance);
void optixrt_create_groups(RaytraceInstance* instance);
void optixrt_create_pipeline(RaytraceInstance* instance);
void optixrt_create_shader_bindings(RaytraceInstance* instance);
void optixrt_create_params(RaytraceInstance* instance);
void optixrt_update_params(RaytraceInstance* instance);
void optixrt_execute(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* OPTIXRT_H */
