#ifndef OPTIXRT_H
#define OPTIXRT_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void optixrt_compile_kernel(const OptixDeviceContext optix_ctx, const char* kernels_name, OptixKernel* kernel);
void optixrt_init(RaytraceInstance* instance);
void optixrt_update_params(OptixKernel kernel);
void optixrt_execute(OptixKernel kernel);

#if __cplusplus
}
#endif

#endif /* OPTIXRT_H */
