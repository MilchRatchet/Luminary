#ifndef DENOISER_H
#define DENOISER_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void denoise_with_optix(raytrace_instance* instance);

#if __cplusplus
}
#endif

#endif /* DENOISER_H */
