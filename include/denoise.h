#ifndef DENOISE_H
#define DENOISE_H

#if __cplusplus
extern "C" {
#endif

void denoise_create(RaytraceInstance* instance);
DeviceBuffer* denoise_apply(RaytraceInstance* instance, RGBF* src);
float denoise_auto_exposure(RaytraceInstance* instance);
void denoise_free(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* DENOISE_H */
