#ifndef CU_CAMERA_POST_LENS_FLARE_H
#define CU_CAMERA_POST_LENS_FLARE_H

#include "utils.h"

extern "C" void device_lens_flare_init(RaytraceInstance* instance) {
}

static void _lens_flare_apply_ghosts(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst) {
}

extern "C" void device_lens_flare_apply(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst) {
}

extern "C" void device_lens_flare_clear(RaytraceInstance* instance) {
}

#endif /* CU_CAMERA_POST_LENS_FLARE_H */
