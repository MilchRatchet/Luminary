#ifndef CU_OCEAN_UTILS_H
#define CU_OCEAN_UTILS_H

#include "utils.cuh"

#define OCEAN_POLLUTION (device.scene.ocean.pollution * 0.25f)
#define OCEAN_SCATTERING (scale_color(device.scene.ocean.scattering, OCEAN_POLLUTION))
#define OCEAN_ABSORPTION (scale_color(device.scene.ocean.absorption, device.scene.ocean.absorption_strength * 0.25f))
#define OCEAN_EXTINCTION (add_color(OCEAN_SCATTERING, OCEAN_ABSORPTION))

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 2.66f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)

#endif /* CU_OCEAN_UTILS_H */
