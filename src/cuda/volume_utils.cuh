#ifndef CU_VOLUME_UTILS_H
#define CU_VOLUME_UTILS_H

#include "ocean_utils.cuh"
#include "utils.cuh"

#define FOG_DENSITY (0.001f * device.scene.fog.density)

struct VolumeDescriptor {
  // TODO: Correctly pass descriptor to G-Buffer and use in ReSTIR.
  VolumeType type;
  float water_droplet_diameter;
  RGBF absorption;
  RGBF scattering;
  float avg_scattering;
  float dist;
  float max_height;
  float min_height;
} typedef VolumeDescriptor;

__device__ RGBF volume_get_transmittance(const VolumeDescriptor volume) {
  return add_color(volume.absorption, volume.scattering);
}

__device__ VolumeDescriptor volume_get_descriptor_preset_fog() {
  VolumeDescriptor volume;

  volume.type                   = VOLUME_TYPE_FOG;
  volume.water_droplet_diameter = device.scene.fog.droplet_diameter;
  volume.absorption             = get_color(0.0f, 0.0f, 0.0f);
  volume.scattering             = get_color(FOG_DENSITY, FOG_DENSITY, FOG_DENSITY);
  volume.avg_scattering         = FOG_DENSITY;
  volume.dist                   = device.scene.fog.dist;
  volume.max_height             = device.scene.fog.height;
  volume.min_height             = (device.scene.ocean.active) ? OCEAN_MAX_HEIGHT : 0.0f;

  return volume;
}

__device__ VolumeDescriptor volume_get_descriptor_preset_ocean() {
  VolumeDescriptor volume;

  volume.type                   = VOLUME_TYPE_OCEAN;
  volume.water_droplet_diameter = 50.0f;
  volume.absorption             = OCEAN_ABSORPTION;
  volume.scattering             = OCEAN_SCATTERING;
  volume.dist                   = 10000.0f;
  volume.max_height             = OCEAN_MIN_HEIGHT * (1.0f - eps);
  volume.min_height             = 0.0f;

  volume.avg_scattering = RGBF_avg(volume.scattering);

  return volume;
}

__device__ VolumeDescriptor volume_get_descriptor_preset(const VolumeType type) {
  switch (type) {
    case VOLUME_TYPE_FOG:
      return volume_get_descriptor_preset_fog();
    case VOLUME_TYPE_OCEAN:
      return volume_get_descriptor_preset_ocean();
    default:
      return {};
  }
}

#endif /* CU_VOLUME_UTILS_H */
