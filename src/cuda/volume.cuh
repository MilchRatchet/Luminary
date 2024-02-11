#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "directives.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
#include "restir.cuh"
#include "state.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

//
// This implements a volume renderer. These volumes are homogenous and bound by a disk-box (A horizontal disk with a width in the vertical
// axis). Closed form tracking is used to solve any light interaction with such a volume. While this implementation is mostly generic, some
// details are handled simply with the constraint that Luminary only has two types of volumes, fog and ocean water. Fog does not use
// absorption and only has scalar scattering values while ocean scattering and absorption is performed using three color channels.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [FonWKH17]
// J. Fong, M. Wrenninge, C. Kulla, R. Habel, "Production Volume Rendering", SIGGRAPH 2017 Course, 2017
// https://graphics.pixar.com/library/ProductionVolumeRendering/

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUM_DEVICE_FUNC GBufferData volume_generate_g_buffer(const VolumeTask task, const int pixel) {
  uint32_t flags = (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) ? G_BUFFER_REQUIRES_SAMPLING : 0;
  flags |= G_BUFFER_VOLUME_HIT;

  GBufferData data;
  data.hit_id    = task.hit_id;
  data.albedo    = RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
  data.emission  = get_color(0.0f, 0.0f, 0.0f);
  data.normal    = get_vector(0.0f, 0.0f, 0.0f);
  data.position  = task.position;
  data.V         = scale_vector(task.ray, -1.0f);
  data.roughness = 1.0f;
  data.metallic  = 0.0f;
  data.flags     = flags;

  return data;
}

#endif /* CU_VOLUME_H */
