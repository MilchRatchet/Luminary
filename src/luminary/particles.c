#include "particles.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult particles_get_default(Particles* particles) {
  __CHECK_NULL_ARGUMENT(particles);

  particles->active             = false;
  particles->scale              = 10.0f;
  particles->albedo.r           = 1.0f;
  particles->albedo.g           = 1.0f;
  particles->albedo.b           = 1.0f;
  particles->direction_altitude = 1.234f;
  particles->direction_azimuth  = 0.0f;
  particles->speed              = 0.0f;
  particles->phase_diameter     = 50.0f;
  particles->seed               = 0;
  particles->count              = 8192;
  particles->size               = 1.0f;
  particles->size_variation     = 0.1f;

  return LUMINARY_SUCCESS;
}

#define __PARTICLES_CHECK_DIRTY(var)                                            \
  {                                                                             \
    if (input->var != old->var) {                                               \
      *dirty_flags = SCENE_DIRTY_FLAG_PARTICLES | SCENE_DIRTY_FLAG_INTEGRATION; \
      return LUMINARY_SUCCESS;                                                  \
    }                                                                           \
  }

LuminaryResult particles_check_for_dirty(const Particles* input, const Particles* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __PARTICLES_CHECK_DIRTY(active);

  if (input->active) {
    __PARTICLES_CHECK_DIRTY(scale);
    __PARTICLES_CHECK_DIRTY(albedo.r);
    __PARTICLES_CHECK_DIRTY(albedo.g);
    __PARTICLES_CHECK_DIRTY(albedo.b);
    __PARTICLES_CHECK_DIRTY(direction_altitude);
    __PARTICLES_CHECK_DIRTY(direction_azimuth);
    __PARTICLES_CHECK_DIRTY(speed);
    __PARTICLES_CHECK_DIRTY(phase_diameter);
    __PARTICLES_CHECK_DIRTY(seed);
    __PARTICLES_CHECK_DIRTY(count);
    __PARTICLES_CHECK_DIRTY(size);
    __PARTICLES_CHECK_DIRTY(size_variation);
  }

  return LUMINARY_SUCCESS;
}
