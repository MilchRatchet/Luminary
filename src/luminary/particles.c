#include "particles.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult particles_get_default(Particles* particles) {
  __CHECK_NULL_ARGUMENT(particles);

  particles->active             = false;
  particles->tile_size          = 10.0f;
  particles->type               = LUMINARY_PARTICLE_TYPE_SNOW;
  particles->fall_speed         = 1.8f;
  particles->wind_speed         = 10.0f;
  particles->wind_direction     = 0.0f;
  particles->phase_diameter     = 50.0f;
  particles->seed               = 0;
  particles->count              = 8192;
  particles->diameter           = 10.0f;
  particles->diameter_variation = 8.0f;

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
    __PARTICLES_CHECK_DIRTY(tile_size);
    __PARTICLES_CHECK_DIRTY(type);
    __PARTICLES_CHECK_DIRTY(fall_speed);
    __PARTICLES_CHECK_DIRTY(wind_speed);
    __PARTICLES_CHECK_DIRTY(wind_direction);
    __PARTICLES_CHECK_DIRTY(phase_diameter);
    __PARTICLES_CHECK_DIRTY(seed);
    __PARTICLES_CHECK_DIRTY(count);
    __PARTICLES_CHECK_DIRTY(diameter);
    __PARTICLES_CHECK_DIRTY(diameter_variation);
  }

  return LUMINARY_SUCCESS;
}
