#include "ocean.h"

#include "internal_error.h"

LuminaryResult ocean_get_default(Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(ocean);

  ocean->active                      = false;
  ocean->height                      = 0.0f;
  ocean->amplitude                   = 0.2f;
  ocean->frequency                   = 0.12f;
  ocean->choppyness                  = 4.0f;
  ocean->refractive_index            = 1.333f;
  ocean->water_type                  = LUMINARY_JERLOV_WATER_TYPE_IB;
  ocean->caustics_active             = false;
  ocean->caustics_ris_sample_count   = 32;
  ocean->caustics_domain_scale       = 0.5f;
  ocean->multiscattering             = false;
  ocean->triangle_light_contribution = false;

  return LUMINARY_SUCCESS;
}

#define __OCEAN_DIRTY(var)      \
  {                             \
    if (new->var != old->var) { \
      *dirty = true;            \
      return LUMINARY_SUCCESS;  \
    }                           \
  }

LuminaryResult ocean_check_for_dirty(const Ocean* new, const Ocean* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(new);
  __CHECK_NULL_ARGUMENT(old);

  *dirty = false;

  __OCEAN_DIRTY(active);

  if (new->active) {
    __OCEAN_DIRTY(height);
    __OCEAN_DIRTY(amplitude);
    __OCEAN_DIRTY(frequency);
    __OCEAN_DIRTY(choppyness);
    __OCEAN_DIRTY(refractive_index);
    __OCEAN_DIRTY(water_type);
    __OCEAN_DIRTY(caustics_active);

    if (new->caustics_active) {
      __OCEAN_DIRTY(caustics_ris_sample_count);
      __OCEAN_DIRTY(caustics_domain_scale);
    }

    __OCEAN_DIRTY(multiscattering);
    __OCEAN_DIRTY(triangle_light_contribution);
  }

  return LUMINARY_SUCCESS;
}
