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
