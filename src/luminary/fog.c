#include "fog.h"

#include "internal_error.h"

LuminaryResult fog_get_default(Fog* fog) {
  __CHECK_NULL_ARGUMENT(fog);

  fog->active           = false;
  fog->density          = 1.0f;
  fog->droplet_diameter = 10.0f;
  fog->height           = 500.0f;
  fog->dist             = 500.0f;

  return LUMINARY_SUCCESS;
}
