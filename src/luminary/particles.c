#include "particles.h"

#include "internal_error.h"

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
