
#include "toy.h"

#include "internal_error.h"

LuminaryResult toy_get_default(Toy* toy) {
  __CHECK_NULL_ARGUMENT(toy);

  toy->active           = false;
  toy->emissive         = false;
  toy->position.x       = 0.0f;
  toy->position.y       = 10.0f;
  toy->position.z       = 0.0f;
  toy->rotation.x       = 0.0f;
  toy->rotation.y       = 0.0f;
  toy->rotation.z       = 0.0f;
  toy->scale            = 1.0f;
  toy->refractive_index = 1.0f;
  toy->albedo.r         = 0.9f;
  toy->albedo.g         = 0.9f;
  toy->albedo.b         = 0.9f;
  toy->albedo.a         = 1.0f;
  toy->material.r       = 0.3f;
  toy->material.g       = 0.0f;
  toy->material.b       = 1.0f;
  toy->material.a       = 0.0f;
  toy->emission.r       = 0.0f;
  toy->emission.g       = 0.0f;
  toy->emission.b       = 0.0f;

  return LUMINARY_SUCCESS;
}
