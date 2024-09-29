
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

#define __TOY_DIRTY(var)        \
  {                             \
    if (new->var != old->var) { \
      *dirty = true;            \
      return LUMINARY_SUCCESS;  \
    }                           \
  }

LuminaryResult toy_check_for_dirty(const Toy* new, const Toy* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(new);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __TOY_DIRTY(active);

  if (new->active) {
    __TOY_DIRTY(emissive);
    __TOY_DIRTY(position.x);
    __TOY_DIRTY(position.y);
    __TOY_DIRTY(position.z);
    __TOY_DIRTY(rotation.x);
    __TOY_DIRTY(rotation.y);
    __TOY_DIRTY(rotation.z);
    __TOY_DIRTY(scale);
    __TOY_DIRTY(albedo.r);
    __TOY_DIRTY(albedo.g);
    __TOY_DIRTY(albedo.b);
    __TOY_DIRTY(albedo.a);

    if (new->albedo.a < 1.0f) {
      __TOY_DIRTY(refractive_index);
    }

    __TOY_DIRTY(material.r);
    __TOY_DIRTY(material.g);
    __TOY_DIRTY(material.b);
    __TOY_DIRTY(material.a);

    if (new->emissive) {
      __TOY_DIRTY(emission.r);
      __TOY_DIRTY(emission.g);
      __TOY_DIRTY(emission.b);
    }
  }

  return LUMINARY_SUCCESS;
}
