#ifndef LUMINARY_LENS_LIBRARY_H
#define LUMINARY_LENS_LIBRARY_H

#include "utils.h"

#define LENS_MAX_NUM_INTERFACES (16)
#define LENS_MAX_NUM_MEDIA (LENS_MAX_NUM_INTERFACES + 1)

struct LensInterface {
  float radius;
  float vertex;
  float cylindrical_radius;
} typedef LensInterface;

struct LensMedium {
  float design_ior;
  float abbe;
  float cylindrical_radius;
} typedef LensMedium;

struct LensTemplateData {
  uint32_t num_interfaces;
  LensInterface interfaces[LENS_MAX_NUM_INTERFACES];
  LensMedium media[LENS_MAX_NUM_MEDIA];
  float design_focal_length;
  float aperture_point;
  float last_vertex;
} typedef LensTemplateData;

LuminaryResult lens_library_get_template_data(LuminaryLensTemplate template, LensTemplateData* data);

#endif /* LUMINARY_LENS_LIBRARY_H */
