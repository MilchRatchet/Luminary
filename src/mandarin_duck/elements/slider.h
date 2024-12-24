#ifndef MANDARIN_DUCK_ELEMENTS_SLIDER_H
#define MANDARIN_DUCK_ELEMENTS_SLIDER_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

enum ElementSliderDataType {
  ELEMENT_SLIDER_DATA_TYPE_FLOAT,
  ELEMENT_SLIDER_DATA_TYPE_UINT,
  ELEMENT_SLIDER_DATA_TYPE_VECTOR,
  ELEMENT_SLIDER_DATA_TYPE_RGB
} typedef ElementSliderDataType;

struct ElementSliderData {
  ElementSliderDataType type;
  float data_float;
  uint32_t data_uint;
  LuminaryVec3 data_vec3;
  ElementSize size;
  uint32_t color;
  uint32_t component_padding;
  uint32_t margins;
  bool center_x;
  bool center_y;
} typedef ElementSliderData;
static_assert(sizeof(ElementSliderData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementSliderArgs {
  ElementSliderDataType type;
  void* data_binding;
  float max;
  float min;
  ElementSize size;
  uint32_t color;
  uint32_t component_padding;
  uint32_t margins;
  bool center_x;
  bool center_y;
} typedef ElementSliderArgs;

bool element_slider(Window* window, Display* display, ElementSliderArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_SLIDER_H */
