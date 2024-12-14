#ifndef MANDARIN_DUCK_ELEMENTS_COMMON_H
#define MANDARIN_DUCK_ELEMENTS_COMMON_H

struct Window typedef Window;

#include "element.h"
#include "utils.h"
#include "window.h"

struct ElementSize {
  bool is_relative;
  float rel_width;
  float rel_height;
  uint32_t width;
  uint32_t height;
} typedef ElementSize;

enum ElementBindingDataType { ELEMENT_BINDING_DATA_TYPE_FLOAT = 0 } typedef ElementBindingDataType;

struct ElementDataBinding {
  void* ptr;
  ElementBindingDataType data_type;
} typedef ElementDataBinding;

void element_compute_size_and_position(Element* element, WindowContext* context, ElementSize* size);

// TODO: This must be done after element_compute_size_and_position, also this API easily causes duplicate computation, I should combine them
// all and output a struct with all the info.
bool element_is_mouse_hover(Element* element, Display* display);
bool element_is_pressed(Element* element, Display* display);
bool element_is_clicked(Element* element, Display* display);

#endif /* MANDARIN_DUCK_ELEMENTS_COMMON_H */
