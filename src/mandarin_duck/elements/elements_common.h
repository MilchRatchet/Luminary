#ifndef MANDARIN_DUCK_ELEMENTS_COMMON_H
#define MANDARIN_DUCK_ELEMENTS_COMMON_H

struct Window typedef Window;

#include <SDL3/SDL.h>

#include "element.h"
#include "utils.h"
#include "window.h"

struct ElementSize {
  bool is_relative;
  union {
    struct {
      float rel_width;
      float rel_height;
    };
    struct {
      uint32_t width;
      uint32_t height;
    };
  };
} typedef ElementSize;

enum ElementBindingDataType { ELEMENT_BINDING_DATA_TYPE_FLOAT = 0 } typedef ElementBindingDataType;

struct ElementDataBinding {
  void* ptr;
  ElementBindingDataType data_type;
} typedef ElementDataBinding;

struct ElementMouseResult {
  bool is_hovered;
  bool is_pressed;
  bool is_clicked;
} typedef ElementMouseResult;

void element_apply_context(Element* element, WindowContext* context, ElementSize* size, Display* display, ElementMouseResult* mouse_result);

#endif /* MANDARIN_DUCK_ELEMENTS_COMMON_H */
