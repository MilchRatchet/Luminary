#ifndef MANDARIN_DUCK_ELEMENTS_COMMON_H
#define MANDARIN_DUCK_ELEMENTS_COMMON_H

struct Window typedef Window;

#include <SDL3/SDL.h>

#include "element.h"
#include "utils.h"
#include "window.h"

#define ELEMENT_SIZE_INVALID 0xFFFFFFFF

struct ElementSize {
  float rel_width;
  uint32_t width;
  float rel_height;
  uint32_t height;
} typedef ElementSize;

enum ElementBindingDataType { ELEMENT_BINDING_DATA_TYPE_FLOAT = 0 } typedef ElementBindingDataType;

struct ElementDataBinding {
  void* ptr;
  ElementBindingDataType data_type;
} typedef ElementDataBinding;

struct ElementMouseResult {
  bool is_hovered;
  bool is_down;
  bool is_pressed;
  bool is_clicked;
  float click_rel_x;
  float click_rel_y;
} typedef ElementMouseResult;

void element_apply_context(
  Element* element, WindowContext* context, ElementSize* size, const MouseState* mouse_state, ElementMouseResult* mouse_result);
uint64_t element_compute_hash(const char* context_indentifier, const char* identifier);

#endif /* MANDARIN_DUCK_ELEMENTS_COMMON_H */
