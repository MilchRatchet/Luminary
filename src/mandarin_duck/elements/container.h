#ifndef MANDARIN_DUCK_ELEMENTS_CONTAINER_H
#define MANDARIN_DUCK_ELEMENTS_CONTAINER_H

#include "element.h"
#include "elements_common.h"
#include "utils.h"

struct ElementContainerData {
  float target_width;
  float target_height;
  uint32_t padding;
  bool is_horizontal;
  COMPUTED uint32_t width;
  COMPUTED uint32_t height;
  Element** elements;
} typedef ElementContainerData;

struct ElementContainerArgs {
  float target_width;
  float target_height;
  uint32_t padding;
  bool is_horizontal;
} typedef ElementContainerArgs;

void element_container_create(Element** element);
void element_container_set(Element* element, ElementContainerArgs* args);
void element_container_add_element(Element* element, Element* added_element);

#endif /* MANDARIN_DUCK_ELEMENTS_CONTAINER_H */
