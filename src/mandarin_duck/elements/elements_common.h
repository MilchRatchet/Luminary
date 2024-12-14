#ifndef MANDARIN_DUCK_ELEMENTS_COMMON_H
#define MANDARIN_DUCK_ELEMENTS_COMMON_H

#include "utils.h"

enum ElementDataBindingType { ELEMENT_DATA_BINDING_TYPE_SETTINGS = 0 } typedef ElementDataBindingType;
enum ElementDataBindingDataType { ELEMENT_DATA_BINDING_DATA_TYPE_FLOAT = 0 } typedef ElementDataBindingDataType;

#define COMPUTED

struct ElementDataBinding {
  ElementDataBindingType type;
  size_t offset;
  ElementDataBindingDataType data_type;
} typedef ElementDataBinding;

#endif /* MANDARIN_DUCK_ELEMENTS_COMMON_H */
