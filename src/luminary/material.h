#ifndef LUMINARY_MATERIAL_H
#define LUMINARY_MATERIAL_H

#include "utils.h"

LuminaryResult material_get_default(Material* material);
LuminaryResult material_check_for_dirty(const Material* input, const Material* old, bool* dirty);

#endif /* LUMINARY_MATERIAL_H */
