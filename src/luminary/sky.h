#ifndef LUMINARY_SKY_H
#define LUMINARY_SKY_H

#include "utils.h"

LuminaryResult sky_get_default(Sky* sky);
LuminaryResult sky_check_for_dirty(const Sky* input, const Sky* old, bool* passive_dirty, bool* integration_dirty, bool* hdri_dirty);

#endif /* LUMINARY_SKY_H */
