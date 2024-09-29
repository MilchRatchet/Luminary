#ifndef LUMINARY_OCEAN_H
#define LUMINARY_OCEAN_H

#include "utils.h"

LuminaryResult ocean_get_default(Ocean* ocean);
LuminaryResult ocean_check_for_dirty(const Ocean* new, const Ocean* old, bool* dirty);

#endif /* LUMINARY_OCEAN_H */
