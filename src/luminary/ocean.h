#ifndef LUMINARY_OCEAN_H
#define LUMINARY_OCEAN_H

#include "utils.h"

LuminaryResult ocean_get_default(Ocean* ocean);
LuminaryResult ocean_check_for_dirty(const Ocean* input, const Ocean* old, uint32_t* dirty_flags);

#endif /* LUMINARY_OCEAN_H */
