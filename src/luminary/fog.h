#ifndef LUMINARY_FOG_H
#define LUMINARY_FOG_H

#include "utils.h"

LuminaryResult fog_get_default(Fog* fog);
LuminaryResult fog_check_for_dirty(const Fog* input, const Fog* old, bool* dirty);

#endif /* LUMINARY_FOG_H */
