#ifndef LUMINARY_PARTICLES_H
#define LUMINARY_PARTICLES_H

#include "utils.h"

LuminaryResult particles_get_default(Particles* particles);
LuminaryResult particles_check_for_dirty(const Particles* input, const Particles* old, bool* dirty);

#endif /* LUMINARY_PARTICLES_H */
