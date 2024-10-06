#ifndef LUMINARY_TOY_H
#define LUMINARY_TOY_H

#include "utils.h"

LuminaryResult toy_get_default(Toy* toy);
LuminaryResult toy_check_for_dirty(const Toy* input, const Toy* old, bool* dirty);

#endif /* LUMINARY_TOY_H */
