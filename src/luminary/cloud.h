#ifndef LUMINARY_CLOUD_H
#define LUMINARY_CLOUD_H

#include "utils.h"

#define CLOUD_DEFAULT_SEED 1

LuminaryResult cloud_get_default(Cloud* cloud);
LuminaryResult cloud_check_for_dirty(const Cloud* input, const Cloud* old, bool* dirty);

#endif /* LUMINARY_CLOUD_H */
