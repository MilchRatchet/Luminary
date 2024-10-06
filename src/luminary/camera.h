#ifndef LUMINARY_CAMERA_H
#define LUMINARY_CAMERA_H

#include "utils.h"

LuminaryResult camera_get_default(Camera* camera);
LuminaryResult camera_check_for_dirty(const Camera* input, const Camera* old, bool* output_dirty, bool* integration_dirty);

#endif /* LUMINARY_CAMERA_H */
