#ifndef LUMINARY_SETTINGS_H
#define LUMINARY_SETTINGS_H

#include "utils.h"

LuminaryResult settings_get_default(RendererSettings* settings);
LuminaryResult settings_check_for_dirty(const RendererSettings* input, const RendererSettings* old, uint32_t* dirty_flags);

#endif /* LUMINARY_SETTINGS_H */
