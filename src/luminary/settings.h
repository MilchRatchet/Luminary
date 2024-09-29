#ifndef LUMINARY_SETTINGS_H
#define LUMINARY_SETTINGS_H

#include "utils.h"

LuminaryResult settings_get_default(RendererSettings* settings);
LuminaryResult settings_check_for_dirty(const RendererSettings* new, const RendererSettings* old, bool* dirty);

#endif /* LUMINARY_SETTINGS_H */
