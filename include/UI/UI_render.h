#ifndef UI_RENDER_H
#define UI_RENDER_H

#include "UI_structs.h"

#if __cplusplus
extern "C" {
#endif

void ui_render_update_context(UI* ui);
void ui_render_internal(UI* ui, void* dst, int width, int height, int ld);

#if __cplusplus
}
#endif

#endif /* UI_RENDER_H */
