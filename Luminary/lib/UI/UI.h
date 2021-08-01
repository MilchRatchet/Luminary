#ifndef UI_H
#define UI_H

#include <stdint.h>
#include "SDL_ttf.h"
#include "UI_panel.h"
#include "UI_structs.h"
#include "utils.h"

/*
 * Must be a multiple of 32
 */
#define UI_WIDTH 320
#define UI_HEIGHT 600

#define UI_PANELS_GENERAL_TAB 0x1
#define UI_PANELS_GENERAL_COUNT 3

#if !defined(__AVX2__)
#warning Using non AVX2 version of Luminary UI.
#endif

UI init_UI(RaytraceInstance* instance);
void toggle_UI(UI* ui);
void render_UI(UI* ui);
void blit_UI(UI* ui, uint8_t* target, int width, int height);
void free_UI(UI* ui);

#endif /* UI_H */
