#ifndef UI_H
#define UI_H

#include <stdint.h>

#include "SDL_ttf.h"
#include "UI_panel.h"
#include "UI_structs.h"
#include "utils.h"
#include "window_instance.h"

#define UI_HEIGHT_IN_PANELS 20

/*
 * Must be a multiple of 8 and 5
 */
#define UI_WIDTH 320
#define UI_HEIGHT (UI_HEIGHT_IN_PANELS * PANEL_HEIGHT)
#define UI_HEIGHT_BUFFER (UI_HEIGHT + PANEL_HEIGHT)
#define UI_BORDER_SIZE 20

#define UI_PANELS_TAB_COUNT 4

#if !defined(__AVX2__)
#warning Using non AVX2 version of Luminary UI.
#endif

UI init_post_process_UI(RaytraceInstance* instance, WindowInstance* realtime);
UI init_UI(RaytraceInstance* instance, WindowInstance* realtime);
void toggle_UI(UI* ui);
void set_input_events_UI(UI* ui, int mouse_xrel, int mouse_wheel);
void handle_mouse_UI(UI* ui, RaytraceInstance* instance);
void render_UI(UI* ui);
void blit_UI(UI* ui, uint8_t* target, int width, int height, int ld);
void free_UI(UI* ui);

#endif /* UI_H */
