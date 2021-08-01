#ifndef UI_PANEL_H
#define UI_PANEL_H

#include "SDL.h"
#include "UI_structs.h"

#define PANEL_SLIDER 0x1
#define PANEL_CHECK 0x2
#define PANEL_DROPDOWN 0x3
#define PANEL_COLOR 0x4
#define PANEL_INFO 0x5

#define PANEL_HEIGHT 40

UIPanel create_slider(UI* ui, int num, const char* text, float* data_binding);
UIPanel create_check(UI* ui, int num, const char* text, int* data_binding);
UIPanel create_info(UI* ui, int num, const char* text, void* data_binding, int data_type, int kind);
void handle_mouse_UIPanel(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel(UI* ui, UIPanel* panel);
void free_UIPanel(UIPanel* panel);

#endif /* UI_PANEL_H */
