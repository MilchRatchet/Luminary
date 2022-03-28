#ifndef UI_PANEL_H
#define UI_PANEL_H

#include "SDL.h"
#include "UI_structs.h"

#define PANEL_SLIDER 0x1
#define PANEL_CHECK 0x2
#define PANEL_DROPDOWN 0x3
#define PANEL_COLOR 0x4
#define PANEL_INFO 0x5
#define PANEL_TAB 0x6
#define PANEL_BUTTON 0x7

#define PANEL_HEIGHT 36

#define HOVER_R 0x00
#define HOVER_G 0xff
#define HOVER_B 0xa5

UIPanel create_slider(
  UI* ui, const char* text, void* data_binding, int voids_frames, float scale, float min, float max, int refresh, int integral);
UIPanel create_check(UI* ui, const char* text, int* data_binding, int voids_frames);
UIPanel create_dropdown(UI* ui, const char* text, int* data_binding, int voids_frames, int option_count, char* options, int index);
UIPanel create_color(UI* ui, const char* text, float* data_binding);
UIPanel create_info(UI* ui, const char* text, void* data_binding, int data_type, int kind);
UIPanel create_tab(UI* ui, int* data_binding, char* options);
UIPanel create_button(UI* ui, const char* text, void* data_binding, void (*func)(void*), int voids_frames);
void handle_mouse_UIPanel(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel(UI* ui, UIPanel* panel, int y);
void free_UIPanel(UIPanel* panel);

#endif /* UI_PANEL_H */
