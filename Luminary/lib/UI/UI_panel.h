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

#define PANEL_HEIGHT 40

#define HOVER_R 0xff
#define HOVER_G 0xa5
#define HOVER_B 0x00

UIPanel create_slider(
  UI* ui, int num, const char* text, float* data_binding, int voids_frames, float scale, float min,
  float max);
UIPanel create_check(UI* ui, int num, const char* text, int* data_binding, int voids_frames);
UIPanel create_color(UI* ui, int num, const char* text, float* data_binding);
UIPanel create_info(UI* ui, int num, const char* text, void* data_binding, int data_type, int kind);
UIPanel create_tab(UI* ui, int num, int* data_bindin);
void handle_mouse_UIPanel(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel(UI* ui, UIPanel* panel);
void free_UIPanel(UIPanel* panel);

#endif /* UI_PANEL_H */
