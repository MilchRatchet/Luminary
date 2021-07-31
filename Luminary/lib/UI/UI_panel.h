#ifndef UI_PANEL_H
#define UI_PANEL_H

#include "SDL.h"
#include "UI_structs.h"

#define PANEL_SLIDER 0x1
#define PANEL_CHECK 0x2
#define PANEL_DROPDOWN 0x3
#define PANEL_COLOR 0x4

#define PANEL_HEIGHT 40

UIPanel create_slider(UI* ui, const char* text, float* data_binding);
void free_UIPanel(UIPanel* panel);

#endif /* UI_PANEL_H */
