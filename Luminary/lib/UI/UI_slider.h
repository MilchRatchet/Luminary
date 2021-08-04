#ifndef UI_SLIDER_H
#define UI_SLIDER_H

#include "UI_structs.h"

void handle_mouse_UIPanel_slider(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel_slider(UI* ui, UIPanel* panel, int y);

#endif /* UI_SLIDER_H */
