#ifndef UI_BUTTON_H
#define UI_BUTTON_H

#include "UI_structs.h"

#define BUTTON_SIZE 30

void handle_mouse_UIPanel_button(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel_button(UI* ui, UIPanel* panel, int y);

#endif /* UI_BUTTON_H */
