#ifndef UI_CHECK_H
#define UI_CHECK_H

#include "UI_structs.h"

#define PANEL_CHECK_BOX_SIZE 28
#define PANEL_CHECK_BOX_BORDER 1

void handle_mouse_UIPanel_check(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel_check(UI* ui, UIPanel* panel, int y);

#endif /* UI_CHECK_H */
