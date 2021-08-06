#ifndef UI_DROPDOWN_H
#define UI_DROPDOWN_H

#include "UI_structs.h"

/*
 * prop1: Number of options
 * prop2: Dropdown open
 * prop3: Index in list of panels
 * prop4: Maximum width of the options
 * prop5: Hover of dropdown item
 */

#define PANEL_DROPDOWN_BOX_HEIGHT 32

void handle_mouse_UIPanel_dropdown(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
int get_intersection_dropdown(UI* ui, UIPanel* panel, int x, int y);
void render_UIPanel_dropdown(UI* ui, UIPanel* panel, int y);
void render_dropdown(UI* ui, UIPanel* panel, int offset);

#endif /* UI_DROPDOWN_H */
