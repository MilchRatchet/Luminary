#ifndef UI_TAB_H
#define UI_TAB_H

#include "UI_structs.h"

/*
 * prop1: Which options is hovered
 * prop2: Gap between options
 * prop3: Option Count
 *
 * prop1 is -1 if no option is hovered.
 * prop2 is halved for the gap to the left and right border.
 */

void handle_mouse_UIPanel_tab(UI* ui, UIPanel* panel, int mouse_state, int x, int y);
void render_UIPanel_tab(UI* ui, UIPanel* panel, int y);

#endif /* UI_TAB_H */
