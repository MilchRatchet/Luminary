#include "UI.h"
#include "UI_text.h"
#include "UI_tab.h"

void handle_mouse_UIPanel_tab(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
}

void render_UIPanel_tab(UI* ui, UIPanel* panel) {
  blit_text(
    ui, panel->title, 5, ui->scroll_pos + panel->y + ((PANEL_HEIGHT - panel->title->h) >> 1));
  blit_text(
    ui, panel->data_text, UI_WIDTH - 5 - panel->data_text->w,
    ui->scroll_pos + panel->y + ((PANEL_HEIGHT - panel->data_text->h) >> 1));
}
