#include "UI_text.h"
#include "UI_slider.h"

void handle_mouse_UIPanel_slider(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
}

void render_UIPanel_slider(UI* ui, UIPanel* panel) {
  blit_text(
    ui, panel->title, 5, ui->scroll_pos + panel->y + ((PANEL_HEIGHT - panel->title->h) >> 1));
}
