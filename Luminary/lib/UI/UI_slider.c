#include "UI_text.h"
#include "UI_slider.h"

void render_UIPanel_slider(UI* ui, UIPanel* panel) {
  blit_text(ui, panel->title, 5, ui->scroll_pos + panel->y + 10);
}
