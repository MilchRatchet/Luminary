#include "UI.h"
#include "UI_panel.h"
#include "UI_text.h"

static UIPanel init_UIPanel(UI* ui, int type, const char* text, void* data_binding) {
  UIPanel panel;

  panel.type  = type;
  panel.title = render_text(ui, text);
  panel.data  = data_binding;

  return panel;
}

UIPanel create_slider(UI* ui, const char* text, float* data_binding) {
  UIPanel slider = init_UIPanel(ui, PANEL_SLIDER, text, data_binding);

  return slider;
}

void free_UIPanel(UIPanel* panel) {
  SDL_FreeSurface(panel->title);
}
