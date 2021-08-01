#include "UI.h"
#include "UI_panel.h"
#include "UI_text.h"
#include "UI_slider.h"
#include "UI_check.h"

static UIPanel init_UIPanel(UI* ui, int num, int type, const char* text, void* data_binding) {
  UIPanel panel;

  panel.type  = type;
  panel.hover = 0;
  panel.title = render_text(ui, text);
  panel.data  = data_binding;
  panel.y     = num * PANEL_HEIGHT;

  return panel;
}

UIPanel create_slider(UI* ui, int num, const char* text, float* data_binding) {
  UIPanel slider = init_UIPanel(ui, num, PANEL_SLIDER, text, data_binding);

  return slider;
}

UIPanel create_check(UI* ui, int num, const char* text, int* data_binding) {
  UIPanel check = init_UIPanel(ui, num, PANEL_CHECK, text, data_binding);

  return check;
}

void render_UIPanel(UI* ui, UIPanel* panel) {
  switch (panel->type) {
  case PANEL_SLIDER:
    render_UIPanel_slider(ui, panel);
    break;
  case PANEL_CHECK:
    render_UIPanel_check(ui, panel);
    break;
  case PANEL_DROPDOWN:
    break;
  case PANEL_COLOR:
    break;
  case PANEL_INFO:
    break;
  }
}

void free_UIPanel(UIPanel* panel) {
  SDL_FreeSurface(panel->title);
}
