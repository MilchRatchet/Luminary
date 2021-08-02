#include "UI.h"
#include "UI_panel.h"
#include "UI_text.h"
#include "UI_slider.h"
#include "UI_check.h"
#include "UI_info.h"
#include "UI_tab.h"

static UIPanel init_UIPanel(
  UI* ui, int num, int type, const char* text, void* data_binding, int voids_frames) {
  UIPanel panel;

  panel.type         = type;
  panel.hover        = 0;
  panel.title        = render_text(ui, text);
  panel.data_text    = (SDL_Surface*) 0;
  panel.data         = data_binding;
  panel.y            = num * PANEL_HEIGHT;
  panel.voids_frames = voids_frames;

  return panel;
}

UIPanel create_slider(UI* ui, int num, const char* text, float* data_binding, int voids_frames) {
  UIPanel slider = init_UIPanel(ui, num, PANEL_SLIDER, text, data_binding, voids_frames);

  return slider;
}

UIPanel create_check(UI* ui, int num, const char* text, int* data_binding, int voids_frames) {
  UIPanel check = init_UIPanel(ui, num, PANEL_CHECK, text, data_binding, voids_frames);

  return check;
}

UIPanel create_info(
  UI* ui, int num, const char* text, void* data_binding, int data_type, int kind) {
  UIPanel info = init_UIPanel(ui, num, PANEL_INFO, text, data_binding, 0);

  info.prop1 = data_type;
  info.prop2 = kind;

  return info;
}

UIPanel create_tab(UI* ui, int num, int* data_binding, int kind) {
  UIPanel tab;

  switch (kind) {
  case UI_PANELS_GENERAL_TAB:
    tab           = init_UIPanel(ui, num, PANEL_TAB, "General", data_binding, 0);
    tab.data_text = render_text(ui, "C|O|A|T");
    break;
  }

  tab.prop1        = kind;
  tab.voids_frames = 0;

  return tab;
}

void handle_mouse_UIPanel(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  switch (panel->type) {
  case PANEL_SLIDER:
    handle_mouse_UIPanel_slider(ui, panel, mouse_state, x, y);
    break;
  case PANEL_CHECK:
    handle_mouse_UIPanel_check(ui, panel, mouse_state, x, y);
    break;
  case PANEL_DROPDOWN:
    break;
  case PANEL_COLOR:
    break;
  case PANEL_INFO:
    break;
  case PANEL_TAB:
    handle_mouse_UIPanel_tab(ui, panel, mouse_state, x, y);
    break;
  }
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
    render_UIPanel_info(ui, panel);
    break;
  case PANEL_TAB:
    render_UIPanel_tab(ui, panel);
    break;
  }
}

void free_UIPanel(UIPanel* panel) {
  if (panel->title)
    SDL_FreeSurface(panel->title);

  if (panel->data_text)
    SDL_FreeSurface(panel->data_text);
}
