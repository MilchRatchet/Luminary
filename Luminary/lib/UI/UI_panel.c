#include "UI.h"
#include "UI_panel.h"
#include "UI_text.h"
#include "UI_slider.h"
#include "UI_check.h"
#include "UI_color.h"
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

UIPanel create_slider(
  UI* ui, int num, const char* text, float* data_binding, int voids_frames, float scale, float min,
  float max) {
  UIPanel slider = init_UIPanel(ui, num, PANEL_SLIDER, text, data_binding, voids_frames);

  slider.prop1 = *((int*) &scale);
  slider.prop2 = *((int*) &min);
  slider.prop3 = *((int*) &max);

  return slider;
}

UIPanel create_check(UI* ui, int num, const char* text, int* data_binding, int voids_frames) {
  UIPanel check = init_UIPanel(ui, num, PANEL_CHECK, text, data_binding, voids_frames);

  return check;
}

UIPanel create_color(UI* ui, int num, const char* text, float* data_binding) {
  UIPanel check = init_UIPanel(ui, num, PANEL_COLOR, text, data_binding, 0);

  return check;
}

UIPanel create_info(
  UI* ui, int num, const char* text, void* data_binding, int data_type, int kind) {
  UIPanel info = init_UIPanel(ui, num, PANEL_INFO, text, data_binding, 0);

  info.prop1 = data_type;
  info.prop2 = kind;

  return info;
}

UIPanel create_tab(UI* ui, int num, int* data_binding) {
  UIPanel tab;

  tab = init_UIPanel(ui, num, PANEL_TAB, "", data_binding, 0);

  tab.data_text = (SDL_Surface*) malloc(sizeof(SDL_Surface*) * UI_PANELS_TAB_COUNT);

  SDL_Surface** texts = (SDL_Surface**) tab.data_text;

  SDL_Surface* surface;

  surface  = render_text(ui, "General");
  texts[0] = surface;
  surface  = render_text(ui, "Camera");
  texts[1] = surface;
  surface  = render_text(ui, "Sky");
  texts[2] = surface;
  surface  = render_text(ui, "Ocean");
  texts[3] = surface;
  surface  = render_text(ui, "Toy");
  texts[4] = surface;

  int total_width = 0;

  for (int i = 0; i < UI_PANELS_TAB_COUNT; i++) {
    total_width += texts[i]->w;
  }

  tab.prop2 = (UI_WIDTH - total_width - 10) >> 2;

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
    render_UIPanel_color(ui, panel);
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

  if (panel->data_text) {
    if (panel->type == PANEL_TAB) {
      for (int i = 0; i < UI_PANELS_TAB_COUNT; i++) {
        SDL_FreeSurface(((SDL_Surface**) panel->data_text)[i]);
      }

      free(panel->data_text);
    }
    else {
      SDL_FreeSurface(panel->data_text);
    }
  }
}
