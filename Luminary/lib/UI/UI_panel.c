#include "UI_panel.h"

#include <string.h>

#include "UI.h"
#include "UI_button.h"
#include "UI_check.h"
#include "UI_color.h"
#include "UI_dropdown.h"
#include "UI_info.h"
#include "UI_slider.h"
#include "UI_tab.h"
#include "UI_text.h"

static UIPanel init_UIPanel(UI* ui, int type, const char* text, void* data_binding, int voids_frames) {
  UIPanel panel;

  panel.type         = type;
  panel.hover        = 0;
  panel.title        = render_text(ui, text);
  panel.data_text    = (SDL_Surface*) 0;
  panel.data         = data_binding;
  panel.voids_frames = voids_frames;

  return panel;
}

UIPanel create_slider(
  UI* ui, const char* text, void* data_binding, int voids_frames, float scale, float min, float max, int refresh, int integral) {
  UIPanel slider = init_UIPanel(ui, PANEL_SLIDER, text, data_binding, voids_frames);

  slider.prop1 = *((int*) &scale);
  slider.prop2 = *((int*) &min);
  slider.prop3 = *((int*) &max);
  slider.prop4 = refresh;
  slider.prop5 = integral;

  if (slider.prop5) {
    slider.data_buffer = *((int*) slider.data);
  }

  return slider;
}

UIPanel create_check(UI* ui, const char* text, int* data_binding, int voids_frames) {
  UIPanel check = init_UIPanel(ui, PANEL_CHECK, text, data_binding, voids_frames);

  return check;
}

UIPanel create_dropdown(UI* ui, const char* text, int* data_binding, int voids_frames, int option_count, char* options, int index) {
  UIPanel dropdown = init_UIPanel(ui, PANEL_DROPDOWN, text, data_binding, voids_frames);

  dropdown.prop1 = option_count;
  dropdown.prop2 = 0;
  dropdown.prop3 = index;
  dropdown.prop5 = -1;

  int* offsets               = malloc(sizeof(int) * option_count);
  SDL_Surface** option_texts = malloc(sizeof(SDL_Surface*) * option_count);

  offsets[0] = 0;

  int ptr = 0;

  for (int i = 1; i < option_count; i++) {
    while (options[ptr] != '\0')
      ptr++;

    offsets[i] = ++ptr;
  }

  int max_width = 0;

  for (int i = 0; i < option_count; i++) {
    option_texts[i] = render_text(ui, options + offsets[i]);
    max_width       = max(max_width, option_texts[i]->w);
  }

  dropdown.prop4     = max_width;
  dropdown.data_text = (SDL_Surface*) option_texts;
  free(offsets);

  return dropdown;
}

UIPanel create_color(UI* ui, const char* text, float* data_binding) {
  UIPanel check = init_UIPanel(ui, PANEL_COLOR, text, data_binding, 0);

  return check;
}

UIPanel create_info(UI* ui, const char* text, void* data_binding, int data_type, int kind) {
  UIPanel info = init_UIPanel(ui, PANEL_INFO, text, data_binding, 0);

  info.prop1 = data_type;
  info.prop2 = kind;

  return info;
}

UIPanel create_tab(UI* ui, int* data_binding, char* options) {
  UIPanel tab;

  char* tmp = options;
  char c    = *tmp;
  int count = 1;

  while (c != '\0') {
    c = *(++tmp);
    count += (c == '\n');
  }

  int* starts = malloc(sizeof(int) * (count + 1));

  tmp = options;

  starts[0] = 0;

  int offset = 0;
  int k      = 1;

  c = *tmp;

  while (c != '\0') {
    c = tmp[offset++];
    if (c == '\n')
      starts[k++] = offset;
  }

  starts[count] = offset;

  char** strings = malloc(sizeof(char*) * count);

  for (int i = 0; i < count; i++) {
    const int len = starts[i + 1] - starts[i] - 1;
    strings[i]    = malloc(sizeof(char) * (len + 1));
    memcpy(strings[i], options + starts[i], len);
    strings[i][len] = '\0';
  }

  free(starts);

  tab = init_UIPanel(ui, PANEL_TAB, "", data_binding, 0);

  tab.data_text = (SDL_Surface*) malloc(sizeof(SDL_Surface*) * count);

  SDL_Surface** texts = (SDL_Surface**) tab.data_text;

  for (int i = 0; i < count; i++) {
    texts[i] = render_text(ui, strings[i]);
  }

  int total_width = 0;

  for (int i = 0; i < count; i++) {
    total_width += texts[i]->w;
  }

  tab.prop2 = (UI_WIDTH - total_width) / count;
  tab.prop3 = count;

  tab.voids_frames = 0;

  free(strings);

  return tab;
}

UIPanel create_button(UI* ui, const char* text, void* data_binding, void (*func)(void*), int voids_frames) {
  UIPanel button = init_UIPanel(ui, PANEL_BUTTON, text, data_binding, voids_frames);

  button.func = func;

  return button;
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
      handle_mouse_UIPanel_dropdown(ui, panel, mouse_state, x, y);
      break;
    case PANEL_COLOR:
      break;
    case PANEL_INFO:
      break;
    case PANEL_TAB:
      handle_mouse_UIPanel_tab(ui, panel, mouse_state, x, y);
      break;
    case PANEL_BUTTON:
      handle_mouse_UIPanel_button(ui, panel, mouse_state, x, y);
      break;
  }
}

void render_UIPanel(UI* ui, UIPanel* panel, int y) {
  switch (panel->type) {
    case PANEL_SLIDER:
      render_UIPanel_slider(ui, panel, y);
      break;
    case PANEL_CHECK:
      render_UIPanel_check(ui, panel, y);
      break;
    case PANEL_DROPDOWN:
      render_UIPanel_dropdown(ui, panel, y);
      break;
    case PANEL_COLOR:
      render_UIPanel_color(ui, panel, y);
      break;
    case PANEL_INFO:
      render_UIPanel_info(ui, panel, y);
      break;
    case PANEL_TAB:
      render_UIPanel_tab(ui, panel, y);
      break;
    case PANEL_BUTTON:
      render_UIPanel_button(ui, panel, y);
      break;
  }
}

void free_UIPanel(UIPanel* panel) {
  if (panel->title)
    SDL_FreeSurface(panel->title);

  if (panel->data_text) {
    if (panel->type == PANEL_TAB) {
      for (int i = 0; i < panel->prop3; i++) {
        SDL_FreeSurface(((SDL_Surface**) panel->data_text)[i]);
      }

      free(panel->data_text);
    }
    else if (panel->type == PANEL_DROPDOWN) {
      for (int i = 0; i < panel->prop1; i++) {
        SDL_FreeSurface(((SDL_Surface**) panel->data_text)[i]);
      }

      free(panel->data_text);
    }
    else {
      SDL_FreeSurface(panel->data_text);
    }
  }
}
