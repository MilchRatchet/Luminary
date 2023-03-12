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
  panel.render       = (void (*)(UI*, struct UIPanel*, int)) 0;
  panel.handle_mouse = (void (*)(UI*, UIPanel*, int, int, int)) 0;

  return panel;
}

UIPanel create_slider(
  UI* ui, const char* text, void* data_binding, int voids_frames, float scale, float min, float max, int refresh, int integral) {
  UIPanel slider = init_UIPanel(ui, PANEL_SLIDER, text, data_binding, voids_frames);

  slider.prop1        = *((int*) &scale);
  slider.prop2        = *((int*) &min);
  slider.prop3        = *((int*) &max);
  slider.prop4        = refresh;
  slider.prop5        = integral;
  slider.render       = render_UIPanel_slider;
  slider.handle_mouse = handle_mouse_UIPanel_slider;

  if (slider.prop5) {
    slider.data_buffer = *((int*) slider.data);
  }

  return slider;
}

UIPanel create_check(UI* ui, const char* text, int* data_binding, int voids_frames) {
  UIPanel check = init_UIPanel(ui, PANEL_CHECK, text, data_binding, voids_frames);

  check.render       = render_UIPanel_check;
  check.handle_mouse = handle_mouse_UIPanel_check;

  return check;
}

UIPanel create_dropdown(UI* ui, const char* text, int* data_binding, int voids_frames, int option_count, char* options, int index) {
  UIPanel dropdown = init_UIPanel(ui, PANEL_DROPDOWN, text, data_binding, voids_frames);

  dropdown.prop1        = option_count;
  dropdown.prop2        = 0;
  dropdown.prop3        = index;
  dropdown.prop5        = -1;
  dropdown.render       = render_UIPanel_dropdown;
  dropdown.handle_mouse = handle_mouse_UIPanel_dropdown;

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
  UIPanel color = init_UIPanel(ui, PANEL_COLOR, text, data_binding, 0);

  color.render = render_UIPanel_color;

  return color;
}

UIPanel create_info(UI* ui, const char* text, void* data_binding, int data_type, int kind) {
  UIPanel info = init_UIPanel(ui, PANEL_INFO, text, data_binding, 0);

  info.prop1  = data_type;
  info.prop2  = kind;
  info.render = render_UIPanel_info;

  return info;
}

UIPanel create_tab(UI* ui, int depth, char* options) {
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

  tab = init_UIPanel(ui, PANEL_TAB, "", (void*) 0, 0);

  tab.data_text = (SDL_Surface*) malloc(sizeof(SDL_Surface*) * count);

  SDL_Surface** texts = (SDL_Surface**) tab.data_text;

  for (int i = 0; i < count; i++) {
    texts[i] = render_text(ui, strings[i]);
  }

  int total_width = 0;

  for (int i = 0; i < count; i++) {
    total_width += texts[i]->w;
  }

  tab.prop2        = (UI_WIDTH - total_width) / count;
  tab.prop3        = count;
  tab.prop4        = depth;
  tab.render       = render_UIPanel_tab;
  tab.handle_mouse = handle_mouse_UIPanel_tab;

  tab.voids_frames = 0;

  free(strings);

  return tab;
}

UIPanel create_button(UI* ui, const char* text, void* data_binding, void (*func)(void*), int voids_frames) {
  UIPanel button = init_UIPanel(ui, PANEL_BUTTON, text, data_binding, voids_frames);

  button.func         = func;
  button.render       = render_UIPanel_button;
  button.handle_mouse = handle_mouse_UIPanel_button;

  return button;
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
