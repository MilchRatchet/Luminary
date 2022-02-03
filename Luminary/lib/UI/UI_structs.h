#ifndef UI_STRUCTS_H
#define UI_STRUCTS_H

#include "SDL.h"
#include "SDL_ttf.h"

struct UIPanel {
  int type;
  int hover;
  int prop1;
  int prop2;
  int prop3;
  int prop4;
  int prop5;
  float data_buffer;
  int voids_frames;
  void (*func)(void*);
  void* data;
  SDL_Surface* title;
  SDL_Surface* data_text;
} typedef UIPanel;

struct UITab {
  int count;
  struct UITab* subtabs;
  int panel_count;
  UIPanel* panels;
} typedef UITab;

struct UI {
  int active;
  int tab;
  int subtab;
  int border_hover;
  int panel_hover;
  UIPanel* last_panel;
  UIPanel* dropdown;
  int x;
  int y;
  int max_x;
  int max_y;
  int mouse_flags;
  int scroll_pos;
  int mouse_xrel;
  int mouse_wheel;
  TTF_Font* font;
  uint8_t* pixels;
  uint8_t* pixels_mask;
  UITab* tabs;
  int* temporal_frames;
  void* scratch;
} typedef UI;

#endif /* UI_STRUCTS_H */
