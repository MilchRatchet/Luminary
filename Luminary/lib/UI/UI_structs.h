#ifndef UI_STRUCTS_H
#define UI_STRUCTS_H

#include "SDL.h"
#include "SDL_ttf.h"

struct UIPanel {
  int type;
  int y;
  int hover;
  int prop1;
  int prop2;
  int prop3;
  int voids_frames;
  void* data;
  SDL_Surface* title;
  SDL_Surface* data_text;
} typedef UIPanel;

struct UI {
  int active;
  int tab;
  int border_hover;
  int panel_hover;
  UIPanel* last_panel;
  int x;
  int y;
  int max_x;
  int max_y;
  int mouse_flags;
  int scroll_pos;
  TTF_Font* font;
  uint8_t* pixels;
  uint8_t* pixels_mask;
  UIPanel* general_panels;
  int* temporal_frames;
  void* scratch;
} typedef UI;

#endif /* UI_STRUCTS_H */
