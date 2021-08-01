#ifndef UI_STRUCTS_H
#define UI_STRUCTS_H

#include "SDL.h"
#include "SDL_ttf.h"

struct UIPanel {
  int type;
  int y;
  int hover;
  void* data;
  SDL_Surface* title;
  SDL_Surface* data_text;
} typedef UIPanel;

struct UI {
  int active;
  int tab;
  int x;
  int y;
  int scroll_pos;
  TTF_Font* font;
  uint8_t* pixels;
  uint8_t* pixels_mask;
  UIPanel* general_panels;
  void* scratch;
} typedef UI;

#endif /* UI_STRUCTS_H */
