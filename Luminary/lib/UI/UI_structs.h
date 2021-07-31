#ifndef UI_STRUCTS_H
#define UI_STRUCTS_H

#include "SDL.h"
#include "SDL_ttf.h"

struct UIPanel {
  int type;
  void* data;
  SDL_Surface* title;
  SDL_Surface* data_text;
} typedef UIPanel;

struct UI {
  int active;
  int tab;
  int x;
  int y;
  TTF_Font* font;
  uint8_t* pixels;
  UIPanel* general_panels;
  float alpha;
  void* scratch;
} typedef UI;

#endif /* UI_STRUCTS_H */
