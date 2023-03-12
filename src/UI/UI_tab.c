#include "UI_tab.h"

#include <stdio.h>

#include "UI.h"
#include "UI_blit.h"
#include "UI_text.h"
#include "utils.h"

void handle_mouse_UIPanel_tab(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  LUM_UNUSED(y)

  panel->hover = 1;

  int offset          = 0;
  SDL_Surface** texts = (SDL_Surface**) panel->data_text;
  panel->prop1        = -1;

  for (int i = 0; i < panel->prop3; i++) {
    if (x >= offset && x < offset + texts[i]->w + panel->prop2) {
      panel->prop1 = i;
      if (SDL_BUTTON_LMASK & mouse_state) {
        memset(ui->tab + panel->prop4 + 1, 0, sizeof(int) * (UI_MAX_TAB_DEPTH - panel->prop4 - 1));
        ui->tab[panel->prop4] = i;
        ui->scroll_pos        = 0;
        panel->hover          = 0;
      }
      break;
    }
    offset += texts[i]->w + panel->prop2;
  }
}

void render_UIPanel_tab(UI* ui, UIPanel* panel, int y) {
  int offset          = 0;
  SDL_Surface** texts = (SDL_Surface**) panel->data_text;
  int tab             = ui->tab[panel->prop4];

  for (int i = 0; i < panel->prop3; i++) {
    blit_text(ui, texts[i], (panel->prop2 >> 1) + offset, ((PANEL_HEIGHT - texts[i]->h) >> 1) + y, UI_WIDTH, UI_HEIGHT_BUFFER);

    offset += texts[i]->w + panel->prop2;
  }

  if (panel->hover && panel->prop1 != tab && panel->prop1 != -1) {
    int x = panel->prop2 >> 1;
    for (int i = 0; i < panel->prop1; i++) {
      x += texts[i]->w + panel->prop2;
    }
    blit_color_shaded(ui->pixels, x, y, UI_WIDTH, UI_HEIGHT_BUFFER, texts[panel->prop1]->w, PANEL_HEIGHT, HOVER_R, HOVER_G, HOVER_B);
  }

  {
    int x = panel->prop2 >> 1;
    for (int i = 0; i < tab; i++) {
      x += texts[i]->w + panel->prop2;
    }
    blit_gray(ui->pixels_mask, x - 5, PANEL_HEIGHT - 2 + y, UI_WIDTH, UI_HEIGHT_BUFFER, texts[tab]->w + 10, 2, 0xff);
    blit_color(ui->pixels, x - 5, PANEL_HEIGHT - 2 + y, UI_WIDTH, UI_HEIGHT_BUFFER, texts[tab]->w + 10, 2, HOVER_R, HOVER_G, HOVER_B);
    blit_color_shaded(ui->pixels, x, y, UI_WIDTH, UI_HEIGHT_BUFFER, texts[tab]->w, PANEL_HEIGHT, HOVER_R, HOVER_G, HOVER_B);
  }
}
