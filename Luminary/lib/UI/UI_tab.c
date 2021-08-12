#include "UI_tab.h"

#include "UI.h"
#include "UI_blit.h"
#include "UI_text.h"

#define TAB_R 0x00
#define TAB_G 0xff
#define TAB_B 0xa5

void handle_mouse_UIPanel_tab(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  panel->hover = 1;

  int offset          = 0;
  SDL_Surface** texts = (SDL_Surface**) panel->data_text;
  panel->prop1        = 0;

  for (int i = 0; i < UI_PANELS_TAB_COUNT; i++) {
    if (x >= offset && x < offset + texts[i]->w + panel->prop2) {
      panel->prop1 = i;
      if (SDL_BUTTON_LMASK & mouse_state) {
        ui->tab        = i;
        ui->scroll_pos = 0;
        panel->hover   = 0;
      }
      break;
    }
    offset += texts[i]->w + panel->prop2;
  }
}

void render_UIPanel_tab(UI* ui, UIPanel* panel) {
  int offset          = 0;
  SDL_Surface** texts = (SDL_Surface**) panel->data_text;

  for (int i = 0; i < UI_PANELS_TAB_COUNT; i++) {
    blit_text(ui, texts[i], 5 + offset, ((PANEL_HEIGHT - texts[i]->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

    offset += texts[i]->w + panel->prop2;
  }

  if (panel->hover && panel->prop1 != ui->tab) {
    int x = 5;
    for (int i = 0; i < panel->prop1; i++) {
      x += texts[i]->w + panel->prop2;
    }
    blit_color_shaded(ui->pixels, x, 0, UI_WIDTH, UI_HEIGHT_BUFFER, texts[panel->prop1]->w, PANEL_HEIGHT, HOVER_R, HOVER_G, HOVER_B);
  }

  {
    int x = 5;
    for (int i = 0; i < ui->tab; i++) {
      x += texts[i]->w + panel->prop2;
    }
    blit_gray(ui->pixels_mask, x - 5, PANEL_HEIGHT - 2, UI_WIDTH, UI_HEIGHT_BUFFER, texts[ui->tab]->w + 10, 2, 0xff);
    blit_color(ui->pixels, x - 5, PANEL_HEIGHT - 2, UI_WIDTH, UI_HEIGHT_BUFFER, texts[ui->tab]->w + 10, 2, TAB_R, TAB_G, TAB_B);
    blit_color_shaded(ui->pixels, x, 0, UI_WIDTH, UI_HEIGHT_BUFFER, texts[ui->tab]->w, PANEL_HEIGHT, TAB_R, TAB_G, TAB_B);
  }
}
