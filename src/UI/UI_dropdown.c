#include "UI_dropdown.h"

#include "UI.h"
#include "UI_blit.h"
#include "UI_panel.h"
#include "UI_text.h"

void handle_mouse_UIPanel_dropdown(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  panel->hover = 1;

  if (panel->prop2) {
    const int left  = UI_WIDTH - 10 - panel->prop4 - 5;
    const int right = UI_WIDTH - 5;

    if (x >= left && x <= right) {
      y -= panel->prop3 * PANEL_HEIGHT + UI_BORDER_SIZE + PANEL_DROPDOWN_BOX_HEIGHT + 5;

      if (y > 0) {
        y--;
        y /= PANEL_DROPDOWN_BOX_HEIGHT;
        panel->prop5 = y;

        if (SDL_BUTTON_LMASK & mouse_state) {
          *((int*) panel->data) = y;
          if (panel->voids_frames)
            *(ui->temporal_frames) = 0.0f;
        }
      }
    }

    if (SDL_BUTTON_LMASK & mouse_state) {
      panel->prop2 = 0;
      ui->dropdown = (UIPanel*) 0;
    }
  }
  else {
    if (SDL_BUTTON_LMASK & mouse_state) {
      panel->prop2 = 1;
      ui->dropdown = panel;
    }
  }
}

int get_intersection_dropdown(UIPanel* panel, int x, int y) {
  int top    = panel->prop3 * PANEL_HEIGHT + UI_BORDER_SIZE + PANEL_DROPDOWN_BOX_HEIGHT + 5;
  int bottom = top + (panel->prop1) * PANEL_DROPDOWN_BOX_HEIGHT;
  int left   = UI_WIDTH - 10 - panel->prop4 - 5;
  int right  = UI_WIDTH - 5;

  return (y >= top) && (y <= bottom) && (x >= left) && (x <= right);
}

void render_UIPanel_dropdown(UI* ui, UIPanel* panel, int y) {
  blit_text(ui, panel->title, 5, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  int selected             = *((int*) panel->data);
  SDL_Surface** texts      = (SDL_Surface**) panel->data_text;
  SDL_Surface* option_text = texts[selected];

  const int top  = (PANEL_HEIGHT - PANEL_DROPDOWN_BOX_HEIGHT) >> 1;
  const int left = UI_WIDTH - 10 - panel->prop4;

  if (panel->hover || panel->prop2) {
    blit_gray(ui->pixels, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 10, PANEL_DROPDOWN_BOX_HEIGHT, 0xff);
    blit_gray(ui->pixels_mask, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 10, PANEL_DROPDOWN_BOX_HEIGHT, 0xff);
    blit_gray(ui->pixels, left - 4, y + top + 1, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 8, PANEL_DROPDOWN_BOX_HEIGHT - 2, 0x00);
    blit_gray(ui->pixels_mask, left - 4, y + top + 1, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 8, PANEL_DROPDOWN_BOX_HEIGHT - 2, 0x00);
    blit_text(ui, option_text, left, y + ((PANEL_HEIGHT - option_text->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);
    blit_color_shaded(
      ui->pixels, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 10, PANEL_DROPDOWN_BOX_HEIGHT, HOVER_R, HOVER_G, HOVER_B);
  }
  else {
    blit_text(ui, option_text, left, y + ((PANEL_HEIGHT - option_text->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);
  }
}

void render_dropdown(UI* ui, UIPanel* panel, int offset) {
  if (!panel->prop2)
    return;

  const int y = (panel->prop3 - offset) * PANEL_HEIGHT;

  SDL_Surface** texts = (SDL_Surface**) panel->data_text;

  const int top  = (PANEL_DROPDOWN_BOX_HEIGHT - texts[0]->h) >> 1;
  const int left = UI_WIDTH - 10 - panel->prop4;

  blit_gray(
    ui->pixels, left - 5, y + PANEL_DROPDOWN_BOX_HEIGHT + 5, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 10,
    panel->prop1 * PANEL_DROPDOWN_BOX_HEIGHT, 0xff);
  blit_gray(
    ui->pixels_mask, left - 5, y + PANEL_DROPDOWN_BOX_HEIGHT + 5, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 10,
    panel->prop1 * PANEL_DROPDOWN_BOX_HEIGHT, 0xff);
  blit_gray(
    ui->pixels, left - 4, y + PANEL_DROPDOWN_BOX_HEIGHT + 6, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 8,
    panel->prop1 * PANEL_DROPDOWN_BOX_HEIGHT - 2, 0x00);
  blit_gray(
    ui->pixels_mask, left - 4, y + PANEL_DROPDOWN_BOX_HEIGHT + 6, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 8,
    panel->prop1 * PANEL_DROPDOWN_BOX_HEIGHT - 2, 0x00);

  for (int i = 0; i < panel->prop1; i++) {
    blit_text(ui, texts[i], left, y + top + (i + 1) * PANEL_DROPDOWN_BOX_HEIGHT + 5, UI_WIDTH, UI_HEIGHT_BUFFER);

    if (panel->prop5 == i) {
      blit_color_shaded(
        ui->pixels, left - 4, y + (i + 1) * PANEL_DROPDOWN_BOX_HEIGHT + 6, UI_WIDTH, UI_HEIGHT_BUFFER, panel->prop4 + 8,
        PANEL_DROPDOWN_BOX_HEIGHT - 2, HOVER_R, HOVER_G, HOVER_B);
    }
  }

  panel->prop5 = -1;
}
