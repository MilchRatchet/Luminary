#include "UI_button.h"

#include "UI_blit.h"
#include "UI_structs.h"
#include "UI_text.h"
#include "utils.h"

void handle_mouse_UIPanel_button(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  LUM_UNUSED(y);

  if (x >= UI_WIDTH - 15 - panel->title->w && x <= UI_WIDTH - 5) {
    panel->hover = 1;

    if (SDL_BUTTON_LMASK & mouse_state) {
      panel->func(panel->data);

      if (panel->voids_frames)
        *(ui->temporal_frames) = 0.0f;
    }
  }
}

void render_UIPanel_button(UI* ui, UIPanel* panel, int y) {
  const int top  = (PANEL_HEIGHT - BUTTON_SIZE) >> 1;
  const int left = UI_WIDTH - 10 - panel->title->w;

  blit_gray(ui->pixels, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->title->w + 10, BUTTON_SIZE, 0xff);
  blit_gray(ui->pixels_mask, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->title->w + 10, BUTTON_SIZE, 0xff);
  blit_gray(ui->pixels, left - 4, y + top + 1, UI_WIDTH, UI_HEIGHT_BUFFER, panel->title->w + 8, BUTTON_SIZE - 2, 0x00);
  blit_gray(ui->pixels_mask, left - 4, y + top + 1, UI_WIDTH, UI_HEIGHT_BUFFER, panel->title->w + 8, BUTTON_SIZE - 2, 0x00);
  blit_text(ui, panel->title, left, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  if (panel->hover) {
    blit_color_shaded(
      ui->pixels, left - 5, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, panel->title->w + 10, BUTTON_SIZE, HOVER_R, HOVER_G, HOVER_B);
  }
}
