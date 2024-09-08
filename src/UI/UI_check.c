#include "UI_check.h"

#include "UI_blit.h"
#include "UI_structs.h"
#include "UI_text.h"
#include "utils.h"

void handle_mouse_UIPanel_check(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  LUM_UNUSED(x);
  LUM_UNUSED(y);

  panel->hover = 1;

  if (SDL_BUTTON_LMASK & mouse_state) {
    int checked = *((int*) panel->data);
    checked ^= 1;
    *((int*) panel->data) = checked;

    if (panel->voids_frames)
      *(ui->temporal_frames) = 0.0f;
  }
}

void render_UIPanel_check(UI* ui, UIPanel* panel, int y) {
  blit_text(ui, panel->title, 5, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  const int top  = (PANEL_HEIGHT - PANEL_CHECK_BOX_SIZE) >> 1;
  const int left = UI_WIDTH - 5 - PANEL_CHECK_BOX_SIZE;

  if (panel->hover) {
    blit_color(
      ui->pixels, left, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, PANEL_CHECK_BOX_SIZE, PANEL_CHECK_BOX_SIZE, HOVER_R, HOVER_G, HOVER_B);
  }
  else {
    blit_gray(ui->pixels, left, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, PANEL_CHECK_BOX_SIZE, PANEL_CHECK_BOX_SIZE, 255);
  }

  blit_gray(ui->pixels_mask, left, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, PANEL_CHECK_BOX_SIZE, PANEL_CHECK_BOX_SIZE, 0xff);

  const int checked = *((int*) panel->data);

  if (!checked) {
    blit_gray(
      ui->pixels_mask, left + PANEL_CHECK_BOX_BORDER, y + top + PANEL_CHECK_BOX_BORDER, UI_WIDTH, UI_HEIGHT_BUFFER,
      PANEL_CHECK_BOX_SIZE - 2 * PANEL_CHECK_BOX_BORDER, PANEL_CHECK_BOX_SIZE - 2 * PANEL_CHECK_BOX_BORDER, 0x00);
  }
}
