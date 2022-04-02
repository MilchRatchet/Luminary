#include "UI_color.h"

#include "UI.h"
#include "UI_blit.h"
#include "UI_panel.h"
#include "UI_text.h"

void render_UIPanel_color(UI* ui, UIPanel* panel, int y) {
  blit_text(ui, panel->title, 5, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  const int top  = (PANEL_HEIGHT - PANEL_COLOR_BOX_SIZE) >> 1;
  const int left = UI_WIDTH - 5 - PANEL_COLOR_BOX_SIZE;

  blit_gray(ui->pixels, left, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, PANEL_COLOR_BOX_SIZE, PANEL_COLOR_BOX_SIZE, 255);

  blit_gray(ui->pixels_mask, left, y + top, UI_WIDTH, UI_HEIGHT_BUFFER, PANEL_COLOR_BOX_SIZE, PANEL_COLOR_BOX_SIZE, 0xff);

  float* colors = (float*) panel->data;

  const uint8_t red   = 255.0f * colors[0];
  const uint8_t green = 255.0f * colors[1];
  const uint8_t blue  = 255.0f * colors[2];

  blit_color(
    ui->pixels, left + PANEL_COLOR_BOX_BORDER, y + top + PANEL_COLOR_BOX_BORDER, UI_WIDTH, UI_HEIGHT_BUFFER,
    PANEL_COLOR_BOX_SIZE - 2 * PANEL_COLOR_BOX_BORDER, PANEL_COLOR_BOX_SIZE - 2 * PANEL_COLOR_BOX_BORDER, red, green, blue);
}
