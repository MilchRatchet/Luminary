#include "UI_slider.h"

#include <math.h>
#include <stdio.h>

#include "UI_blit.h"
#include "UI_text.h"

void handle_mouse_UIPanel_slider(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  panel->hover = 1;

  if (SDL_BUTTON_LMASK & mouse_state) {
    SDL_SetRelativeMouseMode(SDL_TRUE);

    if (panel->prop5) {
      float value = panel->data_buffer;
      value += (*((float*) (&panel->prop1))) * ui->mouse_xrel;
      clamp(value, *((float*) (&panel->prop2)), *((float*) (&panel->prop3)));
      panel->data_buffer    = value;
      *((int*) panel->data) = (int) (value + 0.5f);
    }
    else {
      float value = *((float*) panel->data);
      value += (1.0f + fabsf(value)) * (*((float*) (&panel->prop1))) * ui->mouse_xrel;
      clamp(value, *((float*) (&panel->prop2)), *((float*) (&panel->prop3)));
      *((float*) panel->data) = value;
    }

    if (panel->voids_frames && ui->mouse_xrel)
      *(ui->temporal_frames) = 0;
  }
}

void render_UIPanel_slider(UI* ui, UIPanel* panel, int y) {
  blit_text(ui, panel->title, 5, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  if (panel->prop4 || panel->hover || !panel->data_text) {
    if (panel->data_text)
      SDL_FreeSurface(panel->data_text);

    char* buffer = malloc(64);
    if (panel->prop5) {
      const int value = *((int*) panel->data);
      sprintf(buffer, "%d", value);
    }
    else {
      const float value = *((float*) panel->data);
      sprintf(buffer, "%.3f", value);
    }
    panel->data_text = render_text(ui, buffer);
    free(buffer);
  }

  blit_text(
    ui, panel->data_text, UI_WIDTH - 5 - panel->data_text->w, y + ((PANEL_HEIGHT - panel->data_text->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  if (panel->hover) {
    blit_color_shaded(
      ui->pixels, UI_WIDTH - 5 - panel->data_text->w, y, UI_WIDTH, UI_HEIGHT_BUFFER, panel->data_text->w, PANEL_HEIGHT, HOVER_R, HOVER_G,
      HOVER_B);
  }
}
