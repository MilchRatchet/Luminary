#include <stdio.h>
#include "UI_text.h"
#include "UI_slider.h"
#include "UI_blit.h"

void handle_mouse_UIPanel_slider(UI* ui, UIPanel* panel, int mouse_state, int x, int y) {
  panel->hover = 1;

  if (SDL_BUTTON_LMASK & mouse_state) {
    SDL_SetRelativeMouseMode(SDL_TRUE);

    SDL_PumpEvents();

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        float value = *((float*) panel->data);
        value += (1.0f + fabsf(value)) * (*((float*) (&panel->prop1))) * event.motion.xrel;
        clamp(value, *((float*) (&panel->prop2)), *((float*) (&panel->prop3)));
        *((float*) panel->data) = value;
        if (panel->voids_frames)
          *(ui->temporal_frames) = 0;
      }
    }
  }
}

void render_UIPanel_slider(UI* ui, UIPanel* panel) {
  blit_text(
    ui, panel->title, 5, ui->scroll_pos + panel->y + ((PANEL_HEIGHT - panel->title->h) >> 1));

  if (panel->hover || !panel->data_text) {
    if (panel->data_text)
      SDL_FreeSurface(panel->data_text);

    char* buffer      = malloc(64);
    const float value = *((float*) panel->data);
    sprintf_s(buffer, 64, "%.3f", value);
    panel->data_text = render_text(ui, buffer);
    free(buffer);
  }

  blit_text(
    ui, panel->data_text, UI_WIDTH - 5 - panel->data_text->w,
    ui->scroll_pos + panel->y + ((PANEL_HEIGHT - panel->data_text->h) >> 1));

  if (panel->hover) {
    blit_color_shaded(
      ui->pixels, UI_WIDTH - 5 - panel->data_text->w, ui->scroll_pos + panel->y, UI_WIDTH,
      panel->data_text->w, PANEL_HEIGHT, HOVER_R, HOVER_G, HOVER_B);
  }
}
