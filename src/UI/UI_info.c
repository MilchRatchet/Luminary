#include "UI_info.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "SDL.h"
#include "UI.h"
#include "UI_blit.h"
#include "UI_text.h"

static void rerender_data_text(UI* ui, UIPanel* panel) {
  if (panel->prop2 == PANEL_INFO_STATIC && panel->data_text)
    return;

  if (panel->data_text)
    SDL_FreeSurface(panel->data_text);

  char* buffer = malloc(64);

  switch (panel->prop1) {
    case PANEL_INFO_TYPE_INT8: {
      const uint8_t value = *((uint8_t*) panel->data);
      sprintf(buffer, "%u", value);
    } break;
    case PANEL_INFO_TYPE_INT16: {
      const uint16_t value = *((uint16_t*) panel->data);
      sprintf(buffer, "%u", value);
    } break;
    case PANEL_INFO_TYPE_INT32: {
      const uint32_t value = *((uint32_t*) panel->data);
      sprintf(buffer, "%u", value);
    } break;
    case PANEL_INFO_TYPE_INT64: {
      const uint64_t value = *((uint64_t*) panel->data);
      sprintf(buffer, "%llu", value);
    } break;
    case PANEL_INFO_TYPE_FP32: {
      const float value = *((float*) panel->data);

      if (fabsf(value - truncf(value)) < 1e-3f) {
        sprintf(buffer, "%u", (uint32_t) value);
      }
      else {
        sprintf(buffer, "%.2f", value);
      }
    } break;
    case PANEL_INFO_TYPE_FP64: {
      const double value = *((double*) panel->data);

      if (fabs(value - trunc(value)) < 1e-3) {
        sprintf(buffer, "%u", (uint32_t) value);
      }
      else {
        sprintf(buffer, "%.2f", value);
      }
    } break;
  }

  panel->data_text = render_text(ui, buffer);
  free(buffer);
}

void render_UIPanel_info(UI* ui, UIPanel* panel, int y) {
  blit_text(ui, panel->title, 5, y + ((PANEL_HEIGHT - panel->title->h) >> 1), UI_WIDTH, UI_HEIGHT_BUFFER);

  if (panel->data) {
    rerender_data_text(ui, panel);
    blit_text(
      ui, panel->data_text, UI_WIDTH - 5 - panel->data_text->w, y + ((PANEL_HEIGHT - panel->data_text->h) >> 1), UI_WIDTH,
      UI_HEIGHT_BUFFER);
  }
}
