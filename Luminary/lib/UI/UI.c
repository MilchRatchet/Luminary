#include <stdlib.h>
#include "UI.h"
#include "UI_blur.h"
#include "UI_text.h"

#define BG_RED 10
#define BG_GREEN 10
#define BG_BLUE 10

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

UI init_UI() {
  UI ui;
  ui.x = 100;
  ui.y = 100;

  ui.pixels = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  ui.alpha  = 0.2f;

  size_t scratch_size = compute_scratch_space();
  ui.scratch          = malloc(scratch_size);
  init_text(&ui);

  return ui;
}

void render_UI(UI* ui) {
  for (int i = 0; i < 3 * UI_WIDTH * UI_HEIGHT; i += 3) {
    ui->pixels[i]     = BG_RED;
    ui->pixels[i + 1] = BG_GREEN;
    ui->pixels[i + 2] = BG_BLUE;
  }
}

void blit_UI(UI* ui, uint8_t* target, int width, int height) {
  blur_background(ui, target, width, height);

  for (int i = 0; i < UI_HEIGHT; i++) {
    for (int j = 0; j < 3 * UI_WIDTH; j++) {
      uint8_t ui_pixel     = ui->pixels[j + i * UI_WIDTH * 3];
      uint8_t target_pixel = target[(i + ui->y) * 3 * width + 3 * ui->x + j];
      target[(i + ui->y) * 3 * width + 3 * ui->x + j] =
        ui->alpha * ui_pixel + (1.0f - ui->alpha) * target_pixel;
    }
  }

  SDL_Surface* text    = render_text(ui);
  uint8_t* text_pixels = (uint8_t*) text->pixels;

  for (int i = 0; i < text->h; i++) {
    for (int j = 0; j < 3 * text->w; j++) {
      uint8_t ui_pixel     = text_pixels[j / 3 + i * text->pitch];
      uint8_t target_pixel = target[(i + ui->y) * 3 * width + 3 * ui->x + j];
      target[(i + ui->y) * 3 * width + 3 * ui->x + j] = (ui_pixel) ? 255 : target_pixel;
    }
  }

  SDL_FreeSurface(text);
}

void free_UI(UI* ui) {
  free(ui->pixels);
  free(ui->scratch);
}
