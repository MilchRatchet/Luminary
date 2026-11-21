#include <string.h>

#include "display_internal.h"

// TODO: Let Mandarin Duck have its own embedded files.
void ceb_access(const char* restrict name, void** restrict ptr, int64_t* restrict lmem, uint64_t* restrict info);

#define DISPLAY_MIN_FRAME_TIME_NS 16666666  // Nanoseconds frametime at 60 FPS

#define DISPLAY_SPLASHSCREEN_WIDTH 640
#define DISPLAY_SPLASHSCREEN_HEIGHT 180
#define DISPLAY_SPLASHSCREEN_DATA_OFFSET 0x8A

void display_render_init(Display* display) {
  uint8_t* splash_screen_data;
  int64_t splash_screen_size;
  uint64_t info;
  ceb_access("SplashScreen.bmp", (void**) &splash_screen_data, &splash_screen_size, &info);

  if (info || splash_screen_size < (DISPLAY_SPLASHSCREEN_WIDTH * DISPLAY_SPLASHSCREEN_HEIGHT * 3 + DISPLAY_SPLASHSCREEN_DATA_OFFSET)) {
    crash_message("Splash screen is corrupted. Luminary needs to be rebuild. Ceb Error Code: %zu", info);
  }

  display->splash_screen = (const uint32_t*) (splash_screen_data + DISPLAY_SPLASHSCREEN_DATA_OFFSET);
}

static uint32_t _display_clip_blit_edge(int64_t image_edge, int32_t scale, uint32_t display_size) {
  if (image_edge <= 0)
    return 0;

  const uint64_t edge      = (uint64_t) image_edge;
  const uint64_t projected = (scale >= 0) ? (edge << scale) : (((edge - 1) >> -scale) + 1);
  return (uint32_t) min(projected, display_size);
}

static void _display_fill_background(uint32_t* row, uint32_t begin, uint32_t end) {
  for (uint32_t pixel = begin; pixel < end; pixel++) {
    row[pixel] = MD_COLOR_WINDOW_BACKGROUND;
  }
}

static void _display_blit_to_display_buffer(Display* display, LuminaryImage image) {
  const int32_t scale        = display->zoom_handler->scale;
  const int64_t src_offset_x = display->zoom_handler->offset_x;
  const int64_t src_offset_y = display->zoom_handler->offset_y;

  const uint32_t x_begin   = _display_clip_blit_edge(-src_offset_x, scale, display->width);
  const uint32_t x_end     = _display_clip_blit_edge(image.width - src_offset_x, scale, display->width);
  const uint32_t y_begin   = _display_clip_blit_edge(-src_offset_y, scale, display->height);
  const uint32_t y_end     = _display_clip_blit_edge(image.height - src_offset_y, scale, display->height);
  const LuminaryARGB8* src = (const LuminaryARGB8*) image.buffer;

  for (uint32_t row = 0; row < display->height; row++) {
    uint32_t* background = (uint32_t*) (display->buffer + (size_t) row * display->pitch);
    if (row < y_begin || row >= y_end || x_begin == x_end) {
      _display_fill_background(background, 0, display->width);
      continue;
    }

    _display_fill_background(background, 0, x_begin);
    _display_fill_background(background, x_end, display->width);

    const int64_t src_y          = ((scale >= 0) ? ((int64_t) row >> scale) : ((int64_t) row << -scale)) + src_offset_y;
    const LuminaryARGB8* src_row = src + (size_t) src_y * image.ld;
    LuminaryARGB8* dst_row       = (LuminaryARGB8*) background;

    if (scale == 0) {
      memcpy(dst_row + x_begin, src_row + (x_begin + src_offset_x), (size_t) (x_end - x_begin) * sizeof(*dst_row));
    }
    else if (scale > 0) {
      for (uint32_t column = x_begin; column < x_end; column++) {
        dst_row[column] = src_row[(column >> scale) + src_offset_x];
      }
    }
    else {
      const size_t stride = (size_t) 1 << -scale;
      size_t src_x        = (size_t) (((int64_t) x_begin << -scale) + src_offset_x);
      for (uint32_t column = x_begin; column < x_end; column++, src_x += stride) {
        dst_row[column] = src_row[src_x];
      }
    }
  }
}

static void _display_render_output(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputHandle output_handle;
  LUM_FAILURE_HANDLE(luminary_host_acquire_output(host, &output_handle));

  if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID) {
    for (uint32_t y = 0; y < display->height; y++) {
      uint32_t* dst = (uint32_t*) (display->buffer + (size_t) y * display->pitch);
      _display_fill_background(dst, 0, display->width);
    }

    // Render the splash screen, but only if the display is large enough.
    if (display->width >= DISPLAY_SPLASHSCREEN_WIDTH && display->height >= DISPLAY_SPLASHSCREEN_HEIGHT) {
      uint32_t splash_screen_x = (display->width - DISPLAY_SPLASHSCREEN_WIDTH) >> 1;
      uint32_t splash_screen_y = (display->height - DISPLAY_SPLASHSCREEN_HEIGHT) >> 1;

      for (uint32_t y = 0; y < DISPLAY_SPLASHSCREEN_HEIGHT; y++) {
        uint32_t* dst = (uint32_t*) (display->buffer + (splash_screen_y + y) * display->pitch);
        for (uint32_t x = 0; x < DISPLAY_SPLASHSCREEN_WIDTH; x++) {
          dst[splash_screen_x + x] = display->splash_screen[(DISPLAY_SPLASHSCREEN_HEIGHT - y - 1) * DISPLAY_SPLASHSCREEN_WIDTH + x];
        }
      }
    }

    return;
  }

  LuminaryImage output_image;
  LUM_FAILURE_HANDLE(luminary_host_get_image(host, output_handle, &output_image));

  if (display->zoom_handler->image_width != output_image.width || display->zoom_handler->image_height != output_image.height)
    display_zoom_handler_set_image_size(display->zoom_handler, output_image.width, output_image.height);

  display->current_render_meta_data.elapsed_time = output_image.meta_data.time;
  display->current_render_meta_data.sample_count = output_image.meta_data.sample_count;

  _display_blit_to_display_buffer(display, output_image);

  LUM_FAILURE_HANDLE(luminary_host_release_output(host, output_handle));
}

void display_render(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (display->buffer == NULL)
    return;

  _display_render_output(display, host);

  WindowVisibilityMask visibility_mask = display_get_window_visibility_mask(display);
  user_interface_render(display->ui, display, visibility_mask);
}

void display_update(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  const uint64_t ticks_current = SDL_GetTicksNS();
  const uint64_t elapsed       = ticks_current - display->ticks_last_frame;
  if (display->frame_id != 0 && elapsed < DISPLAY_MIN_FRAME_TIME_NS)
    SDL_DelayPrecise(DISPLAY_MIN_FRAME_TIME_NS - elapsed);

  display->ticks_last_frame = SDL_GetTicksNS();

  SDL_SetCursor(display->sdl_cursors[display->selected_cursor]);

  if (display->sdl_surface != NULL && SDL_UpdateWindowSurface(display->sdl_window))
    display->buffer = display->sdl_surface->pixels;

  display->frame_id++;
}
