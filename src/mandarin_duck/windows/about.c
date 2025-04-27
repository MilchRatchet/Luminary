#include "about.h"

#include <stdio.h>

#include "display.h"
#include "elements/button.h"
#include "elements/checkbox.h"
#include "elements/separator.h"
#include "elements/text.h"

static void _window_about_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  element_separator(window, mouse_state, (ElementSeparatorArgs) {.text = "About", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Mandarin Duck - CPU Graphical User Interface for Luminary",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Copyright(C) 2024 - 2025 Max Jenke",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the GNU Affero General Public Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Luminary - Path Tracing Renderer",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Copyright(C) 2021 - 2025 Max Jenke",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the GNU Affero General Public Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Current Render", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  window_push_section(window, 32, 4);
  {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.85f, .rel_height = 1.0f},
                         .text         = "Sample Count",
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    char string[256];
    sprintf(string, "%5u", display->current_render_meta_data.sample_count);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.15f, .rel_height = 1.0f},
                         .text         = string,
                         .center_x     = true,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = false,
                         .auto_size    = false,
                         .is_clickable = false});
  }
  window_pop_section(window);

  window_push_section(window, 32, 4);
  {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.85f, .rel_height = 1.0f},
                         .text         = "Elapsed Time",
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    char string[256];
    sprintf(string, "%7.1fs", display->current_render_meta_data.elapsed_time);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.15f, .rel_height = 1.0f},
                         .text         = string,
                         .center_x     = true,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = false,
                         .auto_size    = false,
                         .is_clickable = false});
  }
  window_pop_section(window);

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Devices", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  uint32_t device_count;
  LUM_FAILURE_HANDLE(luminary_host_get_device_count(host, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    LuminaryDeviceInfo device_info;
    LUM_FAILURE_HANDLE(luminary_host_get_device_info(host, device_id, &device_info));

    // TODO: Allow for toggling of devices
    bool device_active = device_info.is_enabled;

    window_push_section(window, 32, 4);
    {
      if (device_info.is_unavailable) {
        element_button(
          window, display, mouse_state,
          (ElementButtonArgs) {.shape        = ELEMENT_BUTTON_SHAPE_IMAGE,
                               .image        = ELEMENT_BUTTON_IMAGE_ERROR,
                               .size         = (ElementSize) {.width = 24, .height = 24},
                               .color        = MD_COLOR_ACCENT_LIGHT_2,
                               .hover_color  = MD_COLOR_ACCENT_LIGHT_2,
                               .press_color  = MD_COLOR_ACCENT_LIGHT_2,
                               .tooltip_text = "GPU is unavailable, see logs."});
      }
      else {
        element_checkbox(
          window, display, mouse_state,
          (ElementCheckBoxArgs) {.size = (ElementSize) {.width = 24, .height = 24}, .data_binding = &device_active});
      }

      element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 0.7f, .rel_height = 1.0f},
                           .text         = device_info.name,
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = false});

      char memory_string[64];

      if (device_info.is_unavailable) {
        sprintf(memory_string, "N/A");
      }
      else {
        sprintf(
          memory_string, "%.1f GB / %.1f GB", device_info.allocated_memory_size * (1.0 / (1024.0 * 1024.0 * 1024.0)),
          device_info.memory_size * (1.0 / (1024.0 * 1024.0 * 1024.0)));
      }

      element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 0.25f, .rel_height = 1.0f},
                           .text         = memory_string,
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = false});
    }
    window_pop_section(window);

    if (device_info.is_unavailable == false && device_active != device_info.is_enabled) {
      LUM_FAILURE_HANDLE(luminary_host_set_device_enable(host, device_id, device_active));
    }
  }

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Third Party Licences", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Simple DirectMedia Layer (SDL) Version 3.0",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  if (element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                           .text         = "https://github.com/libsdl-org/SDL",
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = true})) {
    SDL_OpenURL("https://github.com/libsdl-org/SDL");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Copyright(C) 1997 - 2025 Sam Lantinga",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the Zlib Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "SDL_ttf Version 3.0",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  if (element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                           .text         = "https://github.com/libsdl-org/SDL_ttf",
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = true})) {
    SDL_OpenURL("https://github.com/libsdl-org/SDL_ttf");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Copyright(C) 1997 - 2025 Sam Lantinga",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the Zlib Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Freetype",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  if (element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                           .text         = "https://github.com/freetype/freetype",
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = true})) {
    SDL_OpenURL("https://github.com/freetype/freetype");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the Freetype Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Material Symbols",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});

  if (element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                           .text         = "https://github.com/google/material-design-icons",
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = true})) {
    SDL_OpenURL("https://github.com/google/material-design-icons");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {.color        = 0xFFFFFFFF,
                       .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                       .text         = "Licensed under the Apache Version 2.0 Licence",
                       .center_x     = false,
                       .center_y     = true,
                       .highlighting = false,
                       .cache_text   = true,
                       .auto_size    = false,
                       .is_clickable = false});
}

void window_about_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type            = WINDOW_TYPE_ABOUT;
  (*window)->visibility_mask = WINDOW_VISIBILITY_UTILITIES;
  (*window)->x               = 128;
  (*window)->y               = 128;
  (*window)->width           = 512;
  (*window)->height          = 48;
  (*window)->padding         = 8;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = false;
  (*window)->is_movable      = true;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->auto_align      = false;
  (*window)->action_func     = _window_about_action;
  (*window)->fixed_depth     = false;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}
