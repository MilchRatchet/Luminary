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
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Mandarin Duck - CPU Graphical User Interface for Luminary",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Copyright(C) 2024 - 2026 Max Jenke",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the GNU Affero General Public Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Luminary - Path Tracing Renderer",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Copyright(C) 2021 - 2026 Max Jenke",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the GNU Affero General Public Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Current Render", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  window_push_section(window, 32, 4);
  {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 0.85f, .rel_height = 1.0f},
        .text         = "Sample Count",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = false,
      });

    char string[256];
    sprintf(string, "%5u", display->current_render_meta_data.sample_count);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 0.15f, .rel_height = 1.0f},
        .text         = string,
        .center_x     = true,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false,
      });
  }
  window_pop_section(window);

  window_push_section(window, 32, 4);
  {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 0.85f, .rel_height = 1.0f},
        .text         = "Elapsed Time",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = false,
      });

    char string[256];
    sprintf(string, "%7.1fs", display->current_render_meta_data.elapsed_time);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 0.15f, .rel_height = 1.0f},
        .text         = string,
        .center_x     = true,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false,
      });
  }
  window_pop_section(window);

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Third Party Licences", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Simple DirectMedia Layer (SDL) Version 3.0",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  if (
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
        .text         = "https://github.com/libsdl-org/SDL",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = true,
      })) {
    SDL_OpenURL("https://github.com/libsdl-org/SDL");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Copyright(C) 1997 - 2025 Sam Lantinga",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the Zlib Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "SDL_ttf Version 3.0",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  if (
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
        .text         = "https://github.com/libsdl-org/SDL_ttf",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = true,
      })) {
    SDL_OpenURL("https://github.com/libsdl-org/SDL_ttf");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Copyright(C) 1997 - 2025 Sam Lantinga",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the Zlib Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Freetype",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  if (
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
        .text         = "https://github.com/freetype/freetype",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = true,
      })) {
    SDL_OpenURL("https://github.com/freetype/freetype");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the Freetype Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  window_margin(window, 24);

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Material Symbols",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });

  if (
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
        .text         = "https://github.com/google/material-design-icons",
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = true,
      })) {
    SDL_OpenURL("https://github.com/google/material-design-icons");
  }

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .text         = "Licensed under the Apache Version 2.0 Licence",
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true,
      .auto_size    = false,
      .is_clickable = false,
    });
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
