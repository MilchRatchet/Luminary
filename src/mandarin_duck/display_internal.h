#ifndef MANDARIN_DUCK_DISPLAY_INTERNAL_H
#define MANDARIN_DUCK_DISPLAY_INTERNAL_H

#include "display.h"

WindowVisibilityMask display_get_window_visibility_mask(Display* display);
void display_render_init(Display* display);
void display_screenshot_init(Display* display);
void display_screenshot_request(Display* display, LuminaryHost* host);
void display_screenshot_destroy(Display* display);
void display_scene_handle_save(Display* display, LuminaryHost* host);
void display_scene_request_save(Display* display);

#endif /* MANDARIN_DUCK_DISPLAY_INTERNAL_H */
