#ifndef MANDARIN_DUCK_WINDOW_H
#define MANDARIN_DUCK_WINDOW_H

#include "element.h"
#include "utils.h"

struct Display typedef Display;
struct Window typedef Window;

enum WindowType {
  WINDOW_TYPE_CAPTION_CONTROLS  = 0,
  WINDOW_TYPE_RENDERER_STATUS   = 1,
  WINDOW_TYPE_ABOUT             = 2,
  WINDOW_TYPE_ENTITY_PROPERTIES = 3,
  WINDOW_TYPE_COUNT
} typedef WindowType;

struct WindowContext {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uint32_t padding;
  bool is_horizontal;
  uint32_t fill;  // Elements add to the fill based on their width/height, the window will give a warning if the fill exceeds limits,
                  // elements must make sure to not render past the fill limit but they must update the fill as if they were not limited so
                  // an accurate final fill is retrieved
} typedef WindowContext;

#define WINDOW_MAX_CONTEXT_DEPTH 8
#define WINDOW_MAX_BLUR_MIP_COUNT 4

/*
 * The Window is passed to each element so an element can resize itself based on window orientation (vertical/horizontal)
 * and its dimensions.
 */
struct Window {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uint32_t padding;
  bool is_horizontal;
  bool is_visible;
  bool background;
  bool (*action_func)(Window* window, Display* display, LuminaryHost* host);

  // Runtime
  uint32_t context_stack_ptr;
  WindowContext context_stack[WINDOW_MAX_CONTEXT_DEPTH];
  Element* element_queue;
  uint8_t* background_blur_buffers[WINDOW_MAX_BLUR_MIP_COUNT];
  uint32_t background_blur_mip_count;
} typedef Window;

void window_create(Window** window);
void window_allocate_memory(Window* window);

bool window_is_mouse_hover(Window* window, Display* display);
bool window_handle_input(Window* window, Display* display, LuminaryHost* host);

void window_margin(Window* window, uint32_t margin);
void window_push_section(Window* window);
void window_pop_section(Window* window);
void window_render(Window* window, Display* display);

void window_destroy(Window** window);

#endif /* MANDARIN_DUCK_WINDOW_H */
