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
  WINDOW_TYPE_FRAMETIME         = 4,
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

struct WindowMargins {
  int32_t margin_left;
  int32_t margin_right;
  int32_t margin_top;
  int32_t margin_bottom;
} typedef WindowMargins;

enum WindowInteractionState { WINDOW_INTERACTION_STATE_NONE, WINDOW_INTERACTION_STATE_SLIDER } typedef WindowInteractionState;

struct WindowInteractionStateData {
  WindowInteractionState state;
  uint64_t element_hash;
  uint32_t subelement_index;
} typedef WindowInteractionStateData;

#define WINDOW_MAX_CONTEXT_DEPTH 8
#define WINDOW_MAX_BLUR_MIP_COUNT 5
#define WINDOW_MARGIN_INVALID INT32_MAX

/*
 * The Window is passed to each element so an element can resize itself based on window orientation (vertical/horizontal)
 * and its dimensions.
 */
struct Window {
  int32_t x;
  int32_t y;
  uint32_t width;
  uint32_t height;
  uint32_t padding;
  WindowMargins margins;
  bool is_horizontal;
  bool is_visible;
  bool is_movable;
  bool background;
  bool auto_align;
  bool auto_size;
  bool (*action_func)(Window* window, Display* display, LuminaryHost* host);
  WindowInteractionStateData state_data;

  // Runtime
  uint32_t context_stack_ptr;
  WindowContext context_stack[WINDOW_MAX_CONTEXT_DEPTH];
  Element* element_queue;
  uint8_t* background_blur_buffer;
  uint32_t background_blur_buffer_ld;
  bool element_has_hover;
} typedef Window;

void window_create(Window** window);
void window_allocate_memory(Window* window);

bool window_is_mouse_hover(Window* window, Display* display);
bool window_handle_input(Window* window, Display* display, LuminaryHost* host);

void window_margin(Window* window, uint32_t margin);
void window_margin_relative(Window* window, float margin);
void window_push_section(Window* window, uint32_t size, uint32_t padding);
void window_pop_section(Window* window);
void window_render(Window* window, Display* display);

void window_destroy(Window** window);

#endif /* MANDARIN_DUCK_WINDOW_H */
