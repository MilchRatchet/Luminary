#ifndef MANDARIN_DUCK_WINDOW_H
#define MANDARIN_DUCK_WINDOW_H

#include "element.h"
#include "utils.h"

struct Display typedef Display;
struct Window typedef Window;
struct MouseState typedef MouseState;

#define WINDOW_MAX_CONTEXT_DEPTH 8
#define WINDOW_MAX_BLUR_MIP_COUNT 5
#define WINDOW_MARGIN_INVALID INT32_MAX
#define WINDOW_ROUNDING_SIZE 32
#define WINDOW_DATA_SECTION_SIZE 4096
#define WINDOW_STATE_STRING_SIZE 256

#define WINDOW_VISIBILITY_CAPTION_CONTROLS 0b001
#define WINDOW_VISIBILITY_STATUS 0b010
#define WINDOW_VISIBILITY_UTILITIES 0b100

#define WINDOW_VISIBILITY_MASK_ALL (WINDOW_VISIBILITY_CAPTION_CONTROLS | WINDOW_VISIBILITY_STATUS | WINDOW_VISIBILITY_UTILITIES)
#define WINDOW_VISIBILITY_MASK_MOVEMENT (WINDOW_VISIBILITY_CAPTION_CONTROLS | WINDOW_VISIBILITY_STATUS)
#define WINDOW_VISIBILITY_NONE 0

typedef uint32_t WindowVisibilityMask;

enum WindowType {
  WINDOW_TYPE_CAPTION_CONTROLS,
  WINDOW_TYPE_RENDERER_STATUS,
  WINDOW_TYPE_ABOUT,
  WINDOW_TYPE_ENTITY_PROPERTIES,
  WINDOW_TYPE_FRAMETIME,
  WINDOW_TYPE_SIDEBAR,
  WINDOW_TYPE_SUBWINDOW_DROPDOWN,
  WINDOW_TYPE_SUBWINDOW_TOOLTIP,
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

enum WindowInteractionState {
  WINDOW_INTERACTION_STATE_NONE,
  WINDOW_INTERACTION_STATE_DRAG,
  WINDOW_INTERACTION_STATE_SLIDER,
  WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER,
  WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED,
  WINDOW_INTERACTION_STATE_STRING
} typedef WindowInteractionState;

struct WindowInteractionStateData {
  WindowInteractionState state;
  uint64_t element_hash;
  uint32_t subelement_index;
  uint32_t dropdown_selection;
  uint32_t num_characters;
  char string[WINDOW_STATE_STRING_SIZE];
  bool force_string_mode_exit;
} typedef WindowInteractionStateData;

/*
 * The Window is passed to each element so an element can resize itself based on window orientation (vertical/horizontal)
 * and its dimensions.
 */
struct Window {
  WindowType type;
  WindowVisibilityMask visibility_mask;
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
  bool fixed_depth;
  bool is_subwindow;
  bool (*action_func)(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state);
  void (*propagate_parent_func)(Window* window, Window* parent);
  WindowInteractionStateData state_data;
  uint64_t depth;

  // Runtime
  uint8_t data[WINDOW_DATA_SECTION_SIZE];
  uint32_t context_stack_ptr;
  WindowContext context_stack[WINDOW_MAX_CONTEXT_DEPTH];
  Element* element_queue;
  uint8_t* background_blur_buffer;
  uint32_t background_blur_buffer_ld;
  size_t background_blur_buffer_size;
  bool element_has_hover;
  Window* external_subwindow;
} typedef Window;

void window_create(Window** window);
void window_create_subwindow(Window* window);
void window_allocate_memory(Window* window);

bool window_is_mouse_hover(Window* window, Display* display, const MouseState* mouse_state);
bool window_handle_input(Window* window, Display* display, LuminaryHost* host, MouseState* mouse_state);

void window_set_focus(Window* window);

void window_push_element(Window* window, Element* element);
void window_margin(Window* window, uint32_t margin);
void window_margin_relative(Window* window, float margin);
void window_push_section(Window* window, uint32_t size, uint32_t padding);
void window_pop_section(Window* window);
void window_render(Window* window, Display* display);

void window_destroy(Window** window);

#endif /* MANDARIN_DUCK_WINDOW_H */
