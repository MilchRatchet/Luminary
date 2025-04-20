#include "subwindow_dropdown.h"

#include <string.h>

#include "display.h"
#include "elements/text.h"

#define SUBWINDOW_DROPDOWN_MAX_NUM_STRINGS 12
#define SUBWINDOW_DROPDOWN_MAX_STRING_LENGTH 256

struct SubwindowDropdownData {
  uint32_t selected_index;
  uint32_t num_strings;
  char strings[SUBWINDOW_DROPDOWN_MAX_STRING_LENGTH][SUBWINDOW_DROPDOWN_MAX_NUM_STRINGS];
} typedef SubwindowDropdownData;
static_assert(sizeof(SubwindowDropdownData) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static void _subwindow_dropdown_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  SubwindowDropdownData* data = (SubwindowDropdownData*) window->data;

  for (uint32_t string_id = 0; string_id < data->num_strings; string_id++) {
    window_margin(window, 4);
    if (element_text(
          window, display, mouse_state,
          (ElementTextArgs) {.size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                             .color        = 0xFFFFFFFF,
                             .text         = data->strings[string_id],
                             .center_x     = true,
                             .center_y     = true,
                             .highlighting = true,
                             .cache_text   = true,
                             .auto_size    = false,
                             .is_clickable = true})) {
      data->selected_index = string_id;
    }
    window_margin(window, 4);
  }
}

void subwindow_dropdown_add_string(Window* window, const char* string) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(string);

  SubwindowDropdownData* data = (SubwindowDropdownData*) window->data;

  if (data->num_strings == SUBWINDOW_DROPDOWN_MAX_NUM_STRINGS) {
    crash_message("Exceeded maximum number of dropdown elements.");
  }

  const size_t string_length = strlen(string);

  if (string_length >= SUBWINDOW_DROPDOWN_MAX_STRING_LENGTH) {
    crash_message("Exceeded maximum string length in dropdown.");
  }

  memset(data->strings[data->num_strings], 0, SUBWINDOW_DROPDOWN_MAX_STRING_LENGTH * sizeof(char));
  memcpy(data->strings[data->num_strings], string, string_length);

  data->num_strings++;
}

static void _subwindow_dropdown_propagate_parent(Window* window, Window* parent) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(parent);

  SubwindowDropdownData* data = (SubwindowDropdownData*) window->data;

  parent->state_data.dropdown_selection = data->selected_index;
}

void subwindow_dropdown_create(Window* window, uint32_t selected_index, uint32_t width, uint32_t x, uint32_t y) {
  MD_CHECK_NULL_ARGUMENT(window);

  window->type                  = WINDOW_TYPE_SUBWINDOW_DROPDOWN;
  window->visibility_mask       = WINDOW_VISIBILITY_UTILITIES;
  window->x                     = x;
  window->y                     = y;
  window->width                 = width;
  window->height                = WINDOW_ROUNDING_SIZE;
  window->padding               = WINDOW_ROUNDING_SIZE >> 1;
  window->is_horizontal         = false;
  window->is_visible            = true;
  window->is_movable            = false;
  window->background            = true;
  window->auto_size             = true;
  window->action_func           = _subwindow_dropdown_action;
  window->propagate_parent_func = _subwindow_dropdown_propagate_parent;
  window->fixed_depth           = true;
  window->depth                 = UINT64_MAX;
  window->is_subwindow          = true;

  SubwindowDropdownData* data = (SubwindowDropdownData*) window->data;

  data->selected_index = selected_index;
  data->num_strings    = 0;

  window_allocate_memory(window);
}
