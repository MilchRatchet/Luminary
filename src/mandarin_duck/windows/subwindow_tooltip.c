#include "subwindow_tooltip.h"

#include <string.h>

#include "elements/text.h"

#define SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH 512

struct SubwindowTooltipData {
  char string[SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH];
} typedef SubwindowTooltipData;
static_assert(sizeof(SubwindowTooltipData) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static bool _subwindow_tooltip_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  SubwindowTooltipData* data = (SubwindowTooltipData*) window->data;

  element_text(
    window, display, mouse_state,
    (ElementTextArgs){
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .color        = 0xFFFFFFFF,
      .text         = data->string,
      .center_x     = true,
      .center_y     = true,
      .highlighting = true,
      .cache_text   = true,
      .auto_size    = true,
      .is_clickable = false});

  return false;
}

static void _subwindow_tooltip_propagate_parent(Window* window, Window* parent) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(parent);
}

void subwindow_tooltip_create(Window* window, const char* text, uint32_t x, uint32_t y) {
  MD_CHECK_NULL_ARGUMENT(window);

  window->type                  = WINDOW_TYPE_SUBWINDOW_TOOLTIP;
  window->x                     = x;
  window->y                     = y;
  window->width                 = 8;
  window->height                = 32;
  window->padding               = 0;
  window->is_horizontal         = true;
  window->is_visible            = true;
  window->is_movable            = false;
  window->background            = true;
  window->auto_size             = true;
  window->action_func           = _subwindow_tooltip_action;
  window->propagate_parent_func = _subwindow_tooltip_propagate_parent;
  window->fixed_depth           = true;
  window->depth                 = UINT64_MAX;
  window->is_subwindow          = true;

  SubwindowTooltipData* data = (SubwindowTooltipData*) window->data;

  const size_t string_length = strlen(text);

  if (string_length >= SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH) {
    crash_message("Tooltip message is too large.");
  }

  memset(data->string, 0, SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH);
  memcpy(data->string, text, string_length);

  window_allocate_memory(window);
}
