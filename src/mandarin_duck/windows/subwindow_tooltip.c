#include "subwindow_tooltip.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "display.h"
#include "elements/separator.h"
#include "elements/table.h"
#include "elements/text.h"
#include "text_renderer.h"

#define SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH 512
#define SUBWINDOW_TOOLTIP_STACK_TRACE_CHUNK_SIZE 255
#define SUBWINDOW_TOOLTIP_STACK_TRACE_MAX_VISIBLE_ROWS 16
#define SUBWINDOW_TOOLTIP_STACK_TRACE_ROW_HEIGHT 24
#define SUBWINDOW_TOOLTIP_STACK_TRACE_WIDTH 960

struct SubwindowTooltipStackTraceRow {
  char text[SUBWINDOW_TOOLTIP_STACK_TRACE_CHUNK_SIZE + 1];
} typedef SubwindowTooltipStackTraceRow;

struct SubwindowTooltipData {
  char string[SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH];
  char* stack_trace_text;
  SubwindowTooltipStackTraceRow* stack_trace_rows;
  uint32_t stack_trace_row_count;
  uint32_t stack_trace_text_max_width;
  uint32_t visible_row_count;
  uint32_t allocation_index;
  float first_visible_row;
  bool is_stack_trace;
} typedef SubwindowTooltipData;
static_assert(sizeof(SubwindowTooltipData) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static void _subwindow_tooltip_destroy_stack_trace(SubwindowTooltipData* data) {
  free(data->stack_trace_text);
  free(data->stack_trace_rows);
  data->stack_trace_text = (char*) 0;
  data->stack_trace_rows = (SubwindowTooltipStackTraceRow*) 0;

  data->stack_trace_row_count      = 0;
  data->stack_trace_text_max_width = 0;
  data->visible_row_count          = 0;
  data->allocation_index           = ELEMENT_TABLE_NO_SELECTION;
  data->first_visible_row          = 0.0f;
  data->is_stack_trace             = false;
}

void subwindow_tooltip_destroy(Window* window) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (window->type != WINDOW_TYPE_SUBWINDOW_TOOLTIP)
    return;

  _subwindow_tooltip_destroy_stack_trace((SubwindowTooltipData*) window->data);
}

static void _subwindow_tooltip_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  SubwindowTooltipData* data = (SubwindowTooltipData*) window->data;

  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
      .color        = 0xFFFFFFFF,
      .text         = data->string,
      .center_x     = true,
      .center_y     = true,
      .highlighting = true,
      .cache_text   = true,
      .auto_size    = true,
      .is_clickable = false});
}

static bool _subwindow_tooltip_format_stack_trace_row(char* text, size_t text_size, const void* data, size_t data_size) {
  if (data_size != sizeof(((SubwindowTooltipStackTraceRow*) 0)->text) || text_size == 0)
    return false;

  snprintf(text, text_size, "%s", (const char*) data);
  return true;
}

static size_t _subwindow_tooltip_stack_trace_chunk_length(Display* display, const char* text, size_t remaining, uint32_t max_text_width) {
  size_t chunk_length = remaining < SUBWINDOW_TOOLTIP_STACK_TRACE_CHUNK_SIZE ? remaining : SUBWINDOW_TOOLTIP_STACK_TRACE_CHUNK_SIZE;

  while (chunk_length > 0) {
    if (chunk_length < remaining) {
      while (chunk_length > 0 && (((unsigned char) text[chunk_length] & 0xC0) == 0x80))
        chunk_length--;
    }

    if (chunk_length == 0)
      break;

    char chunk_text[SUBWINDOW_TOOLTIP_STACK_TRACE_CHUNK_SIZE + 1] = {0};
    memcpy(chunk_text, text, chunk_length);

    uint32_t text_width;
    uint32_t text_height;
    text_renderer_compute_size(display->text_renderer, chunk_text, TEXT_RENDERER_FONT_REGULAR, false, &text_width, &text_height);

    if (text_width <= max_text_width)
      return chunk_length;

    chunk_length /= 2;
  }

  const unsigned char first_character = (unsigned char) text[0];
  if (first_character >= 0xF0)
    return remaining < 4 ? remaining : 4;
  if (first_character >= 0xE0)
    return remaining < 3 ? remaining : 3;
  if (first_character >= 0xC0)
    return remaining < 2 ? remaining : 2;

  return 1;
}

static uint32_t _subwindow_tooltip_count_stack_trace_rows(Display* display, const char* text, uint32_t max_text_width) {
  const size_t text_length = strlen(text);
  size_t text_offset       = 0;
  size_t row_count         = 0;

  while (text_offset < text_length) {
    size_t line_end = text_offset;
    while (line_end < text_length && text[line_end] != '\n')
      line_end++;

    if (line_end == text_offset) {
      row_count++;
    }
    else {
      while (text_offset < line_end) {
        const size_t chunk_length =
          _subwindow_tooltip_stack_trace_chunk_length(display, text + text_offset, line_end - text_offset, max_text_width);
        if (chunk_length == 0)
          crash_message("Failed to split stack trace text.");

        text_offset += chunk_length;
        row_count++;
      }
    }

    if (line_end < text_length)
      text_offset = line_end + 1;

    if (row_count > ELEMENT_TABLE_NO_SELECTION)
      crash_message("Stack trace has too many rows to display.");
  }

  return row_count > 0 ? (uint32_t) row_count : 1;
}

static void _subwindow_tooltip_build_stack_trace_rows(SubwindowTooltipData* data, Display* display, uint32_t max_text_width) {
  const size_t text_length    = strlen(data->stack_trace_text);
  data->stack_trace_row_count = _subwindow_tooltip_count_stack_trace_rows(display, data->stack_trace_text, max_text_width);

  if ((size_t) data->stack_trace_row_count > SIZE_MAX / sizeof(*data->stack_trace_rows))
    crash_message("Stack trace row storage is too large.");

  data->stack_trace_rows = (SubwindowTooltipStackTraceRow*) malloc(sizeof(*data->stack_trace_rows) * data->stack_trace_row_count);
  if (data->stack_trace_rows == (SubwindowTooltipStackTraceRow*) 0)
    crash_message("Failed to allocate stack trace rows.");

  memset(data->stack_trace_rows, 0, sizeof(*data->stack_trace_rows) * data->stack_trace_row_count);

  size_t text_offset = 0;
  uint32_t row_index = 0;

  while (text_offset < text_length) {
    size_t line_end = text_offset;
    while (line_end < text_length && data->stack_trace_text[line_end] != '\n')
      line_end++;

    if (line_end == text_offset) {
      row_index++;
    }
    else {
      while (text_offset < line_end) {
        const size_t chunk_length = _subwindow_tooltip_stack_trace_chunk_length(
          display, data->stack_trace_text + text_offset, line_end - text_offset, max_text_width);
        memcpy(data->stack_trace_rows[row_index].text, data->stack_trace_text + text_offset, chunk_length);
        text_offset += chunk_length;
        row_index++;
      }
    }

    if (line_end < text_length)
      text_offset = line_end + 1;
  }

  if (row_index == 0) {
    snprintf(data->stack_trace_rows[0].text, sizeof(data->stack_trace_rows[0].text), "No stack trace available.");
  }
}

static uint32_t _subwindow_tooltip_visible_stack_trace_rows(const Display* display, uint32_t row_count) {
  const uint32_t max_popup_height = display->height > 16 ? display->height - 16 : display->height;
  const uint32_t popup_chrome     = 16 + 32 + SUBWINDOW_TOOLTIP_STACK_TRACE_ROW_HEIGHT;
  uint32_t visible_row_count =
    max_popup_height > popup_chrome ? (max_popup_height - popup_chrome) / SUBWINDOW_TOOLTIP_STACK_TRACE_ROW_HEIGHT : 1;

  if (visible_row_count > SUBWINDOW_TOOLTIP_STACK_TRACE_MAX_VISIBLE_ROWS)
    visible_row_count = SUBWINDOW_TOOLTIP_STACK_TRACE_MAX_VISIBLE_ROWS;
  if (visible_row_count > row_count)
    visible_row_count = row_count;
  if (visible_row_count == 0)
    visible_row_count = 1;

  return visible_row_count;
}

static void _subwindow_tooltip_stack_trace_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  SubwindowTooltipData* data         = (SubwindowTooltipData*) window->data;
  const uint32_t first_row           = (uint32_t) floorf(data->first_visible_row);
  const uint32_t available_row_count = data->stack_trace_row_count > first_row ? data->stack_trace_row_count - first_row : 0;
  const uint32_t rendered_row_count  = available_row_count < data->visible_row_count ? available_row_count : data->visible_row_count;
  float first_rendered_row           = 0.0f;

  element_separator(
    window, mouse_state,
    (ElementSeparatorArgs) {.text = "Allocation Stack Trace", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  static const ElementTableColumn stack_trace_column = {
    .name      = "Frame",
    .offset    = offsetof(SubwindowTooltipStackTraceRow, text),
    .data_type = ELEMENT_TABLE_DATA_TYPE_CUSTOM,
    .rel_width = 1.0f,
    .data_size = sizeof(((SubwindowTooltipStackTraceRow*) 0)->text),
    .format    = _subwindow_tooltip_format_stack_trace_row,
  };

  element_table(
    window, display, mouse_state,
    (ElementTableArgs) {
      .identifier        = "allocation_stack_trace",
      .data              = data->stack_trace_rows + first_row,
      .element_size      = sizeof(*data->stack_trace_rows),
      .element_count     = rendered_row_count,
      .visible_row_count = data->visible_row_count,
      .columns           = &stack_trace_column,
      .column_count      = 1,
      .size              = (ElementSize) {.rel_width = 1.0f},
      .row_height        = SUBWINDOW_TOOLTIP_STACK_TRACE_ROW_HEIGHT,
      .first_visible_row = &first_rendered_row,
    });
}

static void _subwindow_tooltip_propagate_parent(Window* window, Window* parent) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(parent);
}

void subwindow_tooltip_create(Window* window, const char* text, uint32_t x, uint32_t y) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (window->type != WINDOW_TYPE_SUBWINDOW_TOOLTIP)
    memset(window->data, 0, sizeof(window->data));

  window->type                  = WINDOW_TYPE_SUBWINDOW_TOOLTIP;
  window->visibility_mask       = WINDOW_VISIBILITY_UTILITIES;
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
  _subwindow_tooltip_destroy_stack_trace(data);

  const size_t string_length = strlen(text);

  if (string_length >= SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH) {
    crash_message("Tooltip message is too large.");
  }

  memset(data->string, 0, SUBWINDOW_TOOLTIP_MAX_STRING_LENGTH);
  memcpy(data->string, text, string_length);

  window_allocate_memory(window);
}

void subwindow_tooltip_create_stack_trace(
  Window* window, Display* display, const char* text, uint32_t allocation_index, float wheel_motion, uint32_t x, uint32_t y) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  if (window->type != WINDOW_TYPE_SUBWINDOW_TOOLTIP)
    memset(window->data, 0, sizeof(window->data));

  if (text == (const char*) 0 || text[0] == '\0')
    text = "No stack trace available.";

  SubwindowTooltipData* data = (SubwindowTooltipData*) window->data;
  if (display->width < 2) {
    window->is_visible = false;
    return;
  }

  const uint32_t cursor_gap = display->width > 32 ? 16 : 1;
  uint32_t popup_width      = display->width > cursor_gap * 2 + 1 ? (display->width - cursor_gap * 2) / 2 : 1;
  if (popup_width > SUBWINDOW_TOOLTIP_STACK_TRACE_WIDTH)
    popup_width = SUBWINDOW_TOOLTIP_STACK_TRACE_WIDTH;
  const uint32_t max_text_width = popup_width > 48 ? popup_width - 48 : 1;
  const bool trace_changed =
    data->is_stack_trace == false || data->stack_trace_text == (char*) 0 || strcmp(data->stack_trace_text, text) != 0;
  const bool width_changed      = data->stack_trace_text_max_width != max_text_width;
  const bool allocation_changed = data->is_stack_trace == false || data->allocation_index != allocation_index;

  if (trace_changed)
    _subwindow_tooltip_destroy_stack_trace(data);

  if (trace_changed) {
    const size_t text_length = strlen(text);
    data->stack_trace_text   = (char*) malloc(text_length + 1);
    if (data->stack_trace_text == (char*) 0)
      crash_message("Failed to allocate stack trace text.");

    memcpy(data->stack_trace_text, text, text_length + 1);
  }
  else if (width_changed && data->stack_trace_rows) {
    free(data->stack_trace_rows);
    data->stack_trace_rows      = (SubwindowTooltipStackTraceRow*) 0;
    data->stack_trace_row_count = 0;
    data->first_visible_row     = 0.0f;
  }

  if (trace_changed || width_changed) {
    _subwindow_tooltip_build_stack_trace_rows(data, display, max_text_width);
    data->stack_trace_text_max_width = max_text_width;
  }

  data->is_stack_trace    = true;
  data->visible_row_count = _subwindow_tooltip_visible_stack_trace_rows(display, data->stack_trace_row_count);
  data->allocation_index  = allocation_index;

  if (allocation_changed)
    data->first_visible_row = 0.0f;

  const float max_scroll =
    data->stack_trace_row_count > data->visible_row_count ? (float) (data->stack_trace_row_count - data->visible_row_count) : 0.0f;
  data->first_visible_row = fminf(max_scroll, fmaxf(0.0f, data->first_visible_row - wheel_motion));

  window->type                  = WINDOW_TYPE_SUBWINDOW_TOOLTIP;
  window->visibility_mask       = WINDOW_VISIBILITY_UTILITIES;
  window->width                 = popup_width;
  window->height                = 16 + 32 + (data->visible_row_count + 1) * SUBWINDOW_TOOLTIP_STACK_TRACE_ROW_HEIGHT;
  window->padding               = 8;
  window->is_horizontal         = false;
  window->is_visible            = true;
  window->is_movable            = false;
  window->background            = true;
  window->auto_size             = true;
  window->action_func           = _subwindow_tooltip_stack_trace_action;
  window->propagate_parent_func = _subwindow_tooltip_propagate_parent;
  window->fixed_depth           = true;
  window->depth                 = UINT64_MAX;
  window->is_subwindow          = true;

  const uint32_t right_limit = display->width > window->width + cursor_gap ? display->width - window->width - cursor_gap : 0;
  const bool place_to_right  = x <= right_limit;
  if (place_to_right) {
    window->x = (int32_t) (x + cursor_gap);
  }
  else if (x >= window->width + cursor_gap) {
    window->x = (int32_t) (x - window->width - cursor_gap);
  }
  else {
    window->x = 0;
  }

  const uint32_t max_y   = display->height > window->height ? display->height - window->height : 0;
  const uint32_t below_y = y + cursor_gap;
  if (below_y <= max_y)
    window->y = (int32_t) below_y;
  else if (y >= window->height + cursor_gap)
    window->y = (int32_t) (y - window->height - cursor_gap);
  else
    window->y = (int32_t) max_y;

  window_allocate_memory(window);
}
