#include "debug_window.h"

#include <luminary/debug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "display.h"
#include "elements/separator.h"
#include "elements/table.h"
#include "elements/text.h"

static int _window_debug_compare_allocations(const void* left_value, const void* right_value) {
  const LuminaryHostMemoryAllocation* left_allocation  = (const LuminaryHostMemoryAllocation*) left_value;
  const LuminaryHostMemoryAllocation* right_allocation = (const LuminaryHostMemoryAllocation*) right_value;

  if (left_allocation->size > right_allocation->size)
    return -1;

  if (left_allocation->size < right_allocation->size)
    return 1;

  return 0;
}

static void _window_debug_format_size(char* text, size_t text_size, uint64_t size) {
  static const char* const units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};

  double formatted_size = (double) size;
  size_t unit_index     = 0;

  while (formatted_size >= 1024.0 && unit_index + 1 < sizeof(units) / sizeof(units[0])) {
    formatted_size /= 1024.0;
    unit_index++;
  }

  if (unit_index == 0)
    snprintf(text, text_size, "%llu %s", (unsigned long long) size, units[unit_index]);
  else
    snprintf(text, text_size, "%.2f %s", formatted_size, units[unit_index]);
}

static const ElementTableColumn _window_debug_host_memory_columns[] = {
  {
    .name      = "Allocation",
    .offset    = offsetof(LuminaryHostMemoryAllocation, name),
    .data_type = ELEMENT_TABLE_DATA_TYPE_STRING,
    .rel_width = 0.8f,
  },
  {
    .name      = "Size",
    .offset    = offsetof(LuminaryHostMemoryAllocation, size),
    .data_type = ELEMENT_TABLE_DATA_TYPE_SIZE,
    .rel_width = 0.2f,
  },
};

static void _window_debug_add_text(Window* window, Display* display, const MouseState* mouse_state, const char* text, float rel_width) {
  element_text(
    window, display, mouse_state,
    (ElementTextArgs) {
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize) {.rel_width = rel_width, .height = 24},
      .text         = text,
      .center_x     = false,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = false,
      .auto_size    = false,
      .is_clickable = false,
    });
}

static const char* const _window_debug_type_names[WINDOW_DEBUG_TYPE_COUNT] = {
  [WINDOW_DEBUG_TYPE_HOST_MEMORY]   = "Host Memory",
  [WINDOW_DEBUG_TYPE_DEVICE_MEMORY] = "Device Memory",
};

static void _window_debug_action(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state, WindowDebugType type) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  element_separator(
    window, mouse_state,
    (ElementSeparatorArgs) {.text = window_debug_type_get_name(type), .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});
}

static void _window_debug_host_memory_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  _window_debug_action(window, display, host, mouse_state, WINDOW_DEBUG_TYPE_HOST_MEMORY);

  LuminaryHostMemoryAllocationList allocation_list = {0};
  LUM_FAILURE_HANDLE(luminary_host_memory_get_allocations(&allocation_list));

  uint64_t total_size = 0;
  for (uint64_t allocation_index = 0; allocation_index < allocation_list.count; allocation_index++)
    total_size += (uint64_t) allocation_list.allocations[allocation_index].size;

  if (allocation_list.count > 1) {
    qsort(
      allocation_list.allocations, (size_t) allocation_list.count, sizeof(LuminaryHostMemoryAllocation), _window_debug_compare_allocations);
  }

  char count_text[64];
  snprintf(count_text, sizeof(count_text), "Allocations: %llu", (unsigned long long) allocation_list.count);

  char total_text[64];
  _window_debug_format_size(total_text, sizeof(total_text), total_size);

  window_push_section(window, 24, 0);
  window_margin(window, 8);
  _window_debug_add_text(window, display, mouse_state, count_text, 0.5f);
  char summary_total_text[80];
  snprintf(summary_total_text, sizeof(summary_total_text), "Total: %s", total_text);
  _window_debug_add_text(window, display, mouse_state, summary_total_text, 0.5f);
  window_pop_section(window);

  if (allocation_list.count == 0) {
    window_push_section(window, 24, 0);
    window_margin(window, 8);
    _window_debug_add_text(window, display, mouse_state, "No host allocations", 1.0f);
    window_pop_section(window);
  }
  else {
    double first_visible_row;
    memcpy(&first_visible_row, window->data, sizeof(first_visible_row));

    element_table(
      window, display, mouse_state,
      (ElementTableArgs) {
        .identifier        = "host_memory_allocations",
        .data              = allocation_list.allocations,
        .element_size      = sizeof(LuminaryHostMemoryAllocation),
        .element_count     = allocation_list.count,
        .visible_row_count = 10,
        .columns           = _window_debug_host_memory_columns,
        .column_count      = sizeof(_window_debug_host_memory_columns) / sizeof(_window_debug_host_memory_columns[0]),
        .size              = (ElementSize) {.rel_width = 1.0f},
        .row_height        = 24,
        .first_visible_row = &first_visible_row,
      });

    memcpy(window->data, &first_visible_row, sizeof(first_visible_row));
  }

  LUM_FAILURE_HANDLE(luminary_host_memory_allocation_list_destroy(&allocation_list));
}

static void _window_debug_device_memory_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  _window_debug_action(window, display, host, mouse_state, WINDOW_DEBUG_TYPE_DEVICE_MEMORY);
}

static void (*const _window_debug_action_funcs[WINDOW_DEBUG_TYPE_COUNT])(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) = {
  [WINDOW_DEBUG_TYPE_HOST_MEMORY]   = _window_debug_host_memory_action,
  [WINDOW_DEBUG_TYPE_DEVICE_MEMORY] = _window_debug_device_memory_action,
};

const char* window_debug_type_get_name(WindowDebugType type) {
  if (type >= WINDOW_DEBUG_TYPE_COUNT) {
    crash_message("Invalid debug window type was passed.");
  }

  return _window_debug_type_names[type];
}

void window_debug_create(Window** window, WindowDebugType type) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (type >= WINDOW_DEBUG_TYPE_COUNT) {
    crash_message("Invalid debug window type was passed.");
  }

  window_create(window);

  (*window)->type            = WINDOW_TYPE_DEBUG;
  (*window)->visibility_mask = WINDOW_VISIBILITY_UTILITIES;
  (*window)->x               = 192 + (((uint32_t) type) * 64);
  (*window)->y               = 192 + (((uint32_t) type) * 64);
  (*window)->width           = 512;
  (*window)->height          = 128;
  (*window)->padding         = 8;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = false;
  (*window)->is_movable      = true;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->action_func     = _window_debug_action_funcs[type];
  (*window)->fixed_depth     = false;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}
