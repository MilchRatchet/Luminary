#include "debug_window.h"

#include <luminary/debug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "display.h"
#include "elements/dropdown.h"
#include "elements/separator.h"
#include "elements/table.h"
#include "elements/text.h"
#include "subwindow_tooltip.h"

#define WINDOW_DEBUG_DEVICE_LABEL_CACHE_CAPACITY 4
#define WINDOW_DEBUG_DEVICE_LABEL_SIZE 256

struct WindowDebugDeviceMemoryState {
  uint32_t device_id;
  float first_visible_row;
  LuminaryHost* cached_host;
  uint32_t cached_device_count;
  bool device_labels_valid;
  char device_labels[WINDOW_DEBUG_DEVICE_LABEL_CACHE_CAPACITY][WINDOW_DEBUG_DEVICE_LABEL_SIZE];
};
static_assert(sizeof(struct WindowDebugDeviceMemoryState) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static void _window_debug_populate_device_labels(
  LuminaryHost* host, uint32_t device_count, LuminaryDeviceInfo* device_info, char (*device_labels)[WINDOW_DEBUG_DEVICE_LABEL_SIZE],
  char** device_label_pointers) {
  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    LUM_FAILURE_HANDLE(luminary_host_get_device_info(host, device_id, &device_info[device_id]));
    bool duplicate_name = false;

    for (uint32_t previous_device_id = 0; previous_device_id < device_id; previous_device_id++) {
      if (strcmp(device_info[device_id].name, device_info[previous_device_id].name) == 0) {
        duplicate_name = true;
        break;
      }
    }

    if (duplicate_name) {
      snprintf(device_labels[device_id], sizeof(*device_labels), "%.230s (%u)", device_info[device_id].name, device_id);
    }
    else {
      snprintf(device_labels[device_id], sizeof(*device_labels), "%.255s", device_info[device_id].name);
    }

    device_label_pointers[device_id] = device_labels[device_id];
  }
}

static int _window_debug_compare_allocations(const void* left_value, const void* right_value) {
  const LuminaryHostMemoryAllocation* left_allocation  = (const LuminaryHostMemoryAllocation*) left_value;
  const LuminaryHostMemoryAllocation* right_allocation = (const LuminaryHostMemoryAllocation*) right_value;

  if (left_allocation->size > right_allocation->size)
    return -1;

  if (left_allocation->size < right_allocation->size)
    return 1;

  return 0;
}

static int _window_debug_compare_device_allocations(const void* left_value, const void* right_value) {
  const LuminaryDeviceMemoryAllocation* left_allocation  = (const LuminaryDeviceMemoryAllocation*) left_value;
  const LuminaryDeviceMemoryAllocation* right_allocation = (const LuminaryDeviceMemoryAllocation*) right_value;

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

static const ElementTableColumn _window_debug_device_memory_columns[] = {
  {
    .name      = "Allocation",
    .offset    = offsetof(LuminaryDeviceMemoryAllocation, name),
    .data_type = ELEMENT_TABLE_DATA_TYPE_STRING,
    .rel_width = 0.8f,
  },
  {
    .name      = "Size",
    .offset    = offsetof(LuminaryDeviceMemoryAllocation, size),
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

static void _window_debug_clear_stack_trace_popup(Window* window) {
  if (window->external_subwindow == (Window*) 0 || window->external_subwindow->type != WINDOW_TYPE_SUBWINDOW_TOOLTIP)
    return;

  subwindow_tooltip_destroy(window->external_subwindow);
  window->external_subwindow->is_visible = false;

  if (window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER)
    window->state_data.state = WINDOW_INTERACTION_STATE_NONE;
}

static void _window_debug_update_stack_trace_popup(
  Window* window, Display* display, const MouseState* mouse_state, uint32_t hovered_index, const char* stack_trace) {
  if (window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED)
    return;

  if (hovered_index != ELEMENT_TABLE_NO_SELECTION) {
    subwindow_tooltip_create_stack_trace(
      window->external_subwindow, display, stack_trace, hovered_index, mouse_state->wheel_motion, mouse_state->x, mouse_state->y);
    window->state_data = (WindowInteractionStateData) {.state = WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER};
    window->status.received_scroll |= mouse_state->wheel_motion != 0.0f;
  }
  else {
    _window_debug_clear_stack_trace_popup(window);
  }
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
    _window_debug_clear_stack_trace_popup(window);
    window_push_section(window, 24, 0);
    window_margin(window, 8);
    _window_debug_add_text(window, display, mouse_state, "No host allocations", 1.0f);
    window_pop_section(window);
  }
  else {
    uint32_t hovered_index = ELEMENT_TABLE_NO_SELECTION;
    float first_visible_row;
    memcpy(&first_visible_row, window->data, sizeof(first_visible_row));

    element_table(
      window, display, mouse_state,
      (ElementTableArgs) {
        .identifier            = "host_memory_allocations",
        .data                  = allocation_list.allocations,
        .element_size          = sizeof(LuminaryHostMemoryAllocation),
        .element_count         = allocation_list.count,
        .visible_row_count     = 10,
        .columns               = _window_debug_host_memory_columns,
        .column_count          = sizeof(_window_debug_host_memory_columns) / sizeof(_window_debug_host_memory_columns[0]),
        .size                  = (ElementSize) {.rel_width = 1.0f},
        .row_height            = 24,
        .first_visible_row     = &first_visible_row,
        .hovered_index         = &hovered_index,
        .defer_wheel_over_rows = true,
      });

    memcpy(window->data, &first_visible_row, sizeof(first_visible_row));

    const char* stack_trace =
      hovered_index < allocation_list.count ? allocation_list.allocations[hovered_index].stack_trace : (const char*) 0;
    _window_debug_update_stack_trace_popup(window, display, mouse_state, hovered_index, stack_trace);
  }

  LUM_FAILURE_HANDLE(luminary_host_memory_allocation_list_destroy(&allocation_list));
}

static void _window_debug_device_memory_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  _window_debug_action(window, display, host, mouse_state, WINDOW_DEBUG_TYPE_DEVICE_MEMORY);

  uint32_t device_count;
  LUM_FAILURE_HANDLE(luminary_host_get_device_count(host, &device_count));

  if (device_count == 0) {
    _window_debug_clear_stack_trace_popup(window);
    window_push_section(window, 24, 0);
    window_margin(window, 8);
    _window_debug_add_text(window, display, mouse_state, "No devices available", 1.0f);
    window_pop_section(window);
    return;
  }

  if (device_count > WINDOW_DEBUG_DEVICE_LABEL_CACHE_CAPACITY)
    crash_message("Too many devices.");

  struct WindowDebugDeviceMemoryState state;
  memcpy(&state, window->data, sizeof(state));

  const bool cache_key_changed = state.device_labels_valid && (state.cached_host != host || state.cached_device_count != device_count);

  if (state.device_id >= device_count || cache_key_changed) {
    state.device_id         = 0;
    state.first_visible_row = 0.0;
  }

  LuminaryDeviceInfo cached_device_info[WINDOW_DEBUG_DEVICE_LABEL_CACHE_CAPACITY];
  char* cached_device_label_pointers[WINDOW_DEBUG_DEVICE_LABEL_CACHE_CAPACITY];
  LuminaryDeviceInfo* device_info                       = cached_device_info;
  char (*device_labels)[WINDOW_DEBUG_DEVICE_LABEL_SIZE] = state.device_labels;
  char** device_label_pointers                          = cached_device_label_pointers;

  if (state.device_labels_valid == false || state.cached_host != host || state.cached_device_count != device_count) {
    _window_debug_populate_device_labels(host, device_count, device_info, device_labels, device_label_pointers);
    state.cached_host         = host;
    state.cached_device_count = device_count;
    state.device_labels_valid = true;
  }
  else {
    for (uint32_t device_id = 0; device_id < device_count; device_id++)
      device_label_pointers[device_id] = device_labels[device_id];
  }

  const uint32_t previous_device_id = state.device_id;
  window_push_section(window, 32, 0);
  window_margin(window, 8);
  _window_debug_add_text(window, display, mouse_state, "Device", 0.35f);
  element_dropdown(
    window, display, mouse_state,
    (ElementDropdownArgs) {
      .identifier     = "device_memory_device",
      .size           = (ElementSize) {.rel_width = 0.65f, .height = 24},
      .selected_index = &state.device_id,
      .num_strings    = device_count,
      .strings        = device_label_pointers,
      .write_access   = true,
    });
  window_pop_section(window);

  if (state.device_id != previous_device_id)
    state.first_visible_row = 0.0;

  memcpy(window->data, &state, sizeof(state));

  LuminaryDeviceMemoryAllocationList allocation_list = {0};
  LUM_FAILURE_HANDLE(luminary_host_device_memory_get_allocations(host, state.device_id, &allocation_list));

  uint64_t total_size = 0;
  for (uint64_t allocation_index = 0; allocation_index < allocation_list.count; allocation_index++)
    total_size += (uint64_t) allocation_list.allocations[allocation_index].size;

  if (allocation_list.count > 1) {
    qsort(
      allocation_list.allocations, (size_t) allocation_list.count, sizeof(LuminaryDeviceMemoryAllocation),
      _window_debug_compare_device_allocations);
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
    _window_debug_clear_stack_trace_popup(window);
    window_push_section(window, 24, 0);
    window_margin(window, 8);
    _window_debug_add_text(window, display, mouse_state, "No device allocations", 1.0f);
    window_pop_section(window);
  }
  else {
    uint32_t hovered_index = ELEMENT_TABLE_NO_SELECTION;
    element_table(
      window, display, mouse_state,
      (ElementTableArgs) {
        .identifier            = "device_memory_allocations",
        .data                  = allocation_list.allocations,
        .element_size          = sizeof(LuminaryDeviceMemoryAllocation),
        .element_count         = allocation_list.count,
        .visible_row_count     = 10,
        .columns               = _window_debug_device_memory_columns,
        .column_count          = sizeof(_window_debug_device_memory_columns) / sizeof(_window_debug_device_memory_columns[0]),
        .size                  = (ElementSize) {.rel_width = 1.0f},
        .row_height            = 24,
        .first_visible_row     = &state.first_visible_row,
        .hovered_index         = &hovered_index,
        .defer_wheel_over_rows = true,
      });

    memcpy(window->data, &state, sizeof(state));

    const char* stack_trace =
      hovered_index < allocation_list.count ? allocation_list.allocations[hovered_index].stack_trace : (const char*) 0;
    _window_debug_update_stack_trace_popup(window, display, mouse_state, hovered_index, stack_trace);
  }

  LUM_FAILURE_HANDLE(luminary_host_device_memory_allocation_list_destroy(&allocation_list));
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
