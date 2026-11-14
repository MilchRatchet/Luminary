#include "table.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "display.h"
#include "elements/text.h"
#include "text_renderer.h"
#include "ui_renderer.h"
#include "ui_renderer_utils.h"

#define ELEMENT_TABLE_TEXT_SIZE 256
#define ELEMENT_TABLE_CELL_PADDING 8
#define ELEMENT_TABLE_SCROLLBAR_WIDTH 14
#define ELEMENT_TABLE_SCROLLBAR_MIN_THUMB_HEIGHT 16

struct ElementTableRenderData {
  uint32_t header_height;
  uint32_t row_height;
  uint32_t visible_row_count;
  uint32_t track_x;
  uint32_t track_y;
  uint32_t track_width;
  uint32_t track_height;
  uint32_t thumb_y;
  uint32_t thumb_height;
  uint32_t hovered_row;
  bool has_scrollbar;
} typedef ElementTableRenderData;
static_assert(sizeof(ElementTableRenderData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

static void _element_table_render_func(Element* table, Display* display) {
  const ElementTableRenderData* data = (const ElementTableRenderData*) table->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, table->width, table->height, table->x, table->y, 0, MD_COLOR_BORDER, MD_COLOR_BLACK,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
  ui_renderer_render_rounded_box(
    display->ui_renderer, display, table->width, data->header_height, table->x, table->y, 0, MD_COLOR_BORDER, MD_COLOR_ACCENT_2,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  if (data->hovered_row != ELEMENT_TABLE_NO_SELECTION) {
    const uint32_t row_y = table->y + data->header_height + data->hovered_row * data->row_height;
    ui_renderer_render_rounded_box(
      display->ui_renderer, display, table->width - data->track_width, data->row_height, table->x, row_y, 0, MD_COLOR_BORDER,
      MD_COLOR_ACCENT_LIGHT_2, UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
  }

  for (uint32_t row = 0; row < data->visible_row_count; row++) {
    const uint32_t line_y = table->y + data->header_height + row * data->row_height;

    ui_renderer_render_rounded_box(
      display->ui_renderer, display, table->width, 1, table->x, line_y, 0, MD_COLOR_BORDER, MD_COLOR_BORDER,
      UI_RENDERER_BACKGROUND_MODE_OPAQUE);
  }

  if (data->has_scrollbar) {
    ui_renderer_render_rounded_box(
      display->ui_renderer, display, data->track_width, data->track_height, data->track_x, data->track_y, 0, MD_COLOR_BORDER,
      MD_COLOR_BLACK, UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
    ui_renderer_render_rounded_box(
      display->ui_renderer, display, data->track_width, data->thumb_height, data->track_x, data->thumb_y, 0, MD_COLOR_BORDER,
      MD_COLOR_ACCENT_LIGHT_2, UI_RENDERER_BACKGROUND_MODE_OPAQUE);
  }
}

static size_t _element_table_builtin_type_size(ElementTableDataType type) {
  switch (type) {
    case ELEMENT_TABLE_DATA_TYPE_STRING:
      return sizeof(const char*);
    case ELEMENT_TABLE_DATA_TYPE_SIZE:
      return sizeof(size_t);
    case ELEMENT_TABLE_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ELEMENT_TABLE_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case ELEMENT_TABLE_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case ELEMENT_TABLE_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    case ELEMENT_TABLE_DATA_TYPE_SINT8:
      return sizeof(int8_t);
    case ELEMENT_TABLE_DATA_TYPE_SINT16:
      return sizeof(int16_t);
    case ELEMENT_TABLE_DATA_TYPE_SINT32:
      return sizeof(int32_t);
    case ELEMENT_TABLE_DATA_TYPE_SINT64:
      return sizeof(int64_t);
    case ELEMENT_TABLE_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ELEMENT_TABLE_DATA_TYPE_DOUBLE:
      return sizeof(double);
    case ELEMENT_TABLE_DATA_TYPE_BOOL:
      return sizeof(bool);
    case ELEMENT_TABLE_DATA_TYPE_CUSTOM:
      return 0;
  }

  return 0;
}

static void _element_table_validate_column(const ElementTableColumn* column, size_t element_size) {
  const size_t field_size =
    (column->data_type == ELEMENT_TABLE_DATA_TYPE_CUSTOM) ? column->data_size : _element_table_builtin_type_size(column->data_type);

  if (field_size == 0 || column->offset > element_size || field_size > element_size - column->offset) {
    crash_message("Table column is outside the element size or has an invalid data type.");
  }

  if (column->data_type == ELEMENT_TABLE_DATA_TYPE_CUSTOM && column->format == (ElementTableFormatFunc) 0) {
    crash_message("Custom table columns require a formatter.");
  }

  if (isfinite(column->rel_width) == false || column->rel_width <= 0.0f) {
    crash_message("Table column width must be a positive finite value.");
  }
}

static void _element_table_format_size(char* text, size_t text_size, size_t size) {
  static const char* const units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};

  double formatted_size = (double) size;
  size_t unit_index     = 0;

  while (formatted_size >= 1024.0 && unit_index + 1 < sizeof(units) / sizeof(units[0])) {
    formatted_size /= 1024.0;
    unit_index++;
  }

  if (unit_index == 0)
    snprintf(text, text_size, "%zu %s", size, units[unit_index]);
  else
    snprintf(text, text_size, "%.2f %s", formatted_size, units[unit_index]);
}

static void _element_table_format_cell(
  char* text, size_t text_size, const unsigned char* record, const ElementTableColumn* column, size_t field_size) {
  const void* field = record + column->offset;

  text[0] = '\0';

  if (column->format != (ElementTableFormatFunc) 0) {
    if (column->format(text, text_size, field, field_size) == false) {
      snprintf(text, text_size, "<format error>");
    }

    text[text_size - 1] = '\0';
    return;
  }

  switch (column->data_type) {
    case ELEMENT_TABLE_DATA_TYPE_STRING: {
      const char* value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%.255s", value == (const char*) 0 ? "(null)" : value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_SIZE: {
      size_t value;
      memcpy(&value, field, sizeof(value));
      _element_table_format_size(text, text_size, value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_UINT8: {
      uint8_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%u", (unsigned int) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_UINT16: {
      uint16_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%u", (unsigned int) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_UINT32: {
      uint32_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%u", (unsigned int) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_UINT64: {
      uint64_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%llu", (unsigned long long) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_SINT8: {
      int8_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%d", (int) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_SINT16: {
      int16_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%d", (int) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_SINT32: {
      int32_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%d", value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_SINT64: {
      int64_t value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%lld", (long long) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_FLOAT: {
      float value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%.6g", (double) value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_DOUBLE: {
      double value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%.6g", value);
    } break;
    case ELEMENT_TABLE_DATA_TYPE_BOOL: {
      bool value;
      memcpy(&value, field, sizeof(value));
      snprintf(text, text_size, "%s", value ? "true" : "false");
    } break;
    case ELEMENT_TABLE_DATA_TYPE_CUSTOM:
      crash_message("Custom table column is missing its formatter.");
  }

  text[text_size - 1] = '\0';
}

static size_t _element_table_previous_utf8_boundary(const char* text, size_t length) {
  if (length == 0)
    return 0;

  size_t boundary = length - 1;
  while (boundary > 0 && (((unsigned char) text[boundary] & 0xC0) == 0x80))
    boundary--;

  return boundary;
}

static void _element_table_fit_text(Display* display, char* text, size_t text_size, uint32_t available_width) {
  uint32_t text_width;
  uint32_t text_height;
  text_renderer_compute_size(display->text_renderer, text, TEXT_RENDERER_FONT_REGULAR, false, &text_width, &text_height);

  if (text_width <= available_width)
    return;

  char original_text[ELEMENT_TABLE_TEXT_SIZE];
  const size_t original_length = strlen(text);
  memcpy(original_text, text, original_length + 1);

  size_t prefix_length = original_length;
  while (true) {
    while (prefix_length + 4 > text_size && prefix_length > 0)
      prefix_length = _element_table_previous_utf8_boundary(original_text, prefix_length);

    if (prefix_length + 4 <= text_size) {
      memcpy(text, original_text, prefix_length);
      memcpy(text + prefix_length, "...", 4);
      text_renderer_compute_size(display->text_renderer, text, TEXT_RENDERER_FONT_REGULAR, false, &text_width, &text_height);

      if (text_width <= available_width)
        return;
    }

    if (prefix_length == 0)
      break;

    prefix_length = _element_table_previous_utf8_boundary(original_text, prefix_length);
  }

  text[0] = '\0';
}

static void _element_table_push_row(
  Window* window, Display* display, const MouseState* mouse_state, uint32_t x, uint32_t y, uint32_t width, uint32_t row_height,
  const ElementTableColumn* columns, uint32_t column_count, float total_column_width, const unsigned char* record) {
  WindowContext* context            = window->context_stack + window->context_stack_ptr;
  const WindowContext saved_context = *context;
  const uint32_t cell_padding       = width >= 2 * ELEMENT_TABLE_CELL_PADDING ? ELEMENT_TABLE_CELL_PADDING : width / 2;

  context->x             = x + cell_padding;
  context->y             = y;
  context->width         = width - 2 * cell_padding;
  context->height        = row_height;
  context->padding       = 0;
  context->is_horizontal = true;
  context->fill          = 0;

  const uint32_t content_width = context->width;
  float used_width             = 0.0f;
  uint32_t used_pixel_width    = 0;

  for (uint32_t column_index = 0; column_index < column_count; column_index++) {
    const ElementTableColumn* column = columns + column_index;
    float rel_width                  = column->rel_width / total_column_width;

    if (column_index + 1 == column_count) {
      rel_width = 1.0f - used_width;
    }

    char cell_text[ELEMENT_TABLE_TEXT_SIZE];
    if (record == (const unsigned char*) 0) {
      snprintf(cell_text, sizeof(cell_text), "%.255s", column->name == (const char*) 0 ? "" : column->name);
    }
    else {
      const size_t field_size =
        (column->data_type == ELEMENT_TABLE_DATA_TYPE_CUSTOM) ? column->data_size : _element_table_builtin_type_size(column->data_type);
      _element_table_format_cell(cell_text, sizeof(cell_text), record, column, field_size);
    }

    uint32_t cell_width = (uint32_t) (content_width * rel_width);
    if (cell_width > content_width - used_pixel_width)
      cell_width = content_width - used_pixel_width;

    _element_table_fit_text(display, cell_text, sizeof(cell_text), cell_width);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {
        .text         = cell_text,
        .size         = (ElementSize) {.rel_width = rel_width, .height = row_height},
        .color        = 0xFFFFFFFF,
        .center_x     = false,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = record == (const unsigned char*) 0,
        .auto_size    = false,
        .is_clickable = false,
      });

    used_width += rel_width;
    used_pixel_width += cell_width;
  }

  *context = saved_context;
}

static uint32_t _element_table_compute_thumb_height(uint32_t track_height, uint32_t visible_row_count, uint64_t element_count) {
  uint64_t thumb_height = ((uint64_t) track_height * visible_row_count) / element_count;

  if (thumb_height < ELEMENT_TABLE_SCROLLBAR_MIN_THUMB_HEIGHT)
    thumb_height = ELEMENT_TABLE_SCROLLBAR_MIN_THUMB_HEIGHT;

  if (thumb_height > track_height)
    thumb_height = track_height;

  return (uint32_t) thumb_height;
}

uint32_t element_table(Window* window, Display* display, const MouseState* mouse_state, ElementTableArgs args) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(mouse_state);
  if (args.hovered_index)
    *args.hovered_index = ELEMENT_TABLE_NO_SELECTION;

  MD_CHECK_NULL_ARGUMENT(args.identifier);
  MD_CHECK_NULL_ARGUMENT(args.first_visible_row);

  if (args.column_count > 0 && args.columns == (const ElementTableColumn*) 0) {
    crash_message("Table column count is non-zero but the descriptor pointer is null.");
  }

  if (args.element_count > 0 && args.data == (const void*) 0) {
    crash_message("Table element count is non-zero but the data pointer is null.");
  }

  const size_t max_addressable_size = ~(size_t) 0;
  if (args.element_count > 0 && (args.element_size == 0 || args.element_count > max_addressable_size / args.element_size)) {
    crash_message("Table element count and stride exceed the addressable data size.");
  }

  if (args.element_count > ELEMENT_TABLE_NO_SELECTION) {
    crash_message("Table element count exceeds the range supported by its uint32_t selection result.");
  }

  const uint64_t max_row_height = ~(uint32_t) 0;
  const uint64_t total_rows     = (uint64_t) args.visible_row_count + 1;
  if (args.row_height == 0 || total_rows > max_row_height / args.row_height) {
    crash_message("Table row dimensions are invalid or too large.");
  }

  float total_column_width = 0.0f;

  for (uint32_t column_index = 0; column_index < args.column_count; column_index++) {
    _element_table_validate_column(args.columns + column_index, args.element_size);
    total_column_width += args.columns[column_index].rel_width;
  }

  if (args.column_count > 0 && (isfinite(total_column_width) == false || total_column_width <= 0.0f)) {
    crash_message("Table columns have an invalid combined width.");
  }

  ElementSize table_size = args.size;
  table_size.height      = (args.visible_row_count + 1) * args.row_height;
  table_size.rel_height  = 0.0f;

  Element table_bounds                  = {0};
  ElementMouseResult table_mouse_result = {0};
  WindowContext* parent_context         = window->context_stack + window->context_stack_ptr;
  element_apply_context(&table_bounds, parent_context, &table_size, mouse_state, &table_mouse_result);

  if (isfinite(*args.first_visible_row) == false)
    *args.first_visible_row = 0.0f;

  const bool is_scrollable = args.visible_row_count > 0 && args.element_count > args.visible_row_count;
  const uint32_t scrollbar_width =
    is_scrollable && table_bounds.width >= ELEMENT_TABLE_SCROLLBAR_WIDTH + 8 ? ELEMENT_TABLE_SCROLLBAR_WIDTH : 0;
  const uint32_t content_width = table_bounds.width - scrollbar_width;
  const uint32_t track_height  = args.visible_row_count * args.row_height;
  const uint32_t track_x       = table_bounds.x + content_width;
  const uint32_t track_y       = table_bounds.y + args.row_height;
  const uint32_t thumb_height =
    scrollbar_width > 0 ? _element_table_compute_thumb_height(track_height, args.visible_row_count, args.element_count) : 0;
  const uint32_t thumb_travel = track_height - thumb_height;
  const float max_scroll      = is_scrollable ? (float) (args.element_count - args.visible_row_count) : 0.0;
  const uint64_t table_hash   = element_compute_hash(window->separator_context_string, args.identifier);

  *args.first_visible_row = fminf(max_scroll, fmaxf(0.0f, *args.first_visible_row));

  uint32_t thumb_y = track_y;
  if (thumb_travel > 0 && max_scroll > 0.0f) {
    const float scroll_ratio = (float) (*args.first_visible_row / max_scroll);
    thumb_y += (uint32_t) floorf(scroll_ratio * thumb_travel + 0.5f);
  }

  const bool is_table_captured =
    window->state_data.state == WINDOW_INTERACTION_STATE_TABLE_SCROLL && window->state_data.element_hash == table_hash;

  const uint32_t data_y              = table_bounds.y + args.row_height;
  const uint32_t mouse_x             = mouse_state->x;
  const uint32_t mouse_y             = mouse_state->y;
  const bool any_row_visible         = args.visible_row_count > 0;
  const bool mouse_in_table_bounds_x = mouse_x >= table_bounds.x && mouse_x < table_bounds.x + content_width;
  const bool mouse_in_table_bounds_y = mouse_y >= data_y && mouse_y < data_y + track_height;
  const bool mouse_in_data_rows      = any_row_visible && mouse_in_table_bounds_x && mouse_in_table_bounds_y;

  if (table_mouse_result.is_hovered || is_table_captured) {
    window->element_has_hover     = true;
    window->status.received_hover = true;
  }

  if (
    is_scrollable && table_mouse_result.is_hovered && mouse_state->wheel_motion != 0.0f
    && (args.defer_wheel_over_rows == false || mouse_in_data_rows == false)) {
    *args.first_visible_row -= mouse_state->wheel_motion;
    window->status.received_scroll = true;
  }

  if (is_table_captured && mouse_state->down) {
    if (thumb_travel > 0) {
      *args.first_visible_row += (double) mouse_state->y_motion * max_scroll / thumb_travel;
    }

    window->status.received_mouse_action = true;
  }
  else if (
    scrollbar_width > 0 && table_mouse_result.is_hovered && mouse_state->phase == MOUSE_PHASE_PRESSED
    && window->state_data.state == WINDOW_INTERACTION_STATE_NONE) {
    const uint32_t mouse_y = (uint32_t) mouse_state->y;

    if (mouse_state->x >= track_x && mouse_state->x < track_x + scrollbar_width) {
      if (mouse_y >= thumb_y && mouse_y < thumb_y + thumb_height) {
        window->state_data.state        = WINDOW_INTERACTION_STATE_TABLE_SCROLL;
        window->state_data.element_hash = table_hash;
      }
      else if (mouse_y < thumb_y) {
        *args.first_visible_row -= args.visible_row_count;
      }
      else {
        *args.first_visible_row += args.visible_row_count;
      }

      window->status.received_mouse_action = true;
    }
  }

  *args.first_visible_row = fminf(max_scroll, fmaxf(0.0f, *args.first_visible_row));

  if (scrollbar_width > 0 && thumb_travel > 0 && max_scroll > 0.0) {
    const float scroll_ratio = (float) (*args.first_visible_row / max_scroll);
    thumb_y                  = track_y + (uint32_t) floorf(scroll_ratio * thumb_travel + 0.5f);
  }

  const uint64_t first_row = (uint64_t) floor(*args.first_visible_row);
  uint32_t hovered_row     = ELEMENT_TABLE_NO_SELECTION;
  uint32_t clicked_index   = ELEMENT_TABLE_NO_SELECTION;

  if (any_row_visible && mouse_in_table_bounds_x && mouse_in_table_bounds_y) {
    const uint32_t visible_row = (mouse_y - data_y) / args.row_height;
    const uint64_t row_index   = first_row + visible_row;

    if (row_index < args.element_count) {
      hovered_row = visible_row;
      if (args.hovered_index)
        *args.hovered_index = (uint32_t) row_index;

      if (mouse_state->phase == MOUSE_PHASE_RELEASED) {
        clicked_index                        = (uint32_t) row_index;
        window->status.received_mouse_action = true;
      }
    }
  }

  Element table     = {0};
  table.type        = ELEMENT_TYPE_TABLE;
  table.render_func = _element_table_render_func;
  table.hash        = table_hash;
  table.x           = table_bounds.x;
  table.y           = table_bounds.y;
  table.width       = table_bounds.width;
  table.height      = table_bounds.height;

  ElementTableRenderData* render_data = (ElementTableRenderData*) table.data;
  render_data->header_height          = args.row_height;
  render_data->row_height             = args.row_height;
  render_data->visible_row_count      = args.visible_row_count;
  render_data->track_x                = track_x;
  render_data->track_y                = track_y;
  render_data->track_width            = scrollbar_width;
  render_data->track_height           = track_height;
  render_data->thumb_y                = thumb_y;
  render_data->thumb_height           = thumb_height;
  render_data->hovered_row            = hovered_row;
  render_data->has_scrollbar          = scrollbar_width > 0;
  window_push_element(window, &table);

  _element_table_push_row(
    window, display, mouse_state, table_bounds.x, table_bounds.y, content_width, args.row_height, args.columns, args.column_count,
    total_column_width, (const unsigned char*) 0);

  for (uint32_t visible_row = 0; visible_row < args.visible_row_count; visible_row++) {
    const uint64_t row_index = first_row + visible_row;
    if (row_index >= args.element_count)
      break;

    const unsigned char* record = (const unsigned char*) args.data + (size_t) row_index * args.element_size;
    const uint32_t row_y        = table_bounds.y + args.row_height + visible_row * args.row_height;

    _element_table_push_row(
      window, display, mouse_state, table_bounds.x, row_y, content_width, args.row_height, args.columns, args.column_count,
      total_column_width, record);
  }

  return clicked_index;
}
