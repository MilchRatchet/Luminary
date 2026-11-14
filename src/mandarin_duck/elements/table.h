#ifndef MANDARIN_DUCK_ELEMENTS_TABLE_H
#define MANDARIN_DUCK_ELEMENTS_TABLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "elements_common.h"

#define ELEMENT_TABLE_NO_SELECTION (~(uint32_t) 0)

enum ElementTableDataType {
  ELEMENT_TABLE_DATA_TYPE_STRING,
  ELEMENT_TABLE_DATA_TYPE_SIZE,
  ELEMENT_TABLE_DATA_TYPE_UINT8,
  ELEMENT_TABLE_DATA_TYPE_UINT16,
  ELEMENT_TABLE_DATA_TYPE_UINT32,
  ELEMENT_TABLE_DATA_TYPE_UINT64,
  ELEMENT_TABLE_DATA_TYPE_SINT8,
  ELEMENT_TABLE_DATA_TYPE_SINT16,
  ELEMENT_TABLE_DATA_TYPE_SINT32,
  ELEMENT_TABLE_DATA_TYPE_SINT64,
  ELEMENT_TABLE_DATA_TYPE_FLOAT,
  ELEMENT_TABLE_DATA_TYPE_DOUBLE,
  ELEMENT_TABLE_DATA_TYPE_BOOL,
  ELEMENT_TABLE_DATA_TYPE_CUSTOM,
} typedef ElementTableDataType;

/* Return false to render a format-error marker; output is truncated to the supplied buffer. */
typedef bool (*ElementTableFormatFunc)(char* text, size_t text_size, const void* data, size_t data_size);

struct ElementTableColumn {
  const char* name;
  size_t offset;
  ElementTableDataType data_type;
  float rel_width;
  size_t data_size;
  ElementTableFormatFunc format;
} typedef ElementTableColumn;

/* size controls the table width; height is derived from row_height and visible_row_count. */
struct ElementTableArgs {
  /* Unique and stable within the window; identifies an active scrollbar drag. */
  const char* identifier;
  const void* data;
  /* Byte stride between records. */
  size_t element_size;
  uint64_t element_count;
  uint32_t visible_row_count;
  const ElementTableColumn* columns;
  uint32_t column_count;
  ElementSize size;
  uint32_t row_height;
  /* Mutable row offset; fractional values allow smooth wheel and thumb movement. */
  float* first_visible_row;
  /* Optional absolute index of the hovered row, or ELEMENT_TABLE_NO_SELECTION. */
  uint32_t* hovered_index;
  /* Let the caller handle wheel input over data rows instead of scrolling this table. */
  bool defer_wheel_over_rows;
} typedef ElementTableArgs;

uint32_t element_table(Window* window, Display* display, const MouseState* mouse_state, ElementTableArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_TABLE_H */
