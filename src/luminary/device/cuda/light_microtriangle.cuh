#ifndef CU_LUMINARY_LIGHT_MICROTRIANGLE_H
#define CU_LUMINARY_LIGHT_MICROTRIANGLE_H

#include "math.cuh"
#include "utils.cuh"

static_assert(LIGHT_NUM_MICROTRIANGLES == 64, "This code requires this specific number of microtriangles.");
LUMINARY_FUNCTION void light_microtriangle_id_to_bary(const uint32_t id, float2& bary0, float2& bary1, float2& bary2) {
  uint32_t row_id;
  uint32_t col_id;

  // Row 0
  if (id <= 15) {
    row_id = 0;
    col_id = (id >> 1);
  }
  // Row 1
  else if (id <= 15 + 13) {
    row_id = 1;
    col_id = ((id - 15) >> 1);
  }
  // Row 2
  else if (id <= 15 + 13 + 11) {
    row_id = 2;
    col_id = ((id - 15 - 13) >> 1);
  }
  // Row 3
  else if (id <= 15 + 13 + 11 + 9) {
    row_id = 3;
    col_id = ((id - 15 - 13 - 11) >> 1);
  }
  // Row 4
  else if (id <= 15 + 13 + 11 + 9 + 7) {
    row_id = 4;
    col_id = ((id - 15 - 13 - 11 - 9) >> 1);
  }
  // Row 5
  else if (id <= 15 + 13 + 11 + 9 + 7 + 5) {
    row_id = 5;
    col_id = ((id - 15 - 13 - 11 - 9 - 7) >> 1);
  }
  // Row 6
  else if (id <= 15 + 13 + 11 + 9 + 7 + 5 + 3) {
    row_id = 6;
    col_id = ((id - 15 - 13 - 11 - 9 - 7 - 5) >> 1);
  }
  // Row 7
  else {
    row_id = 7;
    col_id = 0;
  }

  const bool is_top = (id & 0b1) == (row_id & 0b1);

  bary0 = make_float2(row_id, col_id + 1);
  bary1 = make_float2(row_id + 1, col_id);
  bary2 = is_top ? make_float2(row_id, col_id) : make_float2(row_id + 1, col_id + 1);

  // Normalize
  bary0.x *= 1.0f / 8.0f;
  bary0.y *= 1.0f / 8.0f;
  bary1.x *= 1.0f / 8.0f;
  bary1.y *= 1.0f / 8.0f;
  bary2.x *= 1.0f / 8.0f;
  bary2.y *= 1.0f / 8.0f;
}

LUMINARY_FUNCTION uint32_t light_microtriangle_bary_to_id(const float2 bary) {
  const uint32_t index_row = min((uint32_t) (fmaxf(bary.x, 0.0f) * 8.0f), 7);
  const float row_mod      = bary.x * 8.0f - index_row;

  // This is funny, we need to scale bary.y into [0,1] range (because bary.y has a smaller range in higher rows),
  // then we need to map the [0,1] to the column. In doing all of this, the number of cols cancel out and we
  // just need to scale by 8 regardless of the row we are in.
  const uint32_t index_col = min((uint32_t) (fmaxf(bary.y, 0.0f) * 8.0f), 7);
  const float col_mod      = bary.y * 8.0f - index_col;

  // TODO: There is definitely a branch-less way of doing this.

  uint32_t offset;
  switch (index_row) {
    default:
    case 0:
      offset = 0;
      break;
    case 1:
      offset = 15;
      break;
    case 2:
      offset = 15 + 13;
      break;
    case 3:
      offset = 15 + 13 + 11;
      break;
    case 4:
      offset = 15 + 13 + 11 + 9;
      break;
    case 5:
      offset = 15 + 13 + 11 + 9 + 7;
      break;
    case 6:
      offset = 15 + 13 + 11 + 9 + 7 + 5;
      break;
    case 7:
      offset = 15 + 13 + 11 + 9 + 7 + 5 + 3;
      break;
  }

  return offset + (index_col * 2 + (((row_mod + col_mod) > 1.0f) ? 1 : 0));
}

#endif /* CU_LUMINARY_LIGHT_MICROTRIANGLE_H */
