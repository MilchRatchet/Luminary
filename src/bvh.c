#include "bvh.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "mesh.h"
#include "primitives.h"
#include "utils.h"

#define THRESHOLD_TRIANGLES 1
#define OBJECT_SPLIT_BIN_COUNT 32
#define SPATIAL_SPLIT_THRESHOLD 0.00001f
#define SPATIAL_SPLIT_BIN_COUNT 32
#define COST_OF_TRIANGLE 0.4f
#define COST_OF_NODE 1.0f

#define TRIANGLES_MAX 3

#define BINARY_NODE_IS_INTERNAL_NODE 0b0
#define BINARY_NODE_IS_LEAF_NODE 0b1
#define BINARY_NODE_IS_NULL 0b10

struct vec3_p {
  float x;
  float y;
  float z;
  float _p;
} typedef vec3_p;

struct fragment {
  vec3_p high;
  vec3_p low;
  vec3_p middle;
  uint32_t id;
  uint32_t _p2;
  uint64_t _p3;
} typedef fragment;

struct bin {
  vec3_p high;
  vec3_p low;
  int32_t entry;
  int32_t exit;
  uint64_t _p;
} typedef bin;

static float get_entry_by_axis(const vec3_p p, const int axis) {
  return (axis) ? ((axis == 1) ? p.y : p.z) : p.x;
}

static void __swap_fragments(fragment* a, fragment* b) {
  fragment temp = *a;
  *a            = *b;
  *b            = temp;
}

static int __partition(fragment* fragments, int bottom, int top, const int axis) {
  const int mid = (top - bottom) / 2 + bottom;
  if (get_entry_by_axis(fragments[top].middle, axis) < get_entry_by_axis(fragments[bottom].middle, axis)) {
    __swap_fragments(fragments + bottom, fragments + top);
  }
  if (get_entry_by_axis(fragments[mid].middle, axis) < get_entry_by_axis(fragments[bottom].middle, axis)) {
    __swap_fragments(fragments + mid, fragments + bottom);
  }
  if (get_entry_by_axis(fragments[top].middle, axis) > get_entry_by_axis(fragments[mid].middle, axis)) {
    __swap_fragments(fragments + mid, fragments + top);
  }

  const float x = get_entry_by_axis(fragments[top].middle, axis);
  int i         = bottom - 1;

  for (int j = bottom; j < top; j++) {
    if (get_entry_by_axis(fragments[j].middle, axis) < x) {
      i++;
      __swap_fragments(fragments + i, fragments + j);
    }
  }
  __swap_fragments(fragments + i + 1, fragments + top);

  return (i + 1);
}

static void quick_sort_fragments(fragment* fragments, const unsigned int fragments_length, const int axis) {
  int ptr = 1;

  int quick_sort_stack[64];

  quick_sort_stack[0] = 0;
  quick_sort_stack[1] = fragments_length - 1;

  while (ptr >= 0) {
    const int top    = quick_sort_stack[ptr--];
    const int bottom = quick_sort_stack[ptr--];

    const int p = __partition(fragments, bottom, top, axis);

    if (p - 1 > bottom) {
      quick_sort_stack[++ptr] = bottom;
      quick_sort_stack[++ptr] = p - 1;
    }

    if (p + 1 < top) {
      quick_sort_stack[++ptr] = p + 1;
      quick_sort_stack[++ptr] = top;
    }
  }
}

static void fit_bounds(const fragment* fragments, const unsigned int fragments_length, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_setr_ps(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  __m128 low  = _mm_setr_ps(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (unsigned int i = 0; i < fragments_length; i++) {
    const float* baseptr = (float*) (fragments + i);

    __m128 high_frag = _mm_load_ps(baseptr);
    __m128 low_frag  = _mm_load_ps(baseptr + 4);

    high = _mm_max_ps(high, high_frag);
    low  = _mm_min_ps(low, low_frag);
  }

  if (high_out)
    _mm_storeu_ps((float*) high_out, high);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low);
}

static void fit_bounds_of_bins(const bin* bins, const int bins_length, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_setr_ps(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  __m128 low  = _mm_setr_ps(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (int i = 0; i < bins_length; i++) {
    const float* baseptr = (float*) (bins + i);

    __m128 high_bin = _mm_loadu_ps(baseptr);
    __m128 low_bin  = _mm_loadu_ps(baseptr + 4);

    high = _mm_max_ps(high, high_bin);
    low  = _mm_min_ps(low, low_bin);
  }

  if (high_out)
    _mm_storeu_ps((float*) high_out, high);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low);
}

static void update_bounds_of_bins(const bin* bins, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_loadu_ps((float*) high_out);
  __m128 low  = _mm_loadu_ps((float*) low_out);

  const float* baseptr = (float*) (bins);

  __m128 high_bin = _mm_loadu_ps(baseptr);
  __m128 low_bin  = _mm_loadu_ps(baseptr + 4);

  high = _mm_max_ps(high, high_bin);
  low  = _mm_min_ps(low, low_bin);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low);
}

static float construct_bins(
  bin* restrict bins, const fragment* restrict fragments, const unsigned int fragments_length, const int axis, float* offset) {
  vec3_p high, low;
  fit_bounds(fragments, fragments_length, &high, &low);
  const float span     = get_entry_by_axis(high, axis) - get_entry_by_axis(low, axis);
  const float interval = span / OBJECT_SPLIT_BIN_COUNT;

  if (interval <= FLT_EPSILON * get_entry_by_axis(low, axis))
    return 0.0f;

  *offset = get_entry_by_axis(low, axis);

  for (int i = 0; i < OBJECT_SPLIT_BIN_COUNT; i++) {
    bin b = {
      .high.x = -FLT_MAX,
      .high.y = -FLT_MAX,
      .high.z = -FLT_MAX,
      .low.x  = FLT_MAX,
      .low.y  = FLT_MAX,
      .low.z  = FLT_MAX,
      .entry  = 0,
      .exit   = 0};
    bins[i] = b;
  }

  for (unsigned int i = 0; i < fragments_length; i++) {
    fragment frag     = fragments[i];
    const float value = get_entry_by_axis(frag.middle, axis);
    int pos           = 0;
    while ((pos + 1) * interval + get_entry_by_axis(low, axis) < value) {
      pos++;
    }

    if (pos == OBJECT_SPLIT_BIN_COUNT)
      pos--;

    bin b = bins[pos];
    b.entry++;
    b.exit++;

    b.high.x  = max(b.high.x, frag.high.x);
    b.high.y  = max(b.high.y, frag.high.y);
    b.high.z  = max(b.high.z, frag.high.z);
    b.high._p = max(b.high._p, frag.high._p);
    b.low.x   = min(b.low.x, frag.low.x);
    b.low.y   = min(b.low.y, frag.low.y);
    b.low.z   = min(b.low.z, frag.low.z);
    b.low._p  = min(b.low._p, frag.low._p);
    bins[pos] = b;
  }

  return interval;
}

static float construct_chopped_bins(
  bin* restrict bins, const fragment* restrict fragments, const unsigned int fragments_length, const int axis, float* offset) {
  vec3_p high, low;
  fit_bounds(fragments, fragments_length, &high, &low);
  const float span     = get_entry_by_axis(high, axis) - get_entry_by_axis(low, axis);
  const float interval = span / SPATIAL_SPLIT_BIN_COUNT;

  if (interval <= FLT_EPSILON * get_entry_by_axis(low, axis))
    return 0.0f;

  *offset = get_entry_by_axis(low, axis);

  for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
    bin b = {
      .high.x = -FLT_MAX,
      .high.y = -FLT_MAX,
      .high.z = -FLT_MAX,
      .low.x  = FLT_MAX,
      .low.y  = FLT_MAX,
      .low.z  = FLT_MAX,
      .entry  = 0,
      .exit   = 0};
    bins[i] = b;
  }

  for (unsigned int i = 0; i < fragments_length; i++) {
    vec3_p high_triangle = fragments[i].high;
    vec3_p low_triangle  = fragments[i].low;

    const float value1 = get_entry_by_axis(low_triangle, axis);
    int pos1           = 0;
    while ((pos1 + 1) * interval + get_entry_by_axis(low, axis) < value1) {
      pos1++;
    }
    const float value2 = get_entry_by_axis(high_triangle, axis);
    int pos2           = 0;
    while ((pos2 + 1) * interval + get_entry_by_axis(low, axis) < value2) {
      pos2++;
    }

    const int entry = min(pos1, pos2);
    int exit        = max(pos1, pos2);

    if (exit == SPATIAL_SPLIT_BIN_COUNT)
      exit--;

    for (int j = entry; j <= exit; j++) {
      bin b = bins[j];
      if (j == entry)
        b.entry++;
      if (j == exit)
        b.exit++;

      b.high.x  = max(b.high.x, high_triangle.x);
      b.high.y  = max(b.high.y, high_triangle.y);
      b.high.z  = max(b.high.z, high_triangle.z);
      b.high._p = max(b.high._p, high_triangle._p);
      b.low.x   = min(b.low.x, low_triangle.x);
      b.low.y   = min(b.low.y, low_triangle.y);
      b.low.z   = min(b.low.z, low_triangle.z);
      b.low._p  = min(b.low._p, low_triangle._p);
      bins[j]   = b;
    }
  }

  for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
    bin b = bins[i];
    if (axis == 0) {
      b.low.x  = max(b.low.x, low.x + i * interval);
      b.high.x = min(b.high.x, low.x + (i + 1) * interval);
    }
    else if (axis == 1) {
      b.low.y  = max(b.low.y, low.y + i * interval);
      b.high.y = min(b.high.y, low.y + (i + 1) * interval);
    }
    else {
      b.low.z  = max(b.low.z, low.z + i * interval);
      b.high.z = min(b.high.z, low.z + (i + 1) * interval);
    }
    bins[i] = b;
  }

  return interval;
}

static void divide_middles_along_axis(
  const float split, const int axis, fragment* fragments_out, fragment* fragments_in, const unsigned int fragments_length,
  const int right_offset) {
  int left  = 0;
  int right = 0;

  for (unsigned int i = 0; i < fragments_length; i++) {
    fragment frag = fragments_in[i];

    const float middle = get_entry_by_axis(frag.middle, axis);

    if (middle <= split) {
      fragments_out[left++] = frag;
    }
    else {
      fragments_out[right_offset + right++] = frag;
    }
  }
}

static void divide_along_axis(
  const float split, const int axis, fragment* fragments_out, fragment* fragments_in, const unsigned int fragments_length,
  const int right_offset) {
  int left  = 0;
  int right = 0;

  for (unsigned int i = 0; i < fragments_length; i++) {
    const float low  = get_entry_by_axis(fragments_in[i].low, axis);
    const float high = get_entry_by_axis(fragments_in[i].high, axis);

    if (low <= split) {
      fragment frag_left = fragments_in[i];
      if (axis == 0) {
        frag_left.high.x   = min(frag_left.high.x, split);
        frag_left.middle.x = (frag_left.low.x + frag_left.high.x) / 2.0f;
      }
      else if (axis == 1) {
        frag_left.high.y   = min(frag_left.high.y, split);
        frag_left.middle.y = (frag_left.low.y + frag_left.high.y) / 2.0f;
      }
      else {
        frag_left.high.z   = min(frag_left.high.z, split);
        frag_left.middle.z = (frag_left.low.z + frag_left.high.z) / 2.0f;
      }
      fragments_out[left++] = frag_left;
    }
    if (high > split) {
      fragment frag_right = fragments_in[i];
      if (axis == 0) {
        frag_right.low.x    = max(frag_right.low.x, split);
        frag_right.middle.x = (frag_right.low.x + frag_right.high.x) / 2.0f;
      }
      else if (axis == 1) {
        frag_right.low.y    = max(frag_right.low.y, split);
        frag_right.middle.y = (frag_right.low.y + frag_right.high.y) / 2.0f;
      }
      else {
        frag_right.low.z    = max(frag_right.low.z, split);
        frag_right.middle.z = (frag_right.low.z + frag_right.high.z) / 2.0f;
      }
      fragments_out[right_offset + right++] = frag_right;
    }
  }
}

Node2* build_bvh_structure(Triangle** triangles_io, unsigned int* triangles_length_io, unsigned int* nodes_length_out) {
  bench_tic();
  unsigned int triangles_length = *triangles_length_io;
  Triangle* triangles           = *triangles_io;
  unsigned int node_count       = 1 + triangles_length / THRESHOLD_TRIANGLES;

  const unsigned int initial_triangles_length = triangles_length;

  Node2* nodes = (Node2*) malloc(sizeof(Node2) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node2) * node_count);

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = triangles_length;
  nodes[0].kind              = BINARY_NODE_IS_LEAF_NODE;

  unsigned int* leaf_nodes     = malloc(sizeof(unsigned int) * node_count);
  unsigned int leaf_node_count = 0;

  fragment* fragments           = _mm_malloc(sizeof(fragment) * triangles_length, 64);
  unsigned int fragments_length = triangles_length;

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle t  = triangles[i];
    vec3_p high = {
      .x = max(t.vertex.x, max(t.vertex.x + t.edge1.x, t.vertex.x + t.edge2.x)),
      .y = max(t.vertex.y, max(t.vertex.y + t.edge1.y, t.vertex.y + t.edge2.y)),
      .z = max(t.vertex.z, max(t.vertex.z + t.edge1.z, t.vertex.z + t.edge2.z))};
    vec3_p low = {
      .x = min(t.vertex.x, min(t.vertex.x + t.edge1.x, t.vertex.x + t.edge2.x)),
      .y = min(t.vertex.y, min(t.vertex.y + t.edge1.y, t.vertex.y + t.edge2.y)),
      .z = min(t.vertex.z, min(t.vertex.z + t.edge1.z, t.vertex.z + t.edge2.z))};
    vec3_p middle = {.x = (high.x + low.x) / 2.0f, .y = (high.y + low.y) / 2.0f, .z = (high.z + low.z) / 2.0f};

    fragment frag = {.high = high, .low = low, .middle = middle, .id = i};
    fragments[i]  = frag;
  }

  fragment* fragments_buffer           = _mm_malloc(sizeof(fragment) * fragments_length * 2, 64);
  unsigned int fragments_buffer_length = fragments_length * 2;
  unsigned int fragments_buffer_count  = fragments_length;

  float root_surface_area;
  vec3_p high_root, low_root;
  {
    fit_bounds(fragments, fragments_length, &high_root, &low_root);
    vec3_p diff = {
      .x = high_root.x - low_root.x, .y = high_root.y - low_root.y, .z = high_root.z - low_root.z, ._p = high_root._p - low_root._p};

    root_surface_area = diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
  }

  bin* bins = (bin*) _mm_malloc(sizeof(bin) * max(OBJECT_SPLIT_BIN_COUNT, SPATIAL_SPLIT_BIN_COUNT), 32);

  unsigned int begin_of_current_nodes = 0;
  unsigned int end_of_current_nodes   = 1;
  unsigned int write_ptr              = 1;

  while (begin_of_current_nodes != end_of_current_nodes) {
    unsigned int fragments_ptr           = 0;
    unsigned int buffer_ptr              = 0;
    unsigned int leaf_nodes_in_iteration = 0;

    for (unsigned int node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes; node_ptr++) {
      Node2 node         = nodes[node_ptr];
      node.cost_computed = 0;

      if (fragments_ptr != node.triangles_address) {
        memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(fragment) * (node.triangles_address - fragments_ptr));
        buffer_ptr += node.triangles_address - fragments_ptr;
        fragments_ptr += node.triangles_address - fragments_ptr;
      }

      if (node.triangle_count <= THRESHOLD_TRIANGLES) {
        nodes[node_ptr].triangles_address = buffer_ptr;
        nodes[node_ptr].cost_computed     = 0;

        memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(fragment) * node.triangle_count);
        buffer_ptr += node.triangle_count;
        fragments_ptr += node.triangle_count;
        leaf_nodes[leaf_node_count + leaf_nodes_in_iteration++] = node_ptr;
        continue;
      }

      float parent_surface_area;
      {
        vec3_p high_parent, low_parent;
        fit_bounds(fragments + fragments_ptr, node.triangle_count, &high_parent, &low_parent);
        vec3 diff = {.x = high_parent.x - low_parent.x, .y = high_parent.y - low_parent.y, .z = high_parent.z - low_parent.z};

        parent_surface_area = diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
      }

      const float sequential_cost = COST_OF_TRIANGLE * node.triangle_count;

      vec3_p high, low;
      vec3_p optimal_high_right, optimal_low_right, optimal_high_left, optimal_low_left;
      float optimal_cost = FLT_MAX;
      int axis, optimal_split, optimal_method;
      float optimal_splitting_plane;
      int optimal_total_triangles;

      optimal_method = 0;

      for (int a = 0; a < 3; a++) {
        float low_split;
        const float interval = construct_bins(bins, fragments + fragments_ptr, node.triangle_count, a, &low_split);

        if (interval == 0.0f)
          continue;

        int left  = 0;
        int right = 0;

        for (int k = 0; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          right += bins[k].exit;
        }

        vec3_p high_left  = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX, ._p = -FLT_MAX};
        vec3_p high_right = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX, ._p = -FLT_MAX};
        vec3_p low_left   = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX, ._p = FLT_MAX};
        vec3_p low_right  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX, ._p = FLT_MAX};

        for (int k = 1; k < OBJECT_SPLIT_BIN_COUNT - 1; k++) {
          update_bounds_of_bins(bins + k - 1, &high_left, &low_left);
          fit_bounds_of_bins(bins + k, OBJECT_SPLIT_BIN_COUNT - k, &high_right, &low_right);

          vec3_p diff_left = {
            .x = high_left.x - low_left.x, .y = high_left.y - low_left.y, .z = high_left.z - low_left.z, ._p = high_left._p - low_left._p};

          vec3_p diff_right = {
            .x  = high_right.x - low_right.x,
            .y  = high_right.y - low_right.y,
            .z  = high_right.z - low_right.z,
            ._p = high_right._p - low_right._p};

          const float cost_L = diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;

          const float cost_R = diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

          left += bins[k - 1].entry;
          right -= bins[k - 1].exit;

          const float total_cost =
            COST_OF_NODE + COST_OF_TRIANGLE * cost_L / parent_surface_area * left + COST_OF_TRIANGLE * cost_R / parent_surface_area * right;

          if (total_cost < optimal_cost) {
            optimal_cost            = total_cost;
            optimal_split           = left;
            optimal_total_triangles = node.triangle_count;
            optimal_splitting_plane = low_split + k * interval;
            axis                    = a;
            optimal_high_left       = high_left;
            optimal_high_right      = high_right;
            optimal_low_left        = low_left;
            optimal_low_right       = low_right;
          }
        }
      }

      vec3_p overlap = {
        .x  = max(min(optimal_high_left.x, optimal_high_right.x) - max(optimal_low_left.x, optimal_low_right.x), 0.0f),
        .y  = max(min(optimal_high_left.y, optimal_high_right.y) - max(optimal_low_left.y, optimal_low_right.y), 0.0f),
        .z  = max(min(optimal_high_left.z, optimal_high_right.z) - max(optimal_low_left.z, optimal_low_right.z), 0.0f),
        ._p = max(min(optimal_high_left._p, optimal_high_right._p) - max(optimal_low_left._p, optimal_low_right._p), 0.0f),
      };

      const float o               = overlap.x * overlap.y + overlap.x * overlap.z + overlap.y * overlap.z;
      const int do_spatial_splits = ((o / root_surface_area) > SPATIAL_SPLIT_THRESHOLD);

      if (do_spatial_splits) {
        for (int a = 0; a < 3; a++) {
          float low_split;
          const float interval = construct_chopped_bins(bins, fragments + fragments_ptr, node.triangle_count, a, &low_split);

          if (interval == 0.0f)
            continue;

          int left  = 0;
          int right = 0;

          for (int k = 0; k < SPATIAL_SPLIT_BIN_COUNT; k++) {
            right += bins[k].exit;
          }

          vec3_p high_left  = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
          vec3_p high_right = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
          vec3_p low_left   = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};
          vec3_p low_right  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};

          for (int k = 1; k < SPATIAL_SPLIT_BIN_COUNT - 1; k++) {
            update_bounds_of_bins(bins + k - 1, &high_left, &low_left);
            fit_bounds_of_bins(bins + k, SPATIAL_SPLIT_BIN_COUNT - k, &high_right, &low_right);

            vec3_p diff_left = {
              .x  = high_left.x - low_left.x,
              .y  = high_left.y - low_left.y,
              .z  = high_left.z - low_left.z,
              ._p = high_left._p - low_left._p};

            vec3_p diff_right = {
              .x  = high_right.x - low_right.x,
              .y  = high_right.y - low_right.y,
              .z  = high_right.z - low_right.z,
              ._p = high_right._p - low_right._p};

            const float cost_L = diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;

            const float cost_R = diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

            left += bins[k - 1].entry;
            right -= bins[k - 1].exit;

            const float total_cost = COST_OF_NODE + COST_OF_TRIANGLE * cost_L / parent_surface_area * left
                                     + COST_OF_TRIANGLE * cost_R / parent_surface_area * right;

            if (total_cost < optimal_cost && total_cost < sequential_cost) {
              optimal_cost            = total_cost;
              optimal_split           = left;
              optimal_total_triangles = left + right;
              optimal_splitting_plane = low_split + k * interval;
              optimal_method          = 1;
              axis                    = a;
              optimal_high_left       = high_left;
              optimal_high_right      = high_right;
              optimal_low_left        = low_left;
              optimal_low_right       = low_right;
            }
          }
        }
      }

      if (optimal_cost == FLT_MAX) {
        optimal_split           = node.triangle_count / 2;
        optimal_total_triangles = node.triangle_count;
        optimal_method          = 2;
      }

      if (optimal_method == 0) {
        divide_middles_along_axis(
          optimal_splitting_plane, axis, fragments_buffer + buffer_ptr, fragments + fragments_ptr, node.triangle_count, optimal_split);
      }
      else if (optimal_method == 1) {
        const unsigned int duplicated_triangles = optimal_total_triangles - node.triangle_count;
        fragments_buffer_count += duplicated_triangles;

        for (unsigned int k = 0; k < leaf_node_count; k++) {
          if (nodes[leaf_nodes[k]].triangles_address > buffer_ptr)
            nodes[leaf_nodes[k]].triangles_address += duplicated_triangles;
        }

        if (fragments_buffer_count >= fragments_buffer_length) {
          fragments_buffer_length += triangles_length / 2 + 1;
          fragments_buffer = safe_realloc(fragments_buffer, sizeof(fragment) * fragments_buffer_length);
        }

        divide_along_axis(
          optimal_splitting_plane, axis, fragments_buffer + buffer_ptr, fragments + fragments_ptr, node.triangle_count, optimal_split);
      }
      else {
        quick_sort_fragments(fragments + fragments_ptr, node.triangle_count, axis);
        memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(fragment) * node.triangle_count);
      }

      fragments_ptr += node.triangle_count;

      if (write_ptr + 2 >= node_count) {
        node_count += triangles_length / THRESHOLD_TRIANGLES;
        nodes      = safe_realloc(nodes, sizeof(Node2) * node_count);
        leaf_nodes = safe_realloc(leaf_nodes, sizeof(unsigned int) * node_count);
      }

      fit_bounds(fragments_buffer + buffer_ptr, optimal_split, &high, &low);

      node.left_high.x   = high.x;
      node.left_high.y   = high.y;
      node.left_high.z   = high.z;
      node.left_low.x    = low.x;
      node.left_low.y    = low.y;
      node.left_low.z    = low.z;
      node.child_address = write_ptr;

      nodes[write_ptr].triangle_count    = optimal_split;
      nodes[write_ptr].triangles_address = buffer_ptr;
      nodes[write_ptr].kind              = BINARY_NODE_IS_LEAF_NODE;
      nodes[write_ptr].surface_area =
        (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z);

      nodes[write_ptr].self_high.x = high.x;
      nodes[write_ptr].self_high.y = high.y;
      nodes[write_ptr].self_high.z = high.z;
      nodes[write_ptr].self_low.x  = low.x;
      nodes[write_ptr].self_low.y  = low.y;
      nodes[write_ptr].self_low.z  = low.z;

      write_ptr++;
      buffer_ptr += optimal_split;

      fit_bounds(fragments_buffer + buffer_ptr, optimal_total_triangles - optimal_split, &high, &low);

      node.right_high.x = high.x;
      node.right_high.y = high.y;
      node.right_high.z = high.z;
      node.right_low.x  = low.x;
      node.right_low.y  = low.y;
      node.right_low.z  = low.z;

      nodes[write_ptr].triangle_count    = optimal_total_triangles - optimal_split;
      nodes[write_ptr].triangles_address = buffer_ptr;
      nodes[write_ptr].kind              = BINARY_NODE_IS_LEAF_NODE;
      nodes[write_ptr].surface_area =
        (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z);
      nodes[write_ptr].self_high.x = high.x;
      nodes[write_ptr].self_high.y = high.y;
      nodes[write_ptr].self_high.z = high.z;
      nodes[write_ptr].self_low.x  = low.x;
      nodes[write_ptr].self_low.y  = low.y;
      nodes[write_ptr].self_low.z  = low.z;

      write_ptr++;
      buffer_ptr += optimal_total_triangles - optimal_split;

      node.triangles_address = -1;
      node.kind              = 0;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;
    leaf_node_count += leaf_nodes_in_iteration;

    if (fragments_ptr != fragments_length) {
      memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(fragment) * (fragments_length - fragments_ptr));
      buffer_ptr += fragments_length - fragments_ptr;
      fragments_ptr += fragments_length - fragments_ptr;
    }

    _mm_free(fragments);
    fragments               = fragments_buffer;
    fragments_length        = fragments_buffer_count;
    fragments_buffer        = _mm_malloc(sizeof(fragment) * fragments_length * 2, 64);
    fragments_buffer_length = fragments_length * 2;
  }

  node_count = write_ptr;

  free(leaf_nodes);

  nodes = safe_realloc(nodes, sizeof(Node2) * node_count);

  Triangle* triangles_swap = malloc(sizeof(Triangle) * initial_triangles_length);
  memcpy(triangles_swap, triangles, sizeof(Triangle) * initial_triangles_length);
  triangles = safe_realloc(triangles, sizeof(Triangle) * fragments_length);

  for (unsigned int i = 0; i < fragments_length; i++) {
    triangles[i] = triangles_swap[fragments[i].id];
  }

  free(triangles_swap);
  _mm_free(fragments);
  _mm_free(bins);

  *triangles_io = triangles;

  *triangles_length_io = fragments_length;

  *nodes_length_out = node_count;

  bench_toc("Binary BVH Construction");

  return nodes;
}

static void compute_single_node_triangles(Node2* binary_nodes, const int index) {
  if (binary_nodes[index].kind == BINARY_NODE_IS_LEAF_NODE)
    return;

  const int child_address = binary_nodes[index].child_address;

  compute_single_node_triangles(binary_nodes, child_address);
  compute_single_node_triangles(binary_nodes, child_address + 1);

  binary_nodes[index].triangles_address = binary_nodes[child_address].triangles_address;
  binary_nodes[index].triangle_count    = binary_nodes[child_address].triangle_count + binary_nodes[child_address + 1].triangle_count;
}

// recursion seems to be a sufficient solution in this case
static void compute_node_triangle_properties(Node2* binary_nodes) {
  bench_tic();
  compute_single_node_triangles(binary_nodes, 0);
  bench_toc("Node Triangle Property Recovery");
}

static float cost_distribute(Node2* binary_nodes, Node2 node, int j, int* decision) {
  float min = FLT_MAX;

  for (int k = 0; k < j; k++) {
    const float cost = binary_nodes[node.child_address].sah_cost[k] + binary_nodes[node.child_address + 1].sah_cost[j - 1 - k];
    if (cost <= min) {
      *decision = k + 1;
      min       = cost;
    }
  }

  return min;
}

#define OPTIMAL_LEAF 0xffff

static void compute_single_node_costs(Node2* binary_nodes, const int index) {
  if (binary_nodes[index].kind == BINARY_NODE_IS_LEAF_NODE) {
    Node2 node = binary_nodes[index];

    const float cost = node.surface_area * COST_OF_TRIANGLE;

    for (int i = 0; i < 7; i++) {
      node.sah_cost[i] = cost;
      node.decision[i] = 0;
    }

    node.decision[0] = OPTIMAL_LEAF;

    node.cost_computed = 1;

    binary_nodes[index] = node;
  }
  else {
    int child_address = binary_nodes[index].child_address;
    if (!binary_nodes[child_address].cost_computed)
      compute_single_node_costs(binary_nodes, child_address);

    if (!binary_nodes[child_address + 1].cost_computed)
      compute_single_node_costs(binary_nodes, child_address + 1);

    Node2 node = binary_nodes[index];

    int decision        = 0;
    float cost_leaf     = (node.triangle_count <= TRIANGLES_MAX) ? node.surface_area * node.triangle_count * COST_OF_TRIANGLE : FLT_MAX;
    float cost_internal = node.surface_area * COST_OF_NODE + cost_distribute(binary_nodes, node, 7, &decision);
    node.sah_cost[0]    = fminf(cost_leaf, cost_internal);
    node.decision[0]    = (cost_leaf <= cost_internal) ? OPTIMAL_LEAF : decision;

    for (int i = 1; i < 7; i++) {
      float cost       = cost_distribute(binary_nodes, node, i, &decision);
      node.sah_cost[i] = fminf(cost, node.sah_cost[i - 1]);
      node.decision[i] = (cost <= node.sah_cost[i - 1]) ? decision : 0;
    }

    node.cost_computed = 1;

    binary_nodes[index] = node;
  }
}

// recursion seems to be a sufficient solution in this case
static void compute_sah_costs(Node2* binary_nodes) {
  bench_tic();
  compute_single_node_costs(binary_nodes, 0);
  bench_toc("SAH Cost Computation");
}

struct node_packed {
  Node2* binary_nodes;
  Node2* binary_children;
  vec3* low;
  vec3* high;
  int* binary_addresses;
} typedef node_packed;

static void apply_decision(Node2* node, int node_index, int decision, int slot, node_packed node_data) {
  int split          = 0;
  int decision_index = decision - 1;

  while (!split) {
    split = node->decision[decision_index--];
  }

  decision_index++;

  if (split == OPTIMAL_LEAF) {
    node->kind                       = BINARY_NODE_IS_LEAF_NODE;
    node_data.binary_children[slot]  = *node;
    node_data.low[slot]              = node->self_low;
    node_data.high[slot]             = node->self_high;
    node_data.binary_addresses[slot] = node_index;
  }
  else if (decision_index == 0) {
    node->kind                       = BINARY_NODE_IS_INTERNAL_NODE;
    node_data.binary_children[slot]  = *node;
    node_data.low[slot]              = node->self_low;
    node_data.high[slot]             = node->self_high;
    node_data.binary_addresses[slot] = node_index;
  }
  else {
    apply_decision(node_data.binary_nodes + node->child_address, node->child_address, split, slot, node_data);
    apply_decision(node_data.binary_nodes + node->child_address + 1, node->child_address + 1, decision - split, slot + split, node_data);
  }
}

Node8* collapse_bvh(
  Node2* binary_nodes, const unsigned int binary_nodes_length, Triangle** triangles_io, const int triangles_length,
  unsigned int* nodes_length_out) {
  compute_node_triangle_properties(binary_nodes);
  compute_sah_costs(binary_nodes);

  bench_tic();

  Triangle* triangles = *triangles_io;
  int node_count      = binary_nodes_length;

  Node8* nodes = (Node8*) malloc(sizeof(Node8) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node8) * node_count);

  uint32_t* bvh_triangles = _mm_malloc(sizeof(uint32_t) * triangles_length, 64);
  uint32_t* new_triangles = _mm_malloc(sizeof(uint32_t) * triangles_length, 64);

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangles[i] = i;
    new_triangles[i] = i;
  }

  nodes[0].triangle_base_index   = 0;
  nodes[0].child_node_base_index = 0;

  int begin_of_current_nodes = 0;
  int end_of_current_nodes   = 1;
  int write_ptr              = 1;
  int triangles_ptr          = 0;

  while (begin_of_current_nodes != end_of_current_nodes) {
    for (int node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes; node_ptr++) {
      Node8 node = nodes[node_ptr];

      int binary_index = node.child_node_base_index;

      node.child_node_base_index = write_ptr;
      node.triangle_base_index   = triangles_ptr;

      Node2 binary_children[8];
      vec3 low[8];
      vec3 high[8];
      int binary_addresses[8];

      for (int i = 0; i < 8; i++) {
        binary_children[i].kind = BINARY_NODE_IS_NULL;
      }

      node_packed node_data = {
        .binary_nodes     = binary_nodes,
        .binary_children  = (Node2*) &binary_children,
        .low              = (vec3*) &low,
        .high             = (vec3*) &high,
        .binary_addresses = (int*) &binary_addresses};

      const int split         = binary_nodes[binary_index].decision[0];
      const int child_address = binary_nodes[binary_index].child_address;

      apply_decision(binary_nodes + child_address, child_address, split, 0, node_data);
      apply_decision(binary_nodes + child_address + 1, child_address + 1, 8 - split, split, node_data);

      vec3 node_low  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};
      vec3 node_high = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};

      for (int i = 0; i < 8; i++) {
        if (binary_children[i].kind != BINARY_NODE_IS_NULL) {
          vec3 lo    = low[i];
          node_low.x = min(node_low.x, lo.x);
          node_low.y = min(node_low.y, lo.y);
          node_low.z = min(node_low.z, lo.z);

          vec3 hi     = high[i];
          node_high.x = max(node_high.x, hi.x);
          node_high.y = max(node_high.y, hi.y);
          node_high.z = max(node_high.z, hi.z);
        }
      }

      node.p = node_low;

      node.ex = (int8_t) ceilf(log2f((node_high.x - node_low.x) * 1.0 / 255.0));
      node.ey = (int8_t) ceilf(log2f((node_high.y - node_low.y) * 1.0 / 255.0));
      node.ez = (int8_t) ceilf(log2f((node_high.z - node_low.z) * 1.0 / 255.0));

      const float compression_x = 1.0f / exp2f(node.ex);
      const float compression_y = 1.0f / exp2f(node.ey);
      const float compression_z = 1.0f / exp2f(node.ez);

      float cost_table[8][8];
      int order[8];
      int slot_empty[8];

      for (int i = 0; i < 8; i++) {
        slot_empty[i] = 1;
        order[i]      = -1;
      }

      for (int i = 0; i < 8; i++) {
        vec3 direction = {.x = ((i >> 2) & 0b1) ? -1.0f : 1.0f, .y = ((i >> 1) & 0b1) ? -1.0f : 1.0f, .z = ((i >> 0) & 0b1) ? -1.0f : 1.0f};

        for (int j = 0; j < 8; j++) {
          if (binary_children[j].kind == BINARY_NODE_IS_NULL) {
            cost_table[i][j] = FLT_MAX;
          }
          else {
            vec3 child_centroid = {.x = high[j].x + low[j].x, .y = high[j].y + low[j].y, .z = high[j].z + low[j].z};

            cost_table[i][j] = child_centroid.x * direction.x + child_centroid.y * direction.y + child_centroid.z * direction.z;
          }
        }
      }

      while (1) {
        float min_cost = FLT_MAX;
        int slot       = -1;
        int child      = -1;

        for (int i = 0; i < 8; i++) {
          for (int j = 0; j < 8; j++) {
            if (order[j] == -1 && slot_empty[i] && cost_table[i][j] < min_cost) {
              min_cost = cost_table[i][j];
              slot     = i;
              child    = j;
            }
          }
        }

        if (slot != -1 || child != -1) {
          slot_empty[slot] = 0;
          order[child]     = slot;
        }
        else {
          break;
        }
      }

      for (int i = 0; i < 8; i++) {
        if (order[i] == -1) {
          for (int j = 0; j < 8; j++) {
            if (slot_empty[j]) {
              slot_empty[j] = 0;
              order[i]      = j;
              break;
            }
          }
        }
      }

      Node2 old_binary_children[8];
      vec3 old_low[8];
      vec3 old_high[8];
      int old_binary_addresses[8];

      for (int i = 0; i < 8; i++) {
        old_binary_children[i]  = binary_children[i];
        old_low[i]              = low[i];
        old_high[i]             = high[i];
        old_binary_addresses[i] = binary_addresses[i];
      }

      for (int i = 0; i < 8; i++) {
        binary_children[order[i]]  = old_binary_children[i];
        low[order[i]]              = old_low[i];
        high[order[i]]             = old_high[i];
        binary_addresses[order[i]] = old_binary_addresses[i];
      }

      node.imask                    = 0;
      int triangle_counting_address = 0;

      if (write_ptr + 8 >= node_count) {
        node_count += binary_nodes_length;
        nodes = safe_realloc(nodes, sizeof(Node8) * node_count);
      }

      for (int i = 0; i < 8; i++) {
        Node2 base_child = binary_children[i];
        if (base_child.kind == BINARY_NODE_IS_INTERNAL_NODE) {
          node.imask |= 1 << i;
          nodes[write_ptr++].child_node_base_index = binary_addresses[i];
          node.meta[i]                             = 0b00100000 + 0b11000 + i;
        }
        else if (base_child.kind == BINARY_NODE_IS_LEAF_NODE) {
          assert(base_child.triangle_count < 4, "Error when collapsing nodes. There are too many unsplittable triangles.", 1);
          int meta = 0;
          switch (base_child.triangle_count) {
            case 3:
              meta = 0b111;
              break;
            case 2:
              meta = 0b11;
              break;
            case 1:
              meta = 0b1;
              break;
          }
          meta = meta << 5;
          meta += triangle_counting_address;

          node.meta[i] = meta;

          memcpy(new_triangles + triangles_ptr, bvh_triangles + base_child.triangles_address, sizeof(uint32_t) * base_child.triangle_count);

          triangles_ptr += base_child.triangle_count;
          triangle_counting_address += base_child.triangle_count;
        }
        else {
          node.meta[i] = 0;
        }
      }

      for (int i = 0; i < 8; i++) {
        if (binary_children[i].kind == BINARY_NODE_IS_NULL) {
          node.low_x[i] = 0;
          node.low_y[i] = 0;
          node.low_z[i] = 0;

          node.high_x[i] = 0;
          node.high_y[i] = 0;
          node.high_z[i] = 0;
        }
        else {
          vec3 lo       = low[i];
          vec3 hi       = high[i];
          node.low_x[i] = (uint8_t) floorf((lo.x - node.p.x) * compression_x);
          node.low_y[i] = (uint8_t) floorf((lo.y - node.p.y) * compression_y);
          node.low_z[i] = (uint8_t) floorf((lo.z - node.p.z) * compression_z);

          node.high_x[i] = (uint8_t) ceilf((hi.x - node.p.x) * compression_x);
          node.high_y[i] = (uint8_t) ceilf((hi.y - node.p.y) * compression_y);
          node.high_z[i] = (uint8_t) ceilf((hi.z - node.p.z) * compression_z);
        }
      }

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;
  }

  _mm_free(bvh_triangles);
  bvh_triangles = new_triangles;

  Triangle* triangles_swap = malloc(sizeof(Triangle) * triangles_length);
  memcpy(triangles_swap, triangles, sizeof(Triangle) * triangles_length);

  for (int i = 0; i < triangles_length; i++) {
    triangles[i] = triangles_swap[bvh_triangles[i]];
  }

  free(triangles_swap);

  _mm_free(new_triangles);

  node_count = write_ptr;

  nodes = safe_realloc(nodes, sizeof(Node8) * node_count);

  *triangles_io = triangles;

  *nodes_length_out = node_count;

  bench_toc("Collapsing BVH");

  return nodes;
}

static void sort_triangles_depth_first(Triangle* src, Triangle* dst, Node8* nodes, const int node_index, int* offset) {
  Node8* node = nodes + node_index;

  const uint8_t imask = node->imask;

  const int new_triangle_base_index = *offset;
  int new_rel_offset                = 0;

  // Insert Leaf Nodes
  for (int i = 0; i < 8; i++) {
    if ((imask >> i) & 0b1)
      continue;

    const uint8_t meta = node->meta[i];

    if (meta == 0)
      continue;

    const int count = _mm_popcnt_u32(meta & 0b11100000);
    const int index = node->triangle_base_index + (meta & 0b11111);
    for (int j = 0; j < count; j++) {
      dst[*offset] = src[index + j];
      *offset      = (*offset) + 1;
    }

    node->meta[i] = (meta & 0b11100000) | new_rel_offset;
    new_rel_offset += count;
  }

  node->triangle_base_index = new_triangle_base_index;

  int child_index = node->child_node_base_index;

  // Traverse Internal Nodes
  for (int i = 0; i < 8; i++) {
    if ((~(imask >> i)) & 0b1)
      continue;

    const int index = child_index++;
    sort_triangles_depth_first(src, dst, nodes, index, offset);
  }
}

static void sort_nodes_depth_first(Node8* src, Node8* dst, const int src_index, const int dst_index, int* index) {
  Node8* src_node = src + src_index;
  Node8* dst_node = dst + dst_index;

  const uint8_t imask = src_node->imask;

  dst_node->child_node_base_index = *index;

  int child_index = 0;

  for (int i = 0; i < 8; i++) {
    if ((~(imask >> i)) & 0b1)
      continue;

    const int si = src_node->child_node_base_index + child_index++;

    dst[*index] = src[si];
    *index      = (*index) + 1;
  }

  child_index = 0;

  for (int i = 0; i < 8; i++) {
    if ((~(imask >> i)) & 0b1)
      continue;

    const int si = src_node->child_node_base_index + child_index;
    const int di = dst_node->child_node_base_index + child_index;
    sort_nodes_depth_first(src, dst, si, di, index);
    child_index++;
  }
}

/*
 * Sorts both nodes and triangles into depth first order.
 */
void sort_traversal_elements(Node8** nodes_io, const int nodes_length, Triangle** triangles_io, const int triangles_length) {
  bench_tic();

  Triangle* triangles = *triangles_io;
  Node8* nodes        = *nodes_io;

  Node8* new_nodes = (Node8*) malloc(sizeof(Node8) * nodes_length);

  new_nodes[0] = nodes[0];

  int offset = 1;

  sort_nodes_depth_first(nodes, new_nodes, 0, 0, &offset);

  free(nodes);

  *nodes_io = new_nodes;

  nodes = new_nodes;

  Triangle* new_triangles = (Triangle*) malloc(sizeof(Triangle) * triangles_length);

  offset = 0;

  sort_triangles_depth_first(triangles, new_triangles, nodes, 0, &offset);

  free(triangles);

  *triangles_io = new_triangles;

  bench_toc("Sorting Traversal Structures");
}
