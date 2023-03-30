#include "bvh.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "structs.h"
#include "utils.h"

/*
 * BIN_COUNTS must be power of 2
 */
#define THRESHOLD_TRIANGLES 1
#define OBJECT_SPLIT_BIN_COUNT 8
#define SPATIAL_SPLIT_THRESHOLD 0.00001
#define SPATIAL_SPLIT_BIN_COUNT 64
#define COST_OF_TRIANGLE 0.4
#define COST_OF_NODE 1.0

#define FRAGMENT_ERROR_COMP (FLT_EPSILON * 4.0f)
#define SPATIAL_SPLIT_TOL (FLT_EPSILON * 16.0f)

// We need to bound the dimensions, the number must be large but still much smaller than FLT_MAX
#define MAX_VALUE 1e10f

#define BINARY_BVH_ERROR_CHECK 1

#define TRIANGLES_MAX 3

enum Axis { AxisX = 0, AxisY = 1, AxisZ = 2 } typedef Axis;
enum OptimalStrategy { StrategyNull = 0, StrategyObjectSplit = 1, StrategySpatialSplit = 2 } typedef OptimalStrategy;

struct vec3_p {
  float x;
  float y;
  float z;
  float _p;
} typedef vec3_p;

struct Fragment {
  vec3_p high;
  vec3_p low;
  vec3_p middle;
  uint32_t id;
  uint32_t _p2;
  uint64_t _p3;
} typedef Fragment;

struct Bin {
  vec3_p high;
  vec3_p low;
  int32_t entry;
  int32_t exit;
  uint64_t _p;
} typedef Bin;

static float get_entry_by_axis(const vec3_p p, const Axis axis) {
  switch (axis) {
    case AxisX:
      return p.x;
    case AxisY:
      return p.y;
    case AxisZ:
    default:
      return p.z;
  }
}

static void __swap_fragments(Fragment* a, Fragment* b) {
  Fragment temp = *a;
  *a            = *b;
  *b            = temp;
}

static int __partition(Fragment* fragments, const int bottom, const int top, const Axis axis) {
  const float x = get_entry_by_axis(fragments[(top + bottom) / 2].middle, axis);
  int i         = bottom - 1;
  int j         = top + 1;

  while (1) {
    do {
      i++;
    } while (get_entry_by_axis(fragments[i].middle, axis) < x);

    do {
      j--;
    } while (get_entry_by_axis(fragments[j].middle, axis) > x);

    if (i >= j)
      return j;

    __swap_fragments(fragments + i, fragments + j);
  }
}

static void quick_sort_fragments(Fragment* fragments, const unsigned int fragments_length, const int axis) {
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
      quick_sort_stack[++ptr] = p;
    }

    if (p + 1 < top) {
      quick_sort_stack[++ptr] = p + 1;
      quick_sort_stack[++ptr] = top;
    }
  }
}

static void fit_bounds(const Fragment* fragments, const unsigned int fragments_length, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_set1_ps(-MAX_VALUE);
  __m128 low  = _mm_set1_ps(MAX_VALUE);

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

static void fit_bounds_of_bins(const Bin* bins, const int bins_length, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_set1_ps(-MAX_VALUE);
  __m128 low  = _mm_set1_ps(MAX_VALUE);

  for (int i = 0; i < bins_length; i++) {
    const float* baseptr = (float*) (bins + i);

    const __m128 high_bin = _mm_loadu_ps(baseptr);
    const __m128 low_bin  = _mm_loadu_ps(baseptr + 4);

    high = _mm_max_ps(high, high_bin);
    low  = _mm_min_ps(low, low_bin);
  }

  if (high_out)
    _mm_storeu_ps((float*) high_out, high);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low);
}

static void update_bounds_of_bins(const Bin* bins, vec3_p* restrict high_out, vec3_p* restrict low_out) {
  __m128 high = _mm_loadu_ps((float*) high_out);
  __m128 low  = _mm_loadu_ps((float*) low_out);

  const float* baseptr = (float*) (bins);

  __m128 high_bin = _mm_loadu_ps(baseptr);
  __m128 low_bin  = _mm_loadu_ps(baseptr + 4);

  high = _mm_max_ps(high, high_bin);
  low  = _mm_min_ps(low, low_bin);

  _mm_storeu_ps((float*) high_out, high);
  _mm_storeu_ps((float*) low_out, low);
}

static double construct_bins(
  Bin* restrict bins, const Fragment* restrict fragments, const unsigned int fragments_length, const Axis axis, double* offset) {
  vec3_p high, low;
  fit_bounds(fragments, fragments_length, &high, &low);

  const double high_axis = get_entry_by_axis(high, axis);
  const double low_axis  = get_entry_by_axis(low, axis);

  const double span     = high_axis - low_axis;
  const double interval = span / OBJECT_SPLIT_BIN_COUNT;

  if (interval <= FRAGMENT_ERROR_COMP * fabs(low_axis))
    return 0.0;

  *offset = low_axis;

  const Bin b = {
    .high.x = -MAX_VALUE,
    .high.y = -MAX_VALUE,
    .high.z = -MAX_VALUE,
    .low.x  = MAX_VALUE,
    .low.y  = MAX_VALUE,
    .low.z  = MAX_VALUE,
    .entry  = 0,
    .exit   = 0};
  bins[0] = b;

  for (uint32_t i = 1; i < OBJECT_SPLIT_BIN_COUNT; i = i << 1) {
    memcpy(bins + i, bins, i * sizeof(Bin));
  }

  for (unsigned int i = 0; i < fragments_length; i++) {
    Fragment frag     = fragments[i];
    const float value = get_entry_by_axis(frag.middle, axis);
    int pos           = ((int) ceil((value - low_axis) / interval)) - 1;
    if (pos < 0)
      pos = 0;

    Bin b = bins[pos];
    b.entry++;
    b.exit++;

    b.high.x  = fmaxf(b.high.x, frag.high.x);
    b.high.y  = fmaxf(b.high.y, frag.high.y);
    b.high.z  = fmaxf(b.high.z, frag.high.z);
    b.high._p = fmaxf(b.high._p, frag.high._p);
    b.low.x   = fminf(b.low.x, frag.low.x);
    b.low.y   = fminf(b.low.y, frag.low.y);
    b.low.z   = fminf(b.low.z, frag.low.z);
    b.low._p  = fminf(b.low._p, frag.low._p);
    bins[pos] = b;
  }

  return interval;
}

static double construct_chopped_bins(
  Bin* restrict bins, const Fragment* restrict fragments, const unsigned int fragments_length, const Axis axis, double* restrict offset) {
  vec3_p high, low;
  fit_bounds(fragments, fragments_length, &high, &low);

  const double high_axis = get_entry_by_axis(high, axis);
  const double low_axis  = get_entry_by_axis(low, axis);

  const double span     = high_axis - low_axis;
  const double interval = span / SPATIAL_SPLIT_BIN_COUNT;

  if (interval <= FRAGMENT_ERROR_COMP * fabs(low_axis))
    return 0.0;

  *offset = low_axis;

  const Bin b = {
    .high.x = -MAX_VALUE,
    .high.y = -MAX_VALUE,
    .high.z = -MAX_VALUE,
    .low.x  = MAX_VALUE,
    .low.y  = MAX_VALUE,
    .low.z  = MAX_VALUE,
    .entry  = 0,
    .exit   = 0};
  bins[0] = b;

  for (uint32_t i = 1; i < SPATIAL_SPLIT_BIN_COUNT; i = i << 1) {
    memcpy(bins + i, bins, i * sizeof(Bin));
  }

  for (unsigned int i = 0; i < fragments_length; i++) {
    const double value1 = get_entry_by_axis(fragments[i].low, axis);
    int pos1            = ((int) ceil((value1 - low_axis) / interval)) - 1;
    if (pos1 < 0)
      pos1 = 0;

    if (fabs(value1 - low_axis - pos1 * interval) < SPATIAL_SPLIT_TOL * value1) {
      return 0.0;
    }

    const double value2 = get_entry_by_axis(fragments[i].high, axis);
    int pos2            = ((int) ceil((value2 - low_axis) / interval)) - 1;
    if (pos2 < 0)
      pos2 = 0;

    if (fabs(value2 - low_axis - pos2 * interval) < SPATIAL_SPLIT_TOL * value2) {
      return 0.0;
    }

    const int entry = min(pos1, pos2);
    const int exit  = max(pos1, pos2);

    const __m128 high_frag = _mm_loadu_ps((float*) (fragments + i));
    const __m128 low_frag  = _mm_loadu_ps(((float*) (fragments + i)) + 4);

    for (int j = entry; j <= exit; j++) {
      if (j == entry)
        bins[j].entry++;
      if (j == exit)
        bins[j].exit++;

      __m128 high_bin = _mm_loadu_ps((float*) (bins + j));
      __m128 low_bin  = _mm_loadu_ps(((float*) (bins + j)) + 4);

      high_bin = _mm_max_ps(high_bin, high_frag);
      low_bin  = _mm_min_ps(low_bin, low_frag);

      _mm_store_ps((float*) (bins + j), high_bin);
      _mm_store_ps(((float*) (bins + j)) + 4, low_bin);
    }
  }

  switch (axis) {
    case AxisX:
      for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
        Bin b    = bins[i];
        b.low.x  = fmaxf(b.low.x, low.x + i * interval);
        b.high.x = fminf(b.high.x, low.x + (i + 1) * interval);
        bins[i]  = b;
      }
      break;
    case AxisY:
      for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
        Bin b    = bins[i];
        b.low.y  = fmaxf(b.low.y, low.y + i * interval);
        b.high.y = fminf(b.high.y, low.y + (i + 1) * interval);
        bins[i]  = b;
      }
      break;
    case AxisZ:
    default:
      for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
        Bin b    = bins[i];
        b.low.z  = fmaxf(b.low.z, low.z + i * interval);
        b.high.z = fminf(b.high.z, low.z + (i + 1) * interval);
        bins[i]  = b;
      }
      break;
  }

  return interval;
}

static void divide_middles_along_axis(
  const double split, const Axis axis, Fragment* restrict fragments_out, Fragment* restrict fragments_in,
  const unsigned int fragments_length) {
  int left  = 0;
  int right = 0;

  for (unsigned int i = 0; i < fragments_length; i++) {
    Fragment frag = fragments_in[i];

    const double middle = get_entry_by_axis(frag.middle, axis);

    if (middle <= split) {
      fragments_out[left++] = frag;
    }
    else {
      fragments_out[fragments_length - 1 - right++] = frag;
    }
  }
}

static void divide_along_axis(
  const double split, const Axis axis, Fragment* fragments_out, Fragment* fragments_in, const unsigned int fragments_length,
  const int right_offset, const int fragments_out_length) {
  int left  = 0;
  int right = 0;

  for (unsigned int i = 0; i < fragments_length; i++) {
    const double low  = get_entry_by_axis(fragments_in[i].low, axis);
    const double high = get_entry_by_axis(fragments_in[i].high, axis);

    // The splits may happen so that the fragments are shrunk to a zero volume
    // Through nextafterf, I ensure a minimal volume

    if (low < split) {
      Fragment frag_left = fragments_in[i];
      switch (axis) {
        case AxisX:
          frag_left.high.x = min(frag_left.high.x, split);
          if (frag_left.high.x == frag_left.low.x) {
            frag_left.high.x = nextafterf(frag_left.high.x, FLT_MAX);
          }
          frag_left.middle.x = (frag_left.low.x + frag_left.high.x) * 0.5f;
          break;
        case AxisY:
          frag_left.high.y = min(frag_left.high.y, split);
          if (frag_left.high.y == frag_left.low.y) {
            frag_left.high.y = nextafterf(frag_left.high.y, FLT_MAX);
          }
          frag_left.middle.y = (frag_left.low.y + frag_left.high.y) * 0.5f;
          break;
        case AxisZ:
        default:
          frag_left.high.z = min(frag_left.high.z, split);
          if (frag_left.high.z == frag_left.low.z) {
            frag_left.high.z = nextafterf(frag_left.high.z, FLT_MAX);
          }
          frag_left.middle.z = (frag_left.low.z + frag_left.high.z) * 0.5f;
          break;
      }
      fragments_out[left++] = frag_left;
    }

    if (high > split) {
      Fragment frag_right = fragments_in[i];
      switch (axis) {
        case AxisX:
          frag_right.low.x = max(frag_right.low.x, split);
          if (frag_right.high.x == frag_right.low.x) {
            frag_right.low.x = nextafterf(frag_right.low.x, -FLT_MAX);
          }
          frag_right.middle.x = (frag_right.low.x + frag_right.high.x) * 0.5f;
          break;
        case AxisY:
          frag_right.low.y = max(frag_right.low.y, split);
          if (frag_right.high.y == frag_right.low.y) {
            frag_right.low.y = nextafterf(frag_right.low.y, -FLT_MAX);
          }
          frag_right.middle.y = (frag_right.low.y + frag_right.high.y) * 0.5f;
          break;
        case AxisZ:
        default:
          frag_right.low.z = max(frag_right.low.z, split);
          if (frag_right.high.z == frag_right.low.z) {
            frag_right.low.z = nextafterf(frag_right.low.z, -FLT_MAX);
          }
          frag_right.middle.z = (frag_right.low.z + frag_right.high.z) * 0.5f;
          break;
      }
      fragments_out[right_offset + right++] = frag_right;
    }
  }

  // Due to numerical errors we may see a different number of fragments than predicted
  // Hence I just fill the remaining space

  for (; left < right_offset; left++) {
    fragments_out[left] = fragments_out[0];
  }

  for (; right_offset + right < fragments_out_length; right++) {
    fragments_out[right_offset + right] = fragments_out[right_offset];
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
  nodes[0].type              = NodeTypeLeaf;

  unsigned int* leaf_nodes     = malloc(sizeof(unsigned int) * node_count);
  unsigned int leaf_node_count = 0;

  Fragment* fragments           = _mm_malloc(sizeof(Fragment) * triangles_length, 64);
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
    vec3_p middle = {.x = (high.x + low.x) * 0.5f, .y = (high.y + low.y) * 0.5f, .z = (high.z + low.z) * 0.5f};

    // Fragments are supposed to be hulls that contain a triangle
    // They should be as tight as possible but if they are too tight then numerical errors
    // could result in broken BVHs
    high.x += fabsf(high.x) * FRAGMENT_ERROR_COMP;
    high.y += fabsf(high.y) * FRAGMENT_ERROR_COMP;
    high.z += fabsf(high.z) * FRAGMENT_ERROR_COMP;
    low.x -= fabsf(low.x) * FRAGMENT_ERROR_COMP;
    low.y -= fabsf(low.y) * FRAGMENT_ERROR_COMP;
    low.z -= fabsf(low.z) * FRAGMENT_ERROR_COMP;

    Fragment frag = {.high = high, .low = low, .middle = middle, .id = i};
    fragments[i]  = frag;
  }

  Fragment* fragments_buffer           = _mm_malloc(sizeof(Fragment) * fragments_length * 2, 64);
  unsigned int fragments_buffer_length = fragments_length * 2;
  unsigned int fragments_buffer_count  = fragments_length;

  double root_surface_area;
  vec3_p high_root, low_root;
  {
    fit_bounds(fragments, fragments_length, &high_root, &low_root);
    vec3_p diff = {
      .x = high_root.x - low_root.x, .y = high_root.y - low_root.y, .z = high_root.z - low_root.z, ._p = high_root._p - low_root._p};

    root_surface_area = (double) diff.x * (double) diff.y + (double) diff.x * (double) diff.z + (double) diff.y * (double) diff.z;
  }

  Bin* bins = (Bin*) _mm_malloc(sizeof(Bin) * max(OBJECT_SPLIT_BIN_COUNT, SPATIAL_SPLIT_BIN_COUNT), 32);

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
        memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(Fragment) * (node.triangles_address - fragments_ptr));
        buffer_ptr += node.triangles_address - fragments_ptr;
        fragments_ptr += node.triangles_address - fragments_ptr;
      }

      if (node.triangle_count <= THRESHOLD_TRIANGLES) {
        nodes[node_ptr].triangles_address = buffer_ptr;
        nodes[node_ptr].cost_computed     = 0;

        memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(Fragment) * node.triangle_count);
        buffer_ptr += node.triangle_count;
        fragments_ptr += node.triangle_count;
        leaf_nodes[leaf_node_count + leaf_nodes_in_iteration++] = node_ptr;
        continue;
      }

      double parent_surface_area;
      {
        vec3_p high_parent, low_parent;
        fit_bounds(fragments + fragments_ptr, node.triangle_count, &high_parent, &low_parent);
        vec3 diff = {.x = high_parent.x - low_parent.x, .y = high_parent.y - low_parent.y, .z = high_parent.z - low_parent.z};

        parent_surface_area = (double) diff.x * (double) diff.y + (double) diff.x * (double) diff.z + (double) diff.y * (double) diff.z;
      }

      const double sequential_cost = COST_OF_NODE * 0.5 + COST_OF_TRIANGLE * node.triangle_count;

      vec3_p high, low;
      vec3_p optimal_high_right, optimal_low_right, optimal_high_left, optimal_low_left;
      double optimal_cost = DBL_MAX;
      Axis axis;
      double optimal_splitting_plane;
      OptimalStrategy optimal_strategy = StrategyNull;
      int optimal_split                = node.triangle_count / 2;
      int optimal_total_triangles      = node.triangle_count;

      for (int a = 0; a < 3; a++) {
        double low_split;
        const double interval = construct_bins(bins, fragments + fragments_ptr, node.triangle_count, a, &low_split);

        if (interval == 0.0)
          continue;

        int left  = 0;
        int right = node.triangle_count;

        vec3_p high_left  = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE, ._p = -MAX_VALUE};
        vec3_p high_right = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE, ._p = -MAX_VALUE};
        vec3_p low_left   = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE, ._p = MAX_VALUE};
        vec3_p low_right  = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE, ._p = MAX_VALUE};

        for (int k = 1; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          update_bounds_of_bins(bins + k - 1, &high_left, &low_left);
          fit_bounds_of_bins(bins + k, OBJECT_SPLIT_BIN_COUNT - k, &high_right, &low_right);

          vec3_p diff_left = {
            .x = high_left.x - low_left.x, .y = high_left.y - low_left.y, .z = high_left.z - low_left.z, ._p = high_left._p - low_left._p};

          vec3_p diff_right = {
            .x  = high_right.x - low_right.x,
            .y  = high_right.y - low_right.y,
            .z  = high_right.z - low_right.z,
            ._p = high_right._p - low_right._p};

          const double cost_L = diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;
          const double cost_R = diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

          left += bins[k - 1].entry;
          right -= bins[k - 1].exit;

          const double total_cost =
            COST_OF_NODE + COST_OF_TRIANGLE * (cost_L / parent_surface_area * left + cost_R / parent_surface_area * right);

          if (total_cost < optimal_cost && total_cost < sequential_cost) {
            optimal_cost            = total_cost;
            optimal_split           = left;
            optimal_total_triangles = node.triangle_count;
            optimal_splitting_plane = low_split + k * interval;
            optimal_strategy        = StrategyObjectSplit;
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

      const double o              = overlap.x * overlap.y + overlap.x * overlap.z + overlap.y * overlap.z;
      const int do_spatial_splits = ((o / root_surface_area) > SPATIAL_SPLIT_THRESHOLD);

      if (do_spatial_splits) {
        for (int a = 0; a < 3; a++) {
          double low_split;
          const double interval = construct_chopped_bins(bins, fragments + fragments_ptr, node.triangle_count, a, &low_split);

          if (interval == 0.0)
            continue;

          int left  = 0;
          int right = node.triangle_count;

          vec3_p high_left  = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};
          vec3_p high_right = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};
          vec3_p low_left   = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};
          vec3_p low_right  = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};

          for (int k = 1; k < SPATIAL_SPLIT_BIN_COUNT; k++) {
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

            const double cost_L = diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;
            const double cost_R = diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

            left += bins[k - 1].entry;
            right -= bins[k - 1].exit;

            const double total_cost =
              COST_OF_NODE + COST_OF_TRIANGLE * (cost_L / parent_surface_area * left + cost_R / parent_surface_area * right);

            if (total_cost < optimal_cost && total_cost < sequential_cost) {
              optimal_cost            = total_cost;
              optimal_split           = left;
              optimal_total_triangles = left + right;
              optimal_splitting_plane = low_split + k * interval;
              optimal_strategy        = StrategySpatialSplit;
              axis                    = a;
              optimal_high_left       = high_left;
              optimal_high_right      = high_right;
              optimal_low_left        = low_left;
              optimal_low_right       = low_right;
            }
          }
        }
      }

      switch (optimal_strategy) {
        case StrategyNull:
          memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(Fragment) * node.triangle_count);
          quick_sort_fragments(fragments_buffer + buffer_ptr, node.triangle_count, axis);
          break;
        case StrategyObjectSplit:
          divide_middles_along_axis(
            optimal_splitting_plane, axis, fragments_buffer + buffer_ptr, fragments + fragments_ptr, node.triangle_count);
          break;
        case StrategySpatialSplit: {
          const unsigned int duplicated_triangles = optimal_total_triangles - node.triangle_count;
          fragments_buffer_count += duplicated_triangles;

          for (unsigned int k = 0; k < leaf_node_count; k++) {
            if (nodes[leaf_nodes[k]].triangles_address > buffer_ptr)
              nodes[leaf_nodes[k]].triangles_address += duplicated_triangles;
          }

          if (fragments_buffer_count >= fragments_buffer_length) {
            fragments_buffer_length += triangles_length / 2 + 1;
            fragments_buffer = safe_realloc(fragments_buffer, sizeof(Fragment) * fragments_buffer_length);
          }

          divide_along_axis(
            optimal_splitting_plane, axis, fragments_buffer + buffer_ptr, fragments + fragments_ptr, node.triangle_count, optimal_split,
            optimal_total_triangles);
        } break;
        default:
          crash_message("Invalid strategy chosen in BVH construction!");
          break;
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

      Node2 node_left = {
        .triangle_count    = optimal_split,
        .triangles_address = buffer_ptr,
        .type              = NodeTypeLeaf,
        .surface_area = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z),
        .self_high.x  = high.x,
        .self_high.y  = high.y,
        .self_high.z  = high.z,
        .self_low.x   = low.x,
        .self_low.y   = low.y,
        .self_low.z   = low.z,
      };

      nodes[write_ptr] = node_left;

      write_ptr++;
      buffer_ptr += optimal_split;

      fit_bounds(fragments_buffer + buffer_ptr, optimal_total_triangles - optimal_split, &high, &low);

      node.right_high.x = high.x;
      node.right_high.y = high.y;
      node.right_high.z = high.z;
      node.right_low.x  = low.x;
      node.right_low.y  = low.y;
      node.right_low.z  = low.z;

      Node2 node_right = {
        .triangle_count    = optimal_total_triangles - optimal_split,
        .triangles_address = buffer_ptr,
        .type              = NodeTypeLeaf,
        .surface_area = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z),
        .self_high.x  = high.x,
        .self_high.y  = high.y,
        .self_high.z  = high.z,
        .self_low.x   = low.x,
        .self_low.y   = low.y,
        .self_low.z   = low.z,
      };

      nodes[write_ptr] = node_right;

      write_ptr++;
      buffer_ptr += optimal_total_triangles - optimal_split;

      node.triangles_address = -1;
      node.type              = NodeTypeInternal;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;
    leaf_node_count += leaf_nodes_in_iteration;

    if (fragments_ptr != fragments_length) {
      memcpy(fragments_buffer + buffer_ptr, fragments + fragments_ptr, sizeof(Fragment) * (fragments_length - fragments_ptr));
      buffer_ptr += fragments_length - fragments_ptr;
      fragments_ptr += fragments_length - fragments_ptr;
    }

    _mm_free(fragments);
    fragments               = fragments_buffer;
    fragments_length        = fragments_buffer_count;
    fragments_buffer        = _mm_malloc(sizeof(Fragment) * fragments_length * 2, 64);
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

#if BINARY_BVH_ERROR_CHECK
  for (unsigned int i = 0; i < node_count; i++) {
    Node2 node = nodes[i];

    if (node.type == NodeTypeNull) {
      printf("============== Node NULL ===========\n");
      printf("index: %d (%d)\n", i, node_count);
    }

    if (node.type != NodeTypeLeaf)
      continue;

    int dim_errors = 0;
    dim_errors += (node.self_low.x >= node.self_high.x);
    dim_errors += (node.self_low.y >= node.self_high.y);
    dim_errors += (node.self_low.z >= node.self_high.z);

    if (dim_errors) {
      printf("============== NO VOLUME ===========\n");
      printf("self_high: %f %f %f\n", node.self_high.x, node.self_high.y, node.self_high.z);
      printf("self_low: %f %f %f\n", node.self_low.x, node.self_low.y, node.self_low.z);
      continue;
    }
  }
#endif

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
  if (binary_nodes[index].type == NodeTypeLeaf)
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
    if (cost < min) {
      *decision = k + 1;
      min       = cost;
    }
  }

  return min;
}

#define OPTIMAL_LEAF 0xffff

static void compute_single_node_costs(Node2* binary_nodes, const int index) {
  if (binary_nodes[index].type == NodeTypeLeaf) {
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
    node->type                       = NodeTypeLeaf;
    node_data.binary_children[slot]  = *node;
    node_data.low[slot]              = node->self_low;
    node_data.high[slot]             = node->self_high;
    node_data.binary_addresses[slot] = node_index;
  }
  else if (decision_index == 0) {
    node->type                       = NodeTypeInternal;
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
        binary_children[i].type = NodeTypeNull;
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

      vec3 node_low  = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};
      vec3 node_high = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};

      for (int i = 0; i < 8; i++) {
        if (binary_children[i].type != NodeTypeNull) {
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
          if (binary_children[j].type == NodeTypeNull) {
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
        if (base_child.type == NodeTypeInternal) {
          node.imask |= 1 << i;
          nodes[write_ptr++].child_node_base_index = binary_addresses[i];
          node.meta[i]                             = 0b00100000 + 0b11000 + i;
        }
        else if (base_child.type == NodeTypeLeaf) {
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
        if (binary_children[i].type == NodeTypeNull) {
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
