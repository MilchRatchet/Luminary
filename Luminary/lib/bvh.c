#include "bvh.h"
#include "mesh.h"
#include "primitives.h"
#include "error.h"
#include <float.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#define THRESHOLD_TRIANGLES 3
#define MAXIMUM_BIN_COUNT 1024
#define SPATIAL_SPLIT_THRESHOLD 1.0000f
#define SPATIAL_SPLIT_BIN_COUNT 1024

struct vec3_p {
  float x;
  float y;
  float z;
  float _p;
} typedef vec3_p;

struct bvh_triangle {
  vec3_p edge1;
  vec3_p edge2;
  vec3_p vertex;
  uint32_t id;
  uint32_t _p1;
  uint64_t _p2;
} typedef bvh_triangle;

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

static float get_sweeping_bound_by_axis(const bvh_triangle triangle, const int axis) {
  return (axis) ? ((axis == 1) ? triangle.edge1._p : triangle.edge2._p) : triangle.vertex._p;
}

static void __swap_bvh_triangle(bvh_triangle* a, bvh_triangle* b) {
  bvh_triangle temp = *a;
  *a                = *b;
  *b                = temp;
}

static int __partition(bvh_triangle* triangles, int bottom, int top, const int axis) {
  const int mid = (top - bottom) / 2 + bottom;
  if (
    get_sweeping_bound_by_axis(triangles[top], axis)
    < get_sweeping_bound_by_axis(triangles[bottom], axis)) {
    __swap_bvh_triangle(triangles + bottom, triangles + top);
  }
  if (
    get_sweeping_bound_by_axis(triangles[mid], axis)
    < get_sweeping_bound_by_axis(triangles[bottom], axis)) {
    __swap_bvh_triangle(triangles + mid, triangles + bottom);
  }
  if (
    get_sweeping_bound_by_axis(triangles[top], axis)
    > get_sweeping_bound_by_axis(triangles[mid], axis)) {
    __swap_bvh_triangle(triangles + mid, triangles + top);
  }

  const float x = get_sweeping_bound_by_axis(triangles[top], axis);
  int i         = bottom - 1;

  for (int j = bottom; j < top; j++) {
    if (get_sweeping_bound_by_axis(triangles[j], axis) < x) {
      i++;
      __swap_bvh_triangle(triangles + i, triangles + j);
    }
  }
  __swap_bvh_triangle(triangles + i + 1, triangles + top);

  return (i + 1);
}

static void quick_sort_bvh_triangles(
  bvh_triangle* triangles, const unsigned int triangles_length, const int axis) {
  int ptr = 1;

  int quick_sort_stack[64];

  quick_sort_stack[0] = 0;
  quick_sort_stack[1] = triangles_length - 1;

  while (ptr >= 0) {
    const int top    = quick_sort_stack[ptr--];
    const int bottom = quick_sort_stack[ptr--];

    const int p = __partition(triangles, bottom, top, axis);

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

static void fit_bounds(
  const bvh_triangle* triangles, const unsigned int triangles_length, vec3_p* high_out,
  vec3_p* low_out) {
  __m256 high   = _mm256_set1_ps(-FLT_MAX);
  __m256 low    = _mm256_set1_ps(FLT_MAX);
  __m128 high_a = _mm_setr_ps(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  __m128 low_a  = _mm_setr_ps(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (unsigned int i = 0; i < triangles_length; i++) {
    const float* baseptr = (float*) (triangles + i);

    __m256 bc = _mm256_loadu_ps(baseptr);
    __m128 a  = _mm_loadu_ps(baseptr + 8);

    high   = _mm256_max_ps(bc, high);
    low    = _mm256_min_ps(bc, low);
    high_a = _mm_max_ps(a, high_a);
    low_a  = _mm_min_ps(a, low_a);
  }

  __m128 high_b = _mm256_castps256_ps128(high);
  high          = _mm256_permute2f128_ps(high, high, 0b00000001);
  __m128 high_c = _mm256_castps256_ps128(high);

  high_a = _mm_max_ps(high_a, high_b);
  high_a = _mm_max_ps(high_a, high_c);

  __m128 low_b = _mm256_castps256_ps128(low);
  low          = _mm256_permute2f128_ps(low, low, 0b00000001);
  __m128 low_c = _mm256_castps256_ps128(low);

  low_a = _mm_min_ps(low_a, low_b);
  low_a = _mm_min_ps(low_a, low_c);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high_a);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low_a);
}

static void update_bounds(const bvh_triangle* triangle, vec3_p* high_out, vec3_p* low_out) {
  const float* baseptr = (float*) triangle;

  __m128 high_a = _mm_loadu_ps((float*) high_out);
  __m128 low_a  = _mm_loadu_ps((float*) low_out);
  __m128 a      = _mm_load_ps(baseptr);
  __m128 high_b = _mm_load_ps(baseptr + 4);
  __m128 low_b  = high_b;
  __m128 high_c = _mm_load_ps(baseptr + 8);
  __m128 low_c  = high_c;

  high_a = _mm_max_ps(high_a, a);
  low_a  = _mm_min_ps(low_a, a);
  high_a = _mm_max_ps(high_a, high_b);
  low_a  = _mm_min_ps(low_a, low_b);
  high_a = _mm_max_ps(high_a, high_c);
  low_a  = _mm_min_ps(low_a, low_c);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high_a);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low_a);
}

static void fit_bounds_of_bins(
  const bin* bins, const int bins_length, vec3_p* high_out, vec3_p* low_out) {
  __m256 high = _mm256_set1_ps(-FLT_MAX);
  __m256 low  = _mm256_set1_ps(FLT_MAX);

  for (int i = 0; i < bins_length; i++) {
    const float* baseptr = (float*) (bins + i);

    __m256 b = _mm256_loadu_ps(baseptr);

    high = _mm256_max_ps(b, high);
    low  = _mm256_min_ps(b, low);
  }

  __m128 high_a = _mm256_castps256_ps128(high);
  high          = _mm256_permute2f128_ps(high, high, 0b00000001);
  __m128 high_b = _mm256_castps256_ps128(high);

  high_a = _mm_max_ps(high_a, high_b);

  __m128 low_a = _mm256_castps256_ps128(low);
  low          = _mm256_permute2f128_ps(low, low, 0b00000001);
  __m128 low_b = _mm256_castps256_ps128(low);

  low_a = _mm_min_ps(low_a, low_b);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high_a);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low_a);
}

static void update_bounds_of_bins(const bin* bins, vec3_p* high_out, vec3_p* low_out) {
  __m128 high_a = _mm_loadu_ps((float*) high_out);
  __m128 low_a  = _mm_loadu_ps((float*) low_out);
  __m128 high_b = _mm_set1_ps(-FLT_MAX);
  __m128 low_b  = _mm_set1_ps(FLT_MAX);

  const float* baseptr = (float*) (bins);

  __m128 a = _mm_load_ps(baseptr);
  __m128 b = _mm_load_ps(baseptr + 4);

  high_a = _mm_max_ps(a, high_a);
  low_a  = _mm_min_ps(a, low_a);
  high_b = _mm_max_ps(b, high_b);
  low_b  = _mm_min_ps(b, low_b);
  high_a = _mm_max_ps(high_a, high_b);
  low_a  = _mm_min_ps(low_a, low_b);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high_a);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low_a);
}

static float construct_chopped_bins(
  bin* restrict bins, const bvh_triangle* restrict triangles, const unsigned int triangles_length,
  const int axis) {
  vec3_p high, low;
  fit_bounds(triangles, triangles_length, &high, &low);
  const float span     = get_entry_by_axis(high, axis) - get_entry_by_axis(low, axis);
  const float interval = span / SPATIAL_SPLIT_BIN_COUNT;

  for (int i = 0; i < SPATIAL_SPLIT_BIN_COUNT; i++) {
    bin b;
    b.high.x = -FLT_MAX;
    b.high.y = -FLT_MAX;
    b.high.z = -FLT_MAX;
    b.low.x  = FLT_MAX;
    b.low.y  = FLT_MAX;
    b.low.z  = FLT_MAX;
    b.entry  = 0;
    b.exit   = 0;
    bins[i]  = b;
  }

  for (int i = 0; i < triangles_length; i++) {
    const bvh_triangle t = triangles[i];

    vec3_p high_triangle = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX, ._p = -FLT_MAX};
    vec3_p low_triangle  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX, ._p = FLT_MAX};

    update_bounds(&t, &high_triangle, &low_triangle);

    const float value1 = get_entry_by_axis(t.vertex, axis);
    const int pos1     = (int) (value1 / interval);
    const float value2 = get_entry_by_axis(t.edge1, axis);
    const int pos2     = (int) (value2 / interval);
    const float value3 = get_entry_by_axis(t.edge2, axis);
    const int pos3     = (int) (value3 / interval);

    const int entry = min(pos1, min(pos2, pos3));
    const int exit  = max(pos1, max(pos2, pos3));

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

static void divide_along_axis(
  const float split, const int axis, bvh_triangle* triangles_out, bvh_triangle* triangles_in,
  const unsigned int triangles_length, const int right_offset) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles_in[i];

    int is_right = 0;
    int is_left  = 0;

    const float shifted_split = split - get_entry_by_axis(triangle.vertex, axis);

    is_right += 0.0f >= shifted_split;
    is_right += get_entry_by_axis(triangle.edge1, axis) >= shifted_split;
    is_right += get_entry_by_axis(triangle.edge2, axis) >= shifted_split;

    is_left += 0.0f < shifted_split;
    is_left += get_entry_by_axis(triangle.edge1, axis) < shifted_split;
    is_left += get_entry_by_axis(triangle.edge2, axis) < shifted_split;

    if (is_left)
      triangles_out[left++] = triangle;
    if (is_right)
      triangles_out[right_offset + right++] = triangle;
  }
}

Node2* build_bvh_structure(
  Triangle** triangles_io, unsigned int* triangles_length_io, int* nodes_length_out) {
  unsigned int triangles_length = *triangles_length_io;
  Triangle* triangles           = *triangles_io;
  unsigned int node_count       = 1 + triangles_length / THRESHOLD_TRIANGLES;

  const unsigned int initial_triangles_length = triangles_length;

  Node2* nodes = (Node2*) malloc(sizeof(Node2) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);
  assert(
    (SPATIAL_SPLIT_BIN_COUNT <= MAXIMUM_BIN_COUNT),
    "Build error! There can't be more spatial split bins than normal bins. Change the value and "
    "rebuild!",
    1);

  memset(nodes, 0, sizeof(Node2) * node_count);

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = triangles_length;
  nodes[0].leaf_node         = 1;

  bvh_triangle* bvh_triangles = _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle t = triangles[i];
    vec3 maximum;
    maximum.x = max(t.vertex.x, max(t.vertex.x + t.edge1.x, t.vertex.x + t.edge2.x));
    maximum.y = max(t.vertex.y, max(t.vertex.y + t.edge1.y, t.vertex.y + t.edge2.y));
    maximum.z = max(t.vertex.z, max(t.vertex.z + t.edge1.z, t.vertex.z + t.edge2.z));

    bvh_triangle bt = {
      .vertex = {.x = t.vertex.x, .y = t.vertex.y, .z = t.vertex.z, ._p = maximum.x},
      .edge1 =
        {.x  = t.vertex.x + t.edge1.x,
         .y  = t.vertex.y + t.edge1.y,
         .z  = t.vertex.z + t.edge1.z,
         ._p = maximum.y},
      .edge2 =
        {.x  = t.vertex.x + t.edge2.x,
         .y  = t.vertex.y + t.edge2.y,
         .z  = t.vertex.z + t.edge2.z,
         ._p = maximum.z},
      .id = i};
    bvh_triangles[i] = bt;
  }

  bvh_triangle* bvh_triangles_buffer = _mm_malloc(sizeof(bvh_triangle) * triangles_length * 2, 64);
  unsigned int triangles_buffer_length = triangles_length * 2;

  float root_surface_area;
  {
    vec3_p high_root, low_root;
    fit_bounds(bvh_triangles, triangles_length, &high_root, &low_root);
    vec3_p diff = {
      .x  = high_root.x - low_root.x,
      .y  = high_root.y - low_root.y,
      .z  = high_root.z - low_root.z,
      ._p = high_root._p - low_root._p};

    root_surface_area = diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
  }

  bin* bins = (bin*) _mm_malloc(sizeof(bin) * MAXIMUM_BIN_COUNT, 32);

  unsigned int begin_of_current_nodes = 0;
  unsigned int end_of_current_nodes   = 1;
  int write_ptr                       = 1;

  while (begin_of_current_nodes != end_of_current_nodes) {
    unsigned int triangles_ptr = 0;
    unsigned int buffer_ptr    = 0;

    for (unsigned int node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes;
         node_ptr++) {
      Node2 node = nodes[node_ptr];

      if (triangles_ptr != node.triangles_address) {
        memcpy(
          bvh_triangles_buffer + buffer_ptr, bvh_triangles + triangles_ptr,
          sizeof(bvh_triangle) * (node.triangles_address - triangles_ptr));
        buffer_ptr += node.triangles_address - triangles_ptr;
        triangles_ptr = node.triangles_address;
      }

      if (node.triangle_count <= THRESHOLD_TRIANGLES) {
        memcpy(
          bvh_triangles_buffer + buffer_ptr, bvh_triangles + triangles_ptr,
          sizeof(bvh_triangle) * node.triangle_count);
        buffer_ptr += node.triangle_count;
        triangles_ptr += node.triangle_count;
        continue;
      }

      const int binning_size  = 1 + node.triangle_count / MAXIMUM_BIN_COUNT;
      const int bin_count     = (node.triangle_count + binning_size - 1) / binning_size;
      const int last_bin_size = node.triangle_count - (bin_count - 1) * binning_size;

      vec3_p high, low;
      vec3_p optimal_high_right, optimal_low_right, optimal_high_left, optimal_low_left;
      float optimal_cost = FLT_MAX;
      int axis, optimal_split, optimal_method;
      float optimal_splitting_plane;
      int optimal_total_triangles;

      optimal_method = 0;

      for (int a = 0; a < 3; a++) {
        quick_sort_bvh_triangles(bvh_triangles + triangles_ptr, node.triangle_count, a);

        for (int k = 0; k < bin_count; k++) {
          const int size = (k == bin_count - 1) ? last_bin_size : binning_size;
          fit_bounds(bvh_triangles + triangles_ptr + k * binning_size, size, &high, &low);
          bin b   = {.high = high, .low = low};
          bins[k] = b;
        }

        vec3_p high_left  = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
        vec3_p high_right = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
        vec3_p low_left   = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};
        vec3_p low_right  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};

        for (int k = 1; k < bin_count - 1; k++) {
          update_bounds_of_bins(bins + k - 1, &high_left, &low_left);
          fit_bounds_of_bins(bins + k, bin_count - k, &high_right, &low_right);

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

          const float cost_L =
            diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;

          const float cost_R =
            diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

          const float total_cost = cost_L * k * binning_size
                                   + cost_R * ((bin_count - k - 1) * binning_size + last_bin_size);

          if (total_cost < optimal_cost) {
            optimal_cost       = total_cost;
            optimal_split      = k * binning_size;
            axis               = a;
            optimal_high_left  = high_left;
            optimal_high_right = high_right;
            optimal_low_left   = low_left;
            optimal_low_right  = low_right;
          }
        }
      }

      /*int do_spatial_splits;
      {
        vec3_p overlap = {
          .x = min(optimal_high_left.x, optimal_high_right.x)
               - max(optimal_low_left.x, optimal_low_right.x),
          .y = min(optimal_high_left.y, optimal_high_right.y)
               - max(optimal_low_left.y, optimal_low_right.y),
          .z = min(optimal_high_left.z, optimal_high_right.z)
               - max(optimal_low_left.z, optimal_low_right.z),
          ._p = min(optimal_high_left._p, optimal_high_right._p)
                - max(optimal_low_left._p, optimal_low_right._p),
        };

        if (overlap.x > 0.0f && overlap.y > 0.0f && overlap.z > 0.0f) {
          do_spatial_splits = 0;
        }
        else {
          const float o     = overlap.x * overlap.y + overlap.x * overlap.z + overlap.y * overlap.z;
          do_spatial_splits = ((o / root_surface_area) > SPATIAL_SPLIT_THRESHOLD);
        }
      }

      if (do_spatial_splits) {
        printf("SPATIAL SPLIT AT NODE %d\n", node_ptr);
        for (int a = 0; a < 3; a++) {
          const float interval =
            construct_chopped_bins(bins, bvh_triangles, node.triangle_count, a);

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

            const float cost_L =
              diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;

            const float cost_R = diff_right.x * diff_right.y + diff_right.x * diff_right.z
                                 + diff_right.y * diff_right.z;

            left += bins[k - 1].entry;
            right -= bins[k - 1].exit;

            const float total_cost = cost_L * left + cost_R * right;

            if (total_cost < optimal_cost) {
              optimal_cost            = total_cost;
              optimal_split           = left;
              optimal_total_triangles = left + right;
              optimal_splitting_plane = k * interval;
              optimal_method          = 1;
              axis                    = a;
              optimal_high_left       = high_left;
              optimal_high_right      = high_right;
              optimal_low_left        = low_left;
              optimal_low_right       = low_right;
            }
          }
        }
      }*/

      if (optimal_method == 0) {
        if (axis != 2) {
          quick_sort_bvh_triangles(bvh_triangles + triangles_ptr, node.triangle_count, axis);
        }

        printf("Object Split: %f\n", ((float) (optimal_split)) / node.triangle_count);

        memcpy(
          bvh_triangles_buffer + buffer_ptr, bvh_triangles + triangles_ptr,
          sizeof(bvh_triangle) * node.triangle_count);

        if (write_ptr + 2 >= node_count) {
          node_count += triangles_length / THRESHOLD_TRIANGLES;
          nodes = safe_realloc(nodes, sizeof(Node2) * node_count);
        }

        fit_bounds(bvh_triangles_buffer + buffer_ptr, optimal_split, &high, &low);

        node.left_high.x   = high.x;
        node.left_high.y   = high.y;
        node.left_high.z   = high.z;
        node.left_low.x    = low.x;
        node.left_low.y    = low.y;
        node.left_low.z    = low.z;
        node.child_address = write_ptr;

        nodes[write_ptr].triangle_count    = optimal_split;
        nodes[write_ptr].triangles_address = buffer_ptr;
        nodes[write_ptr].leaf_node         = 1;

        write_ptr++;
        triangles_ptr += optimal_split;
        buffer_ptr += optimal_split;

        fit_bounds(
          bvh_triangles_buffer + buffer_ptr, node.triangle_count - optimal_split, &high, &low);

        node.right_high.x = high.x;
        node.right_high.y = high.y;
        node.right_high.z = high.z;
        node.right_low.x  = low.x;
        node.right_low.y  = low.y;
        node.right_low.z  = low.z;

        nodes[write_ptr].triangle_count    = node.triangle_count - optimal_split;
        nodes[write_ptr].triangles_address = buffer_ptr;
        nodes[write_ptr].leaf_node         = 1;

        write_ptr++;
        triangles_ptr += node.triangle_count - optimal_split;
        buffer_ptr += node.triangle_count - optimal_split;
      }
      /*else {
        printf("SPATIAL SPLIT BEST: %d\n", optimal_total_triangles - node.triangle_count);
        triangles_length += optimal_total_triangles - node.triangle_count;

        if (triangles_length >= triangles_buffer_length) {
          triangles_buffer_length += triangles_length / 2 + 1;
          bvh_triangles_buffer =
            safe_realloc(bvh_triangles_buffer, sizeof(bvh_triangle) * triangles_buffer_length);
        }

        divide_along_axis(
          optimal_splitting_plane, axis, bvh_triangles_buffer + buffer_ptr,
          bvh_triangles + triangles_ptr, optimal_total_triangles, optimal_split);

        triangles_ptr += node.triangle_count;

        if (write_ptr + 2 >= node_count) {
          node_count += triangles_length / THRESHOLD_TRIANGLES;
          nodes = safe_realloc(nodes, sizeof(Node2) * node_count);
        }

        fit_bounds(bvh_triangles_buffer + buffer_ptr, optimal_split, &high, &low);

        node.left_high.x   = high.x;
        node.left_high.y   = high.y;
        node.left_high.z   = high.z;
        node.left_low.x    = low.x;
        node.left_low.y    = low.y;
        node.left_low.z    = low.z;
        node.child_address = write_ptr;

        nodes[write_ptr].triangle_count    = optimal_split;
        nodes[write_ptr].triangles_address = buffer_ptr;
        nodes[write_ptr].leaf_node         = 1;

        write_ptr++;
        buffer_ptr += optimal_split;

        fit_bounds(
          bvh_triangles_buffer + buffer_ptr, optimal_total_triangles - optimal_split, &high, &low);

        node.right_high.x = high.x;
        node.right_high.y = high.y;
        node.right_high.z = high.z;
        node.right_low.x  = low.x;
        node.right_low.y  = low.y;
        node.right_low.z  = low.z;

        nodes[write_ptr].triangle_count    = optimal_total_triangles - optimal_split;
        nodes[write_ptr].triangles_address = buffer_ptr;
        nodes[write_ptr].leaf_node         = 1;

        write_ptr++;
        buffer_ptr += optimal_total_triangles - optimal_split;
      }*/

      node.triangle_count    = 0;
      node.triangles_address = -1;
      node.leaf_node         = 0;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;

    if (buffer_ptr != triangles_length) {
      memcpy(
        bvh_triangles_buffer + buffer_ptr, bvh_triangles + triangles_ptr,
        sizeof(bvh_triangle) * (triangles_length - buffer_ptr));
    }

    _mm_free(bvh_triangles);
    bvh_triangles           = bvh_triangles_buffer;
    bvh_triangles_buffer    = _mm_malloc(sizeof(bvh_triangle) * triangles_length * 2, 64);
    triangles_buffer_length = triangles_length * 2;

    printf("\r                                                      \rBVH Nodes: %d", write_ptr);
  }

  node_count = write_ptr;

  nodes = safe_realloc(nodes, sizeof(Node2) * node_count);

  Triangle* triangles_swap = malloc(sizeof(Triangle) * initial_triangles_length);
  memcpy(triangles_swap, triangles, sizeof(Triangle) * initial_triangles_length);
  // free(triangles);
  // triangles = malloc(sizeof(Triangle) * triangles_length);

  for (unsigned int i = 0; i < triangles_length; i++) {
    triangles[i] = triangles_swap[bvh_triangles[i].id];
  }

  free(triangles_swap);
  _mm_free(bvh_triangles);
  _mm_free(bins);

  *triangles_io = triangles;

  *triangles_length_io = triangles_length;

  *nodes_length_out = node_count;

  return nodes;
}

#define BINARY_NODE_IS_INTERNAL_NODE 0b0
#define BINARY_NODE_IS_LEAF_NODE 0b1
#define BINARY_NODE_IS_NULL 0b10

Node8* collapse_bvh(
  Node2* binary_nodes, const int binary_nodes_length, Triangle** triangles_io,
  const int triangles_length, int* nodes_length_out) {
  Triangle* triangles = *triangles_io;
  int node_count      = binary_nodes_length;

  Node8* nodes = (Node8*) malloc(sizeof(Node8) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node8) * node_count);

  bvh_triangle* bvh_triangles = _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);
  bvh_triangle* new_triangles = _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle t      = triangles[i];
    bvh_triangle bt = {
      .vertex = {.x = t.vertex.x, .y = t.vertex.y, .z = t.vertex.z},
      .edge1  = {.x = t.edge1.x, .y = t.edge1.y, .z = t.edge1.z},
      .edge2  = {.x = t.edge2.x, .y = t.edge2.y, .z = t.edge2.z},
      .id     = i};
    bvh_triangles[i] = bt;
  }

  memcpy(new_triangles, bvh_triangles, sizeof(bvh_triangle) * triangles_length);

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
      binary_children[0] = binary_nodes[binary_index];

      int offset      = 8;
      int half_offset = 4;

      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 8; i += offset) {
          Node2 base_child = binary_children[i];

          if (base_child.leaf_node != BINARY_NODE_IS_INTERNAL_NODE) {
            binary_children[i + half_offset].leaf_node = BINARY_NODE_IS_NULL;
          }
          else {
            binary_children[i]                = binary_nodes[base_child.child_address];
            low[i]                            = base_child.left_low;
            high[i]                           = base_child.left_high;
            binary_addresses[i]               = base_child.child_address;
            binary_children[i + half_offset]  = binary_nodes[base_child.child_address + 1];
            low[i + half_offset]              = base_child.right_low;
            high[i + half_offset]             = base_child.right_high;
            binary_addresses[i + half_offset] = base_child.child_address + 1;
          }
        }
        offset      = offset >> 1;
        half_offset = half_offset >> 1;
      }

      vec3 node_low  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};
      vec3 node_high = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};

      for (int i = 0; i < 8; i++) {
        if (binary_children[i].leaf_node != BINARY_NODE_IS_NULL) {
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

      vec3 centroid = {
        .x = (node_high.x + node_low.x) * 0.5f,
        .y = (node_high.y + node_low.y) * 0.5f,
        .z = (node_high.z + node_low.z) * 0.5f};

      float cost_table[8][8];
      int order[8];
      int slot_empty[8];

      for (int i = 0; i < 8; i++) {
        slot_empty[i] = 1;
        order[i]      = -1;
      }

      for (int i = 0; i < 8; i++) {
        vec3 direction = {
          .x = ((i >> 2) & 0b1) ? -1.0f : 1.0f,
          .y = ((i >> 1) & 0b1) ? -1.0f : 1.0f,
          .z = ((i >> 0) & 0b1) ? -1.0f : 1.0f};

        for (int j = 0; j < 8; j++) {
          if (binary_children[j].leaf_node == BINARY_NODE_IS_NULL) {
            cost_table[i][j] = FLT_MAX;
          }
          else {
            vec3 child_centroid = {
              .x = ((high[j].x + low[j].x) * 0.5f) - centroid.x,
              .y = ((high[j].y + low[j].y) * 0.5f) - centroid.y,
              .z = ((high[j].z + low[j].z) * 0.5f) - centroid.z};

            cost_table[i][j] = child_centroid.x * direction.x + child_centroid.y * direction.y
                               + child_centroid.z * direction.z;
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
        if (base_child.leaf_node == BINARY_NODE_IS_INTERNAL_NODE) {
          node.imask |= 1 << i;
          nodes[write_ptr++].child_node_base_index = binary_addresses[i];
          node.meta[i]                             = 0b00100000 + 0b11000 + i;
        }
        else if (base_child.leaf_node == BINARY_NODE_IS_LEAF_NODE) {
          assert(
            base_child.triangle_count < 4,
            "Error when collapsing nodes. There are too many unsplittable triangles.", 1);
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

          memcpy(
            new_triangles + triangles_ptr, bvh_triangles + base_child.triangles_address,
            sizeof(bvh_triangle) * base_child.triangle_count);

          triangles_ptr += base_child.triangle_count;
          triangle_counting_address += base_child.triangle_count;
        }
        else {
          node.meta[i] = 0;
        }
      }

      for (int i = 0; i < 8; i++) {
        if (binary_children[i].leaf_node == BINARY_NODE_IS_NULL) {
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

  for (unsigned int i = 0; i < triangles_length; i++) {
    triangles[i] = triangles_swap[bvh_triangles[i].id];
  }

  free(triangles_swap);

  _mm_free(new_triangles);

  node_count = write_ptr;

  printf("\r                                                         \r");

  nodes = safe_realloc(nodes, sizeof(Node8) * node_count);

  *triangles_io = triangles;

  *nodes_length_out = node_count;

  return nodes;
}
