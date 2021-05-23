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
#define SAH_SAMPLES 250

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

static float get_value_by_axis(const vec3_p p, const int axis) {
  return (axis) ? ((axis == 1) ? p.y : p.z) : p.x;
}

static void fit_bounds(
  const bvh_triangle* triangles, const int triangles_length, vec3_p* high_out, vec3_p* low_out,
  vec3_p* high_out_inner, vec3_p* low_out_inner) {
  __m256 high   = _mm256_set1_ps(-FLT_MAX);
  __m256 low    = _mm256_set1_ps(FLT_MAX);
  __m128 high_a = _mm_setr_ps(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  __m128 low_a  = _mm_setr_ps(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

  for (unsigned int i = 0; i < triangles_length; i++) {
    const float* baseptr = (float*) (triangles + i);

    __m256 bc = _mm256_load_ps(baseptr);
    __m128 a  = _mm_load_ps(baseptr + 8);

    __m256 a_ext = _mm256_castps128_ps256(a);
    a_ext        = _mm256_insertf128_ps(a_ext, a, 0b1);
    bc           = _mm256_add_ps(a_ext, bc);

    high   = _mm256_max_ps(bc, high);
    low    = _mm256_min_ps(bc, low);
    high_a = _mm_max_ps(a, high_a);
    low_a  = _mm_min_ps(a, low_a);
  }

  __m128 high_b = _mm256_castps256_ps128(high);
  high          = _mm256_permute2f128_ps(high, high, 0b00000001);
  __m128 high_c = _mm256_castps256_ps128(high);

  __m128 high_inner = _mm_min_ps(high_a, high_b);
  high_inner        = _mm_min_ps(high_inner, high_c);

  high_a = _mm_max_ps(high_a, high_b);
  high_a = _mm_max_ps(high_a, high_c);

  __m128 low_b = _mm256_castps256_ps128(low);
  low          = _mm256_permute2f128_ps(low, low, 0b00000001);
  __m128 low_c = _mm256_castps256_ps128(low);

  __m128 low_inner = _mm_max_ps(low_a, low_b);
  low_inner        = _mm_max_ps(low_inner, low_c);

  low_a = _mm_min_ps(low_a, low_b);
  low_a = _mm_min_ps(low_a, low_c);

  if (high_out)
    _mm_storeu_ps((float*) high_out, high_a);
  if (low_out)
    _mm_storeu_ps((float*) low_out, low_a);
  if (high_out_inner)
    _mm_storeu_ps((float*) high_out_inner, high_inner);
  if (low_out_inner)
    _mm_storeu_ps((float*) low_out_inner, low_inner);
}

static void divide_along_axis(
  const float split, const int axis, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length,
  const int use_centroid) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;

    if (use_centroid) {
      is_right =
        (split < get_value_by_axis(triangle.vertex, axis)
                   + 0.5f * get_value_by_axis(triangle.edge1, axis)
                   + 0.5f * get_value_by_axis(triangle.edge2, axis));
    }
    else {
      const float shifted_split = split - get_value_by_axis(triangle.vertex, axis);

      is_right += 0.0f >= shifted_split;
      is_right += get_value_by_axis(triangle.edge1, axis) >= shifted_split;
      is_right += get_value_by_axis(triangle.edge2, axis) >= shifted_split;
    }

    if (is_right)
      triangles_right[right++] = triangle;
    else
      triangles_left[left++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

Node2* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, int* nodes_length_out) {
  Triangle* triangles = *triangles_io;
  int node_count      = 1 + triangles_length / THRESHOLD_TRIANGLES;

  Node2* nodes = (Node2*) malloc(sizeof(Node2) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node2) * node_count);

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = triangles_length;
  nodes[0].leaf_node         = 1;

  bvh_triangle* bvh_triangles = _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle t      = triangles[i];
    bvh_triangle bt = {
      .vertex = {.x = t.vertex.x, .y = t.vertex.y, .z = t.vertex.z},
      .edge1  = {.x = t.edge1.x, .y = t.edge1.y, .z = t.edge1.z},
      .edge2  = {.x = t.edge2.x, .y = t.edge2.y, .z = t.edge2.z},
      .id     = i};
    bvh_triangles[i] = bt;
  }

  int begin_of_current_nodes = 0;
  int end_of_current_nodes   = 1;
  int write_ptr              = 1;

  bvh_triangle* new_triangles =
    (bvh_triangle*) _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);
  bvh_triangle* container = (bvh_triangle*) _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);

  memcpy(new_triangles, bvh_triangles, sizeof(bvh_triangle) * triangles_length);

  while (begin_of_current_nodes != end_of_current_nodes) {
    int triangles_ptr = 0;

    for (int node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes; node_ptr++) {
      int left, right;
      Node2 node = nodes[node_ptr];

      triangles_ptr = node.triangles_address;

      if (node.triangle_count <= THRESHOLD_TRIANGLES) {
        continue;
      }

      vec3_p high, low;
      vec3_p high_inner, low_inner;

      fit_bounds(
        bvh_triangles + node.triangles_address, node.triangle_count, &high, &low, &high_inner,
        &low_inner);

      float search_start, search_end;
      float optimal_split, optimal_cost;
      int axis, optimal_method;

      optimal_cost = FLT_MAX;

      for (int a = 0; a < 3; a++) {
        if (a == 0) {
          search_start = low_inner.x;
          search_end   = high_inner.x;
        }
        else if (a == 1) {
          search_start = low_inner.y;
          search_end   = high_inner.y;
        }
        else {
          search_start = low_inner.z;
          search_end   = high_inner.z;
        }

        int method = 0;

        if (search_end <= search_start) {
          if (a == 0) {
            search_start = low.x;
            search_end   = high.x;
          }
          else if (a == 1) {
            search_start = low.y;
            search_end   = high.y;
          }
          else {
            search_start = low.z;
            search_end   = high.z;
          }
          method = 1;
        }

        const float search_interval = (search_end - search_start) / SAH_SAMPLES;

        int total_right = node.triangle_count;
        memcpy(
          container, bvh_triangles + node.triangles_address, sizeof(bvh_triangle) * total_right);
        int total_left = 0;

        for (int k = 0; k < SAH_SAMPLES; k++) {
          const float split = search_start + k * search_interval;
          vec3_p high_left, high_right, low_left, low_right;

          divide_along_axis(
            split, a, &left, &right, new_triangles + triangles_ptr + total_left, container,
            container, total_right, method);

          if (left == 0) {
            continue;
          }

          if (right == 0) {
            break;
          }

          total_left += left;
          total_right -= left;

          fit_bounds(
            new_triangles + triangles_ptr, total_left, &high_left, &low_left, (vec3_p*) 0,
            (vec3_p*) 0);
          fit_bounds(container, total_right, &high_right, &low_right, (vec3_p*) 0, (vec3_p*) 0);

          vec3 diff_left = {
            .x = high_left.x - low_left.x,
            .y = high_left.y - low_left.y,
            .z = high_left.z - low_left.z};

          vec3 diff_right = {
            .x = high_right.x - low_right.x,
            .y = high_right.y - low_right.y,
            .z = high_right.z - low_right.z};

          const float cost_L =
            diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;

          const float cost_R =
            diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

          const float total_cost = cost_L * total_left + cost_R * total_right;

          if (total_cost < optimal_cost) {
            optimal_cost   = total_cost;
            optimal_split  = split;
            axis           = a;
            optimal_method = method;
          }
          else {
            k += SAH_SAMPLES / 10;
          }
        }
      }

      divide_along_axis(
        optimal_split, axis, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
        bvh_triangles + node.triangles_address, node.triangle_count, optimal_method);

      int actual_split = 1;

      if (left == 0 || right == 0) {
        if (left == 0) {
          memcpy(
            new_triangles + triangles_ptr, bvh_triangles,
            sizeof(bvh_triangle) * node.triangle_count);
        }

        left         = node.triangle_count / 2;
        right        = node.triangle_count - left;
        actual_split = 0;
      }

      fit_bounds(new_triangles + triangles_ptr, left, &high, &low, (vec3_p*) 0, (vec3_p*) 0);

      node.left_high.x   = high.x;
      node.left_high.y   = high.y;
      node.left_high.z   = high.z;
      node.left_low.x    = low.x;
      node.left_low.y    = low.y;
      node.left_low.z    = low.z;
      node.child_address = write_ptr;

      nodes[write_ptr].triangle_count    = left;
      nodes[write_ptr].triangles_address = triangles_ptr;
      nodes[write_ptr].leaf_node         = 1;

      write_ptr++;
      triangles_ptr += left;

      if (write_ptr == node_count) {
        node_count += triangles_length / THRESHOLD_TRIANGLES;
        nodes = safe_realloc(nodes, sizeof(Node2) * node_count);
      }

      if (actual_split)
        memcpy(new_triangles + triangles_ptr, bvh_triangles, right * sizeof(bvh_triangle));

      fit_bounds(new_triangles + triangles_ptr, right, &high, &low, (vec3_p*) 0, (vec3_p*) 0);

      node.right_high.x = high.x;
      node.right_high.y = high.y;
      node.right_high.z = high.z;
      node.right_low.x  = low.x;
      node.right_low.y  = low.y;
      node.right_low.z  = low.z;

      nodes[write_ptr].triangle_count    = right;
      nodes[write_ptr].triangles_address = triangles_ptr;
      nodes[write_ptr].leaf_node         = 1;

      write_ptr++;
      triangles_ptr += right;

      if (write_ptr == node_count) {
        node_count += triangles_length / THRESHOLD_TRIANGLES;
        nodes = safe_realloc(nodes, sizeof(Node2) * node_count);
      }

      node.triangle_count    = 0;
      node.triangles_address = -1;
      node.leaf_node         = 0;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;

    memcpy(bvh_triangles, new_triangles, sizeof(bvh_triangle) * triangles_length);

    printf("\r                                                      \rBVH Nodes: %d", write_ptr);
  }

  _mm_free(new_triangles);
  _mm_free(container);

  node_count = write_ptr;

  nodes = safe_realloc(nodes, sizeof(Node2) * node_count);

  Triangle* triangles_swap = malloc(sizeof(Triangle) * triangles_length);
  memcpy(triangles_swap, triangles, sizeof(Triangle) * triangles_length);

  for (unsigned int i = 0; i < triangles_length; i++) {
    triangles[i] = triangles_swap[bvh_triangles[i].id];
  }

  free(triangles_swap);
  _mm_free(bvh_triangles);

  *triangles_io = triangles;

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
