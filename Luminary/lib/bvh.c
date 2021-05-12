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

static const int samples_in_space    = 250;
static const int threshold_triangles = 8;

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

static void divide_along_x_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;

    const float shifted_split = split - triangle.vertex.x;

    is_right += 0.0f >= shifted_split;
    is_right += triangle.edge1.x >= shifted_split;
    is_right += triangle.edge2.x >= shifted_split;

    if (is_right)
      triangles_right[right++] = triangle;
    else
      triangles_left[left++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_y_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;

    const float shifted_split = split - triangle.vertex.y;

    is_right += 0.0f >= shifted_split;
    is_right += triangle.edge1.y >= shifted_split;
    is_right += triangle.edge2.y >= shifted_split;

    if (is_right)
      triangles_right[right++] = triangle;
    else
      triangles_left[left++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_z_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;

    const float shifted_split = split - triangle.vertex.z;

    is_right += 0.0f >= shifted_split;
    is_right += triangle.edge1.z >= shifted_split;
    is_right += triangle.edge2.z >= shifted_split;

    if (is_right)
      triangles_right[right++] = triangle;
    else
      triangles_left[left++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, const int max_depth,
  int* nodes_length_out) {
  Triangle* triangles = *triangles_io;
  int node_count      = 1 + triangles_length / 10;

  Node* nodes = (Node*) malloc(sizeof(Node) * node_count);

  assert((unsigned long long) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node) * node_count);

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

  const double compression_divisor = 1.0 / 255.0;

  int node_ptr               = 0;
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
      Node node = nodes[node_ptr];

      triangles_ptr = node.triangles_address;

      if (node.triangle_count <= threshold_triangles) {
        continue;
      }

      vec3_p high, low;
      vec3_p high_inner, low_inner;

      fit_bounds(
        bvh_triangles + node.triangles_address, node.triangle_count, &high, &low, &high_inner,
        &low_inner);

      node.ex = (int8_t) ceil(log2f((high.x - low.x) * compression_divisor));
      node.ey = (int8_t) ceil(log2f((high.y - low.y) * compression_divisor));
      node.ez = (int8_t) ceil(log2f((high.z - low.z) * compression_divisor));

      vec3 tp = {.x = low.x, .y = low.y, .z = low.z};
      node.p  = tp;

      const float compression_x = 1.0f / exp2f(node.ex);
      const float compression_y = 1.0f / exp2f(node.ey);
      const float compression_z = 1.0f / exp2f(node.ez);

      int axis;
      float search_start, search_end;
      float optimal_split, optimal_cost;

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

        if (search_end < search_start)
          continue;

        const float search_interval = (search_end - search_start) / samples_in_space;

        int total_right = node.triangle_count;
        memcpy(
          container, bvh_triangles + node.triangles_address, sizeof(bvh_triangle) * total_right);
        int total_left = 0;

        for (int k = 0; k < samples_in_space; k++) {
          const float split = search_start + k * search_interval;
          vec3_p high_left, high_right, low_left, low_right;

          if (a == 0) {
            divide_along_x_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right);
          }
          else if (a == 1) {
            divide_along_y_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right);
          }
          else {
            divide_along_z_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right);
          }

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
            optimal_cost  = total_cost;
            optimal_split = split;
            axis          = a;
          }
          else {
            k += samples_in_space / 10;
          }
        }
      }

      if (axis == 0) {
        divide_along_x_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count);
      }
      else if (axis == 1) {
        divide_along_y_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count);
      }
      else {
        divide_along_z_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count);
      }

      if (left == 0 || right == 0) {
        if (left == 0) {
          memcpy(
            new_triangles + triangles_ptr, bvh_triangles + triangles_ptr,
            sizeof(bvh_triangle) * node.triangle_count);
        }

        continue;
      }

      compressed_vec3 compressed_low, compressed_high;

      fit_bounds(new_triangles + triangles_ptr, left, &high, &low, (vec3_p*) 0, (vec3_p*) 0);

      compressed_low.x = (uint8_t) floor((low.x - node.p.x) * compression_x);
      compressed_low.y = (uint8_t) floor((low.y - node.p.y) * compression_y);
      compressed_low.z = (uint8_t) floor((low.z - node.p.z) * compression_z);

      compressed_high.x = (uint8_t) ceil((high.x - node.p.x) * compression_x);
      compressed_high.y = (uint8_t) ceil((high.y - node.p.y) * compression_y);
      compressed_high.z = (uint8_t) ceil((high.z - node.p.z) * compression_z);

      node.left_high = compressed_high;
      node.left_low  = compressed_low;

      nodes[write_ptr].triangle_count    = left;
      nodes[write_ptr].triangles_address = triangles_ptr;
      nodes[write_ptr].leaf_node         = 1;

      write_ptr++;
      triangles_ptr += left;

      if (write_ptr == node_count) {
        node_count += triangles_length;
        nodes = safe_realloc(nodes, sizeof(Node) * node_count);
      }

      memcpy(new_triangles + triangles_ptr, bvh_triangles, right * sizeof(bvh_triangle));

      fit_bounds(new_triangles + triangles_ptr, right, &high, &low, (vec3_p*) 0, (vec3_p*) 0);

      compressed_low.x = (uint8_t) floor((low.x - node.p.x) * compression_x);
      compressed_low.y = (uint8_t) floor((low.y - node.p.y) * compression_y);
      compressed_low.z = (uint8_t) floor((low.z - node.p.z) * compression_z);

      compressed_high.x = (uint8_t) ceil((high.x - node.p.x) * compression_x);
      compressed_high.y = (uint8_t) ceil((high.y - node.p.y) * compression_y);
      compressed_high.z = (uint8_t) ceil((high.z - node.p.z) * compression_z);

      node.right_high = compressed_high;
      node.right_low  = compressed_low;

      nodes[write_ptr].triangle_count    = right;
      nodes[write_ptr].triangles_address = triangles_ptr;
      nodes[write_ptr].leaf_node         = 1;

      write_ptr++;
      triangles_ptr += right;

      if (write_ptr == node_count) {
        node_count += triangles_length;
        nodes = safe_realloc(nodes, sizeof(Node) * node_count);
      }

      node.triangle_count    = 0;
      node.triangles_address = -1;
      node.leaf_node         = 0;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;

    memcpy(bvh_triangles, new_triangles, sizeof(bvh_triangle) * triangles_length);
  }

  _mm_free(new_triangles);
  _mm_free(container);

  node_count = write_ptr;

  printf("BVH with %d Nodes built.", node_count);

  nodes = safe_realloc(nodes, sizeof(Node) * node_count);

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
