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

static const int samples_in_space    = 25;
static const int space_extension     = 2;
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
  vec3_p* average_out) {
  __m256 high             = _mm256_set1_ps(-FLT_MAX);
  __m256 low              = _mm256_set1_ps(FLT_MAX);
  __m256 average          = _mm256_set1_ps(0.0f);
  __m256 one_over_three   = _mm256_set1_ps(1.0f / 3.0f);
  __m128 high_a           = _mm_setr_ps(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  __m128 low_a            = _mm_setr_ps(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  __m128 average_a        = _mm_setr_ps(0.0f, 0.0f, 0.0f, 0.0f);
  __m128 one_over_three_a = _mm_setr_ps(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);

  for (unsigned int i = 0; i < triangles_length; i++) {
    const float* baseptr = (float*) (triangles + i);

    __m256 bc = _mm256_load_ps(baseptr);
    __m128 a  = _mm_load_ps(baseptr + 8);

    __m256 a_ext = _mm256_castps128_ps256(a);
    a_ext        = _mm256_insertf128_ps(a_ext, a, 0b1);
    bc           = _mm256_add_ps(a_ext, bc);

    high      = _mm256_max_ps(bc, high);
    low       = _mm256_min_ps(bc, low);
    average   = _mm256_fmadd_ps(bc, one_over_three, average);
    high_a    = _mm_max_ps(a, high_a);
    low_a     = _mm_min_ps(a, low_a);
    average_a = _mm_fmadd_ps(a, one_over_three_a, average_a);
  }

  __m128 high_b = _mm256_castps256_ps128(high);
  high          = _mm256_permute2f128_ps(high, high, 0b00000001);
  __m128 high_c = _mm256_castps256_ps128(high);
  high_a        = _mm_max_ps(high_a, high_b);
  high_a        = _mm_max_ps(high_a, high_c);

  __m128 low_b = _mm256_castps256_ps128(low);
  low          = _mm256_permute2f128_ps(low, low, 0b00000001);
  __m128 low_c = _mm256_castps256_ps128(low);
  low_a        = _mm_min_ps(low_a, low_b);
  low_a        = _mm_min_ps(low_a, low_c);

  __m128 average_b = _mm256_castps256_ps128(average);
  average          = _mm256_permute2f128_ps(average, average, 0b00000001);
  __m128 average_c = _mm256_castps256_ps128(average);
  average_a        = _mm_add_ps(average_a, average_b);
  average_a        = _mm_add_ps(average_a, average_c);

  const float divisor = 1.0f / (triangles_length);
  average_a           = _mm_mul_ps(average_a, _mm_setr_ps(divisor, divisor, divisor, divisor));

  _mm_storeu_ps((float*) high_out, high_a);
  _mm_storeu_ps((float*) low_out, low_a);
  _mm_storeu_ps((float*) average_out, average_a);
}

static void divide_along_x_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length,
  const int left_right_bias) {
  int left  = 0;
  int right = 0;

  const int bias = left_right_bias % 2;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.x >= split;
    is_left += triangle.vertex.x <= split;
    is_right += triangle.vertex.x + triangle.edge1.x >= split;
    is_left += triangle.vertex.x + triangle.edge1.x <= split;
    is_right += triangle.vertex.x + triangle.edge2.x >= split;
    is_left += triangle.vertex.x + triangle.edge2.x <= split;

    if (bias) {
      if (is_left)
        triangles_left[left++] = triangle;
      else if (is_right)
        triangles_right[right++] = triangle;
    }
    else {
      if (is_right)
        triangles_right[right++] = triangle;
      else if (is_left)
        triangles_left[left++] = triangle;
    }
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_y_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length,
  const int left_right_bias) {
  int left  = 0;
  int right = 0;

  const int bias = left_right_bias % 2;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.y >= split;
    is_left += triangle.vertex.y <= split;
    is_right += triangle.vertex.y + triangle.edge1.y >= split;
    is_left += triangle.vertex.y + triangle.edge1.y <= split;
    is_right += triangle.vertex.y + triangle.edge2.y >= split;
    is_left += triangle.vertex.y + triangle.edge2.y <= split;

    if (bias) {
      if (is_left)
        triangles_left[left++] = triangle;
      else if (is_right)
        triangles_right[right++] = triangle;
    }
    else {
      if (is_right)
        triangles_right[right++] = triangle;
      else if (is_left)
        triangles_left[left++] = triangle;
    }
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_z_axis(
  const float split, int* left_out, int* right_out, bvh_triangle* triangles_left,
  bvh_triangle* triangles_right, bvh_triangle* triangles, const unsigned int triangles_length,
  const int left_right_bias) {
  int left  = 0;
  int right = 0;

  const int bias = left_right_bias % 2;

  for (int i = 0; i < triangles_length; i++) {
    bvh_triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.z >= split;
    is_left += triangle.vertex.z <= split;
    is_right += triangle.vertex.z + triangle.edge1.z >= split;
    is_left += triangle.vertex.z + triangle.edge1.z <= split;
    is_right += triangle.vertex.z + triangle.edge2.z >= split;
    is_left += triangle.vertex.z + triangle.edge2.z <= split;

    if (bias) {
      if (is_left)
        triangles_left[left++] = triangle;
      else if (is_right)
        triangles_right[right++] = triangle;
    }
    else {
      if (is_right)
        triangles_right[right++] = triangle;
      else if (is_left)
        triangles_left[left++] = triangle;
    }
  }

  *left_out  = left;
  *right_out = right;
}

Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int triangles_length, const int max_depth,
  int* nodes_length_out) {
  Triangle* triangles = *triangles_io;
  int pow             = 2;
  int node_count      = 1;

  for (int i = 0; i < max_depth; i++) {
    node_count += pow;
    pow *= 2;
  }

  Node* nodes = (Node*) malloc(sizeof(Node) * node_count);

  assert((int) nodes, "Failed to allocate BVH nodes!", 1);

  memset(nodes, 0, sizeof(Node) * node_count);

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = triangles_length;

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

  unsigned int nodes_at_depth = 1;

  int node_ptr = 0;

  const double compression_divisor = 1.0 / 255.0;

  for (int i = 0; i < max_depth; i++) {
    unsigned int triangles_ptr = 0;
    bvh_triangle* new_triangles =
      (bvh_triangle*) _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);
    bvh_triangle* container =
      (bvh_triangle*) _mm_malloc(sizeof(bvh_triangle) * triangles_length, 64);

    for (int j = 0; j < nodes_at_depth; j++) {
      vec3_p high, low, node_average, average;
      int left, right;
      Node node = nodes[node_ptr];

      if (node.triangle_count == 0) {
        node_ptr++;
        continue;
      }

      fit_bounds(
        bvh_triangles + node.triangles_address, node.triangle_count, &high, &low, &node_average);

      node.ex = (int8_t) ceil(log2f((high.x - low.x) * compression_divisor));
      node.ey = (int8_t) ceil(log2f((high.y - low.y) * compression_divisor));
      node.ez = (int8_t) ceil(log2f((high.z - low.z) * compression_divisor));

      vec3 tp = {.x = low.x, .y = low.y, .z = low.z};
      node.p  = tp;

      const float compression_x = 1.0f / exp2f(node.ex);
      const float compression_y = 1.0f / exp2f(node.ey);
      const float compression_z = 1.0f / exp2f(node.ez);

      int axis;
      float split_spatial, split_object;
      float optimal_split, optimal_cost;

      optimal_cost = FLT_MAX;

      for (int a = 0; a < 3; a++) {
        if (a == 0) {
          split_spatial = (high.x + low.x) / 2.0f;
          split_object  = node_average.x;
        }
        else if (a == 1) {
          split_spatial = (high.y + low.y) / 2.0f;
          split_object  = node_average.y;
        }
        else {
          split_spatial = (high.z + low.z) / 2.0f;
          split_object  = node_average.z;
        }

        const float search_interval = fabsf((split_object - split_spatial) / samples_in_space);
        const float search_start    = min(split_object, split_spatial);

        int total_right = node.triangle_count;
        memcpy(
          container, bvh_triangles + node.triangles_address, sizeof(bvh_triangle) * total_right);
        int total_left = 0;

        for (int k = -space_extension * samples_in_space;
             k <= (1 + space_extension) * samples_in_space; k++) {
          const float split = search_start + k * search_interval;
          vec3_p high_left, high_right, low_left, low_right;

          if (a == 0) {
            divide_along_x_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right, i);
          }
          else if (a == 1) {
            divide_along_y_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right, i);
          }
          else {
            divide_along_z_axis(
              split, &left, &right, new_triangles + triangles_ptr + total_left, container,
              container, total_right, i);
          }

          total_left += left;
          total_right -= left;

          fit_bounds(new_triangles + triangles_ptr, total_left, &high_left, &low_left, &average);
          fit_bounds(container, total_right, &high_right, &low_right, &average);

          vec3 diff_left = {
            .x = high_left.x - low_left.x,
            .y = high_left.y - low_left.y,
            .z = high_left.z - low_left.z};

          vec3 diff_right = {
            .x = high_right.x - low_right.x,
            .y = high_right.y - low_right.y,
            .z = high_right.z - low_right.z};

          const float cost_L = diff_left.x * diff_left.y * diff_left.z;
          const float cost_R = diff_right.x * diff_right.y * diff_right.z;

          const float total_cost = cost_L * total_left + cost_R * total_right;

          if (total_cost < optimal_cost) {
            optimal_cost  = total_cost;
            optimal_split = split;
            axis          = a;
          }
        }
      }

      if (axis == 0) {
        divide_along_x_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count, i);
      }
      else if (axis == 1) {
        divide_along_y_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count, i);
      }
      else {
        divide_along_z_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, bvh_triangles,
          bvh_triangles + node.triangles_address, node.triangle_count, i);
      }

      compressed_vec3 compressed_low, compressed_high;

      fit_bounds(new_triangles + triangles_ptr, left, &high, &low, &average);

      compressed_low.x = (uint8_t) floor((low.x - node.p.x) * compression_x);
      compressed_low.y = (uint8_t) floor((low.y - node.p.y) * compression_y);
      compressed_low.z = (uint8_t) floor((low.z - node.p.z) * compression_z);

      compressed_high.x = (uint8_t) ceil((high.x - node.p.x) * compression_x);
      compressed_high.y = (uint8_t) ceil((high.y - node.p.y) * compression_y);
      compressed_high.z = (uint8_t) ceil((high.z - node.p.z) * compression_z);

      node.left_high = compressed_high;
      node.left_low  = compressed_low;

      nodes[2 * node_ptr + 1].triangle_count    = left;
      nodes[2 * node_ptr + 1].triangles_address = triangles_ptr;

      triangles_ptr += left;

      memcpy(new_triangles + triangles_ptr, bvh_triangles, right * sizeof(bvh_triangle));

      fit_bounds(new_triangles + triangles_ptr, right, &high, &low, &average);

      compressed_low.x = (uint8_t) floor((low.x - node.p.x) * compression_x);
      compressed_low.y = (uint8_t) floor((low.y - node.p.y) * compression_y);
      compressed_low.z = (uint8_t) floor((low.z - node.p.z) * compression_z);

      compressed_high.x = (uint8_t) ceil((high.x - node.p.x) * compression_x);
      compressed_high.y = (uint8_t) ceil((high.y - node.p.y) * compression_y);
      compressed_high.z = (uint8_t) ceil((high.z - node.p.z) * compression_z);

      node.right_high = compressed_high;
      node.right_low  = compressed_low;

      nodes[2 * node_ptr + 2].triangle_count    = right;
      nodes[2 * node_ptr + 2].triangles_address = triangles_ptr;

      triangles_ptr += right;

      if (node.triangle_count > threshold_triangles) {
        node.triangle_count    = 0;
        node.triangles_address = -1;
      }

      nodes[node_ptr] = node;

      node_ptr++;
    }

    _mm_free(bvh_triangles);
    _mm_free(container);

    bvh_triangles = new_triangles;

    nodes_at_depth *= 2;

    printf("\r                             \rBVH depth %d/%d built.", i + 1, max_depth);
  }

  printf("\r                             \r");

  Triangle* triangles_swap = malloc(sizeof(Triangle) * triangles_length);
  memcpy(triangles_swap, triangles, sizeof(Triangle) * triangles_length);

  for (unsigned int i = 0; i < triangles_length; i++) {
    triangles[i] = triangles_swap[bvh_triangles[i].id];
  }

  *triangles_io = triangles;

  *nodes_length_out = node_count;

  _mm_free(bvh_triangles);
  free(triangles_swap);

  return nodes;
}
