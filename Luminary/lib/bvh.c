#include "bvh.h"
#include "mesh.h"
#include "primitives.h"
#include "float.h"
#include "string.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static const int samples_in_space = 50;
static const int space_extension  = 2;

static void fit_bounds(
  const Triangle* triangles, const int triangles_length, vec3* high_out, vec3* low_out,
  vec3* average_out) {
  vec3 high    = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
  vec3 low     = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};
  vec3 average = {.x = 0.0f, .y = 0.0f, .z = 0.0f};

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle triangle = triangles[i];

    vec3 vertex = triangle.vertex;

    high.x = max(high.x, vertex.x);
    high.y = max(high.y, vertex.y);
    high.z = max(high.z, vertex.z);

    low.x = min(low.x, vertex.x);
    low.y = min(low.y, vertex.y);
    low.z = min(low.z, vertex.z);

    vec3 centroid = vertex;

    vertex.x += triangle.edge1.x;
    vertex.y += triangle.edge1.y;
    vertex.z += triangle.edge1.z;

    high.x = max(high.x, vertex.x);
    high.y = max(high.y, vertex.y);
    high.z = max(high.z, vertex.z);

    low.x = min(low.x, vertex.x);
    low.y = min(low.y, vertex.y);
    low.z = min(low.z, vertex.z);

    centroid.x += vertex.x;
    centroid.y += vertex.y;
    centroid.z += vertex.z;

    vertex = triangle.vertex;

    vertex.x += triangle.edge2.x;
    vertex.y += triangle.edge2.y;
    vertex.z += triangle.edge2.z;

    high.x = max(high.x, vertex.x);
    high.y = max(high.y, vertex.y);
    high.z = max(high.z, vertex.z);

    low.x = min(low.x, vertex.x);
    low.y = min(low.y, vertex.y);
    low.z = min(low.z, vertex.z);

    centroid.x += vertex.x;
    centroid.y += vertex.y;
    centroid.z += vertex.z;

    centroid.x /= 3.0f;
    centroid.y /= 3.0f;
    centroid.z /= 3.0f;

    average.x += centroid.x;
    average.y += centroid.y;
    average.z += centroid.z;
  }

  average.x /= (triangles_length);
  average.y /= (triangles_length);
  average.z /= (triangles_length);

  *high_out    = high;
  *low_out     = low;
  *average_out = average;
}

static void divide_along_x_axis(
  const float split, int* left_out, int* right_out, Triangle* triangles_left,
  Triangle* triangles_right, Triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    Triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.x >= split;
    is_left += triangle.vertex.x <= split;
    is_right += triangle.vertex.x + triangle.edge1.x >= split;
    is_left += triangle.vertex.x + triangle.edge1.x <= split;
    is_right += triangle.vertex.x + triangle.edge2.x >= split;
    is_left += triangle.vertex.x + triangle.edge2.x <= split;

    if (is_left)
      triangles_left[left++] = triangle;
    else if (is_right)
      triangles_right[right++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_y_axis(
  const float split, int* left_out, int* right_out, Triangle* triangles_left,
  Triangle* triangles_right, Triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    Triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.y >= split;
    is_left += triangle.vertex.y <= split;
    is_right += triangle.vertex.y + triangle.edge1.y >= split;
    is_left += triangle.vertex.y + triangle.edge1.y <= split;
    is_right += triangle.vertex.y + triangle.edge2.y >= split;
    is_left += triangle.vertex.y + triangle.edge2.y <= split;

    if (is_left)
      triangles_left[left++] = triangle;
    else if (is_right)
      triangles_right[right++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

static void divide_along_z_axis(
  const float split, int* left_out, int* right_out, Triangle* triangles_left,
  Triangle* triangles_right, Triangle* triangles, const unsigned int triangles_length) {
  int left  = 0;
  int right = 0;

  for (int i = 0; i < triangles_length; i++) {
    Triangle triangle = triangles[i];

    int is_right = 0;
    int is_left  = 0;

    is_right += triangle.vertex.z >= split;
    is_left += triangle.vertex.z <= split;
    is_right += triangle.vertex.z + triangle.edge1.z >= split;
    is_left += triangle.vertex.z + triangle.edge1.z <= split;
    is_right += triangle.vertex.z + triangle.edge2.z >= split;
    is_left += triangle.vertex.z + triangle.edge2.z <= split;

    if (is_left)
      triangles_left[left++] = triangle;
    else if (is_right)
      triangles_right[right++] = triangle;
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

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = triangles_length;

  unsigned int nodes_at_depth = 1;

  int node_ptr = 0;

  for (int i = 0; i < max_depth; i++) {
    unsigned int triangles_ptr = 0;
    Triangle* new_triangles    = (Triangle*) malloc(sizeof(Triangle) * triangles_length);
    Triangle* container        = (Triangle*) malloc(sizeof(Triangle) * triangles_length);

    for (int j = 0; j < nodes_at_depth; j++) {
      vec3 high, low, node_average, average;
      int left, right;
      Node node = nodes[node_ptr];

      fit_bounds(
        triangles + node.triangles_address, node.triangle_count, &high, &low, &node_average);
      high =
        (node_ptr & 1) ? nodes[(node_ptr - 1) / 2].left_high : nodes[(node_ptr - 1) / 2].right_high;
      low =
        (node_ptr & 1) ? nodes[(node_ptr - 1) / 2].left_low : nodes[(node_ptr - 1) / 2].right_low;

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

        const float search_interval = (split_object - split_spatial) / samples_in_space;

        for (int k = -space_extension * samples_in_space;
             k <= (1 + space_extension) * samples_in_space; k++) {
          const float split = split_spatial + k * search_interval;
          vec3 high_left, high_right, low_left, low_right;

          if (a == 0) {
            divide_along_x_axis(
              split, &left, &right, new_triangles + triangles_ptr, container,
              triangles + node.triangles_address, node.triangle_count);
          }
          else if (a == 1) {
            divide_along_y_axis(
              split, &left, &right, new_triangles + triangles_ptr, container,
              triangles + node.triangles_address, node.triangle_count);
          }
          else {
            divide_along_z_axis(
              split, &left, &right, new_triangles + triangles_ptr, container,
              triangles + node.triangles_address, node.triangle_count);
          }

          fit_bounds(new_triangles + triangles_ptr, left, &high_left, &low_left, &average);
          fit_bounds(container, right, &high_right, &low_right, &average);

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

          const float total_cost = cost_L * left + cost_R * right;

          if (total_cost < optimal_cost) {
            optimal_cost  = total_cost;
            optimal_split = split;
            axis          = a;
          }
        }
      }

      if (axis == 0) {
        divide_along_x_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }
      else if (axis == 1) {
        divide_along_y_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }
      else {
        divide_along_z_axis(
          optimal_split, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }

      fit_bounds(new_triangles + triangles_ptr, left, &high, &low, &average);

      node.left_high = high;
      node.left_low  = low;

      nodes[2 * node_ptr + 1].triangle_count    = left;
      nodes[2 * node_ptr + 1].triangles_address = triangles_ptr;

      triangles_ptr += left;

      memcpy(new_triangles + triangles_ptr, triangles, right * sizeof(Triangle));

      fit_bounds(new_triangles + triangles_ptr, right, &high, &low, &average);

      node.right_high = high;
      node.right_low  = low;

      nodes[2 * node_ptr + 2].triangle_count    = right;
      nodes[2 * node_ptr + 2].triangles_address = triangles_ptr;

      triangles_ptr += right;

      node.triangle_count    = 0;
      node.triangles_address = -1;

      nodes[node_ptr] = node;

      node_ptr++;
    }

    free(triangles);
    free(container);

    triangles = new_triangles;

    nodes_at_depth *= 2;

    printf("\r                             \rBVH depth %d/%d built.", i + 1, max_depth);
  }

  puts("");

  *triangles_io = triangles;

  *nodes_length_out = node_count;

  return nodes;
}
