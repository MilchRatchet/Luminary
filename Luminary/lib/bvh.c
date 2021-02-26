#include "bvh.h"
#include "mesh.h"
#include "primitives.h"
#include "float.h"
#include "string.h"
#include <stdlib.h>

static void fit_bounds(
  const Triangle* triangles, const int triangles_length, vec3* high_out, vec3* low_out) {
  vec3 high = {.x = -FLT_MAX, .y = -FLT_MAX, .z = -FLT_MAX};
  vec3 low  = {.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX};

  for (unsigned int i = 0; i < triangles_length; i++) {
    Triangle triangle = triangles[i];

    vec3 vertex = triangle.vertex;

    high.x = max(high.x, vertex.x);
    high.y = max(high.y, vertex.y);
    high.z = max(high.z, vertex.z);

    low.x = min(low.x, vertex.x);
    low.y = min(low.y, vertex.y);
    low.z = min(low.z, vertex.z);

    vertex.x += triangle.edge1.x;
    vertex.y += triangle.edge1.y;
    vertex.z += triangle.edge1.z;

    high.x = max(high.x, vertex.x);
    high.y = max(high.y, vertex.y);
    high.z = max(high.z, vertex.z);

    low.x = min(low.x, vertex.x);
    low.y = min(low.y, vertex.y);
    low.z = min(low.z, vertex.z);

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
  }

  *high_out = high;
  *low_out  = low;
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

    is_right += triangle.vertex.x > split;
    is_left += triangle.vertex.x < split;
    is_right += triangle.vertex.x + triangle.edge1.x > split;
    is_left += triangle.vertex.x + triangle.edge1.x < split;
    is_right += triangle.vertex.x + triangle.edge2.x > split;
    is_left += triangle.vertex.x + triangle.edge2.x < split;

    if (is_left)
      triangles_left[left++] = triangle;

    if (is_right)
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

    is_right += triangle.vertex.y > split;
    is_left += triangle.vertex.y < split;
    is_right += triangle.vertex.y + triangle.edge1.y > split;
    is_left += triangle.vertex.y + triangle.edge1.y < split;
    is_right += triangle.vertex.y + triangle.edge2.y > split;
    is_left += triangle.vertex.y + triangle.edge2.y < split;

    if (is_left)
      triangles_left[left++] = triangle;

    if (is_right)
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

    is_right += triangle.vertex.z > split;
    is_left += triangle.vertex.z < split;
    is_right += triangle.vertex.z + triangle.edge1.z > split;
    is_left += triangle.vertex.z + triangle.edge1.z < split;
    is_right += triangle.vertex.z + triangle.edge2.z > split;
    is_left += triangle.vertex.z + triangle.edge2.z < split;

    if (is_left)
      triangles_left[left++] = triangle;

    if (is_right)
      triangles_right[right++] = triangle;
  }

  *left_out  = left;
  *right_out = right;
}

/*
 * Idea, don't convert leaf node, if triangle_count < threshold
 */
Node* build_bvh_structure(
  Triangle** triangles_io, unsigned int* triangles_length, const int max_depth) {
  Triangle* triangles = *triangles_io;
  int pow             = 2;
  int node_count      = 1;

  for (int i = 0; i < max_depth; i++) {
    node_count += pow;
    pow *= 2;
  }

  Node* nodes = (Node*) malloc(sizeof(Node) * node_count);

  nodes[0].triangles_address   = 0;
  nodes[0].triangle_count      = *triangles_length;
  nodes[0].uncle_address       = 0;
  nodes[0].grand_uncle_address = 0;

  unsigned int new_triangles_length = *triangles_length;

  unsigned int nodes_at_depth = 1;

  int node_ptr = 0;

  for (int i = 0; i < max_depth; i++) {
    unsigned int triangles_ptr = 0;
    Triangle* new_triangles    = (Triangle*) malloc(sizeof(Triangle) * 2 * new_triangles_length);

    for (int j = 0; j < nodes_at_depth; j++) {
      vec3 high, low;
      int left, right;
      Node node = nodes[node_ptr];
      fit_bounds(triangles + node.triangles_address, node.triangle_count, &high, &low);

      if (high.x - low.x > high.y - low.y && high.x - low.x > high.z - low.z) {
        divide_along_x_axis(
          (high.x + low.x) / 2.0f, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }
      else if (high.y - low.y > high.z - low.z) {
        divide_along_y_axis(
          (high.y + low.y) / 2.0f, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }
      else {
        divide_along_z_axis(
          (high.z + low.z) / 2.0f, &left, &right, new_triangles + triangles_ptr, triangles,
          triangles + node.triangles_address, node.triangle_count);
      }

      fit_bounds(new_triangles + triangles_ptr, left, &high, &low);

      node.left_address = 2 * node_ptr + 1;
      node.left_high    = high;
      node.left_low     = low;

      nodes[node.left_address].uncle_address       = node_ptr;
      nodes[node.left_address].grand_uncle_address = node.uncle_address;
      nodes[node.left_address].triangle_count      = left;
      nodes[node.left_address].triangles_address   = triangles_ptr;

      triangles_ptr += left;

      memcpy(new_triangles + triangles_ptr, triangles, right * sizeof(Triangle));

      fit_bounds(new_triangles + triangles_ptr, right, &high, &low);

      node.right_address = 2 * node_ptr + 2;
      node.right_high    = high;
      node.right_low     = low;

      nodes[node.right_address].uncle_address       = node_ptr;
      nodes[node.right_address].grand_uncle_address = node.uncle_address;
      nodes[node.right_address].triangle_count      = right;
      nodes[node.right_address].triangles_address   = triangles_ptr;

      triangles_ptr += right;

      node.triangle_count = 0;

      nodes[node_ptr] = node;

      node_ptr++;
    }

    free(triangles);

    triangles = new_triangles;

    new_triangles_length = triangles_ptr;

    nodes_at_depth *= 2;
  }

  *triangles_io = (Triangle*) realloc(triangles, new_triangles_length * sizeof(Triangle));

  *triangles_length = new_triangles_length;

  return nodes;
}
