#include "light.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "buffer.h"
#include "ceb.h"
#include "device.h"
#include "structs.h"
#include "texture.h"
#include "utils.h"

// #define LIGHT_TREE_DEBUG_OUTPUT

enum LightTreeSweepAxis {
  LIGHT_TREE_SWEEP_AXIS_X = 0,
  LIGHT_TREE_SWEEP_AXIS_Y = 1,
  LIGHT_TREE_SWEEP_AXIS_Z = 2
} typedef LightTreeSweepAxis;

enum LightTreeNodeType {
  LIGHT_TREE_NODE_TYPE_NULL     = 0,
  LIGHT_TREE_NODE_TYPE_INTERNAL = 1,
  LIGHT_TREE_NODE_TYPE_LEAF     = 2
} typedef LightTreeNodeType;

struct LightTreeBinaryNode {
  vec3 left_low;
  vec3 left_high;
  vec3 right_low;
  vec3 right_high;
  vec3 self_low;
  vec3 self_high;
  uint32_t triangle_count;
  uint32_t triangles_address;
  uint32_t child_address;
  float surface_area;
  LightTreeNodeType type;
  float left_energy;
  float right_energy;
  uint32_t path;
  uint32_t depth;
} typedef LightTreeBinaryNode;

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
  float power;
  uint64_t _p;
} typedef Fragment;

struct LightTreeChildNode {
  vec3 point;
  float energy;
  float confidence;
  uint32_t light_count;
} typedef LightTreeChildNode;

struct LightTreeWork {
  TriangleLight* triangles;
  Fragment* fragments;
  uint2* paths;
  uint32_t triangles_count;
  LightTreeBinaryNode* binary_nodes;
  LightTreeNode* nodes;
  LightTreeNode8Packed* nodes8_packed;
  uint32_t nodes_count;
  uint32_t nodes_8_count;
} typedef LightTreeWork;

struct Bin {
  vec3_p high;
  vec3_p low;
  int32_t entry;
  int32_t exit;
  float energy;
  uint32_t _p;
} typedef Bin;

// Note: This is determined by the number of bits that we allocate for each node.
#define THRESHOLD_TRIANGLES 3
#define OBJECT_SPLIT_BIN_COUNT 64

// We need to bound the dimensions, the number must be large but still much smaller than FLT_MAX
#define MAX_VALUE 1e10f

#define FRAGMENT_ERROR_COMP (FLT_EPSILON * 4.0f)

static void _lights_tree_create_fragments(Scene* scene, LightTreeWork* work) {
  const uint32_t triangle_count = scene->triangle_lights_count;

  work->triangles       = scene->triangle_lights;
  work->triangles_count = triangle_count;

  work->fragments = malloc(sizeof(Fragment) * triangle_count);

  for (uint32_t i = 0; i < triangle_count; i++) {
    TriangleLight t = scene->triangle_lights[i];

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

    vec3 cross = {
      .x = t.edge1.y * t.edge2.z - t.edge1.z * t.edge2.y,
      .y = t.edge1.z * t.edge2.x - t.edge1.x * t.edge2.z,
      .z = t.edge1.x * t.edge2.y - t.edge1.y * t.edge2.x};

    const float area = 0.5f * sqrtf(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);

    Fragment frag      = {.high = high, .low = low, .middle = middle, .id = i, .power = t.power * area};
    work->fragments[i] = frag;
  }
}

static float get_entry_by_axis(const vec3_p p, const LightTreeSweepAxis axis) {
  switch (axis) {
    case LIGHT_TREE_SWEEP_AXIS_X:
      return p.x;
    case LIGHT_TREE_SWEEP_AXIS_Y:
      return p.y;
    case LIGHT_TREE_SWEEP_AXIS_Z:
    default:
      return p.z;
  }
}

#define _get_entry_by_axis(ptr, axis) \
  _mm_cvtss_f32(_mm_shuffle_ps(_mm_loadu_ps((float*) (ptr)), _mm_loadu_ps((float*) (ptr)), _MM_SHUFFLE(0, 0, 0, axis)))

static void fit_bounds(const Fragment* fragments, const uint32_t fragments_count, vec3_p* high_out, vec3_p* low_out) {
  __m128 high = _mm_set1_ps(-MAX_VALUE);
  __m128 low  = _mm_set1_ps(MAX_VALUE);

  for (uint32_t i = 0; i < fragments_count; i++) {
    __m128 high_frag = _mm_loadu_ps((float*) &(fragments[i].high));
    __m128 low_frag  = _mm_loadu_ps((float*) &(fragments[i].low));

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
    const __m128 high_bin = _mm_loadu_ps((float*) &(bins[i].high));
    const __m128 low_bin  = _mm_loadu_ps((float*) &(bins[i].low));

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

#define _construct_bins_kernel(_axis_)                                         \
  {                                                                            \
    for (uint32_t i = 0; i < fragments_count; i++) {                           \
      const double value = _get_entry_by_axis(&(fragments[i].middle), _axis_); \
      int pos            = ((int) ceil((value - low_axis) / interval)) - 1;    \
      if (pos < 0)                                                             \
        pos = 0;                                                               \
                                                                               \
      bins[pos].entry++;                                                       \
      bins[pos].exit++;                                                        \
      bins[pos].energy += fragments[i].power;                                  \
                                                                               \
      __m128 high_bin  = _mm_loadu_ps((float*) &(bins[pos].high));             \
      __m128 low_bin   = _mm_loadu_ps((float*) &(bins[pos].low));              \
      __m128 high_frag = _mm_loadu_ps((float*) &(fragments[i].high));          \
      __m128 low_frag  = _mm_loadu_ps((float*) &(fragments[i].low));           \
                                                                               \
      high_bin = _mm_max_ps(high_bin, high_frag);                              \
      low_bin  = _mm_min_ps(low_bin, low_frag);                                \
                                                                               \
      _mm_storeu_ps((float*) &(bins[pos].high), high_bin);                     \
      _mm_storeu_ps((float*) &(bins[pos].low), low_bin);                       \
    }                                                                          \
  }

static double construct_bins(
  Bin* restrict bins, const Fragment* restrict fragments, const uint32_t fragments_count, const LightTreeSweepAxis axis, double* offset) {
  vec3_p high, low;
  fit_bounds(fragments, fragments_count, &high, &low);

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
    .exit   = 0,
    .energy = 0.0f};
  bins[0] = b;

  for (uint32_t i = 1; i < OBJECT_SPLIT_BIN_COUNT; i = i << 1) {
    memcpy(bins + i, bins, i * sizeof(Bin));
  }

  switch (axis) {
    case LIGHT_TREE_SWEEP_AXIS_X:
      _construct_bins_kernel(0);
      break;
    case LIGHT_TREE_SWEEP_AXIS_Y:
      _construct_bins_kernel(1);
      break;
    case LIGHT_TREE_SWEEP_AXIS_Z:
    default:
      _construct_bins_kernel(2);
      break;
  }

  return interval;
}

static void divide_middles_along_axis(
  const double split, const LightTreeSweepAxis axis, Fragment* fragments, const uint32_t fragments_count) {
  uint32_t left  = 0;
  uint32_t right = 0;

  while (left + right < fragments_count) {
    const Fragment frag = fragments[left];

    const double middle = get_entry_by_axis(frag.middle, axis);

    if (middle > split) {
      const uint32_t swap_index = fragments_count - 1 - right;

      Fragment temp         = fragments[swap_index];
      fragments[swap_index] = frag;
      fragments[left]       = temp;

      right++;
    }
    else {
      left++;
    }
  }
}

static void _lights_tree_build_binary_bvh(LightTreeWork* work) {
  Fragment* fragments      = work->fragments;
  uint32_t fragments_count = work->triangles_count;

  uint32_t nodes_length      = 1 + fragments_count;
  LightTreeBinaryNode* nodes = malloc(sizeof(LightTreeBinaryNode) * nodes_length);
  memset(nodes, 0, sizeof(LightTreeBinaryNode) * nodes_length);

  nodes[0].triangles_address = 0;
  nodes[0].triangle_count    = fragments_count;
  nodes[0].type              = LIGHT_TREE_NODE_TYPE_LEAF;
  nodes[0].path              = 0;
  nodes[0].depth             = 0;

  Bin* bins = (Bin*) malloc(sizeof(Bin) * OBJECT_SPLIT_BIN_COUNT);

  uint32_t begin_of_current_nodes = 0;
  uint32_t end_of_current_nodes   = 1;
  uint32_t write_ptr              = 1;

  while (begin_of_current_nodes != end_of_current_nodes) {
    for (uint32_t node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes; node_ptr++) {
      LightTreeBinaryNode node = nodes[node_ptr];

      const uint32_t fragments_ptr   = node.triangles_address;
      const uint32_t fragments_count = node.triangle_count;

      // Node has few enough triangles, finalize it as a leaf node.
      if (fragments_count <= 1) {
        continue;
      }

      // Compute surface area of current node.
      double parent_surface_area;
      float max_axis_interval = 0.0f;
      {
        vec3_p high_parent, low_parent;
        fit_bounds(fragments + fragments_ptr, fragments_count, &high_parent, &low_parent);
        const vec3 diff = {.x = high_parent.x - low_parent.x, .y = high_parent.y - low_parent.y, .z = high_parent.z - low_parent.z};

        max_axis_interval = fmaxf(diff.x, fmaxf(diff.y, diff.z));

        parent_surface_area = (double) diff.x * (double) diff.y + (double) diff.x * (double) diff.z + (double) diff.y * (double) diff.z;
      }

      vec3_p high, low;
      double optimal_cost = DBL_MAX;
      LightTreeSweepAxis axis;
      double optimal_splitting_plane;
      int found_split = 0;
      uint32_t optimal_split;
      float optimal_left_energy, optimal_right_energy;

      // For each axis, perform a greedy search for an optimal split.
      for (int a = 0; a < 3; a++) {
        double low_split;
        const double interval = construct_bins(bins, fragments + fragments_ptr, fragments_count, (LightTreeSweepAxis) a, &low_split);

        if (interval == 0.0)
          continue;

        const double interval_cost = max_axis_interval / interval;

        uint32_t left = 0;

        float left_energy  = 0.0f;
        float right_energy = 0.0f;

        for (int k = 0; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          right_energy += bins[k].energy;
        }

        vec3_p high_left  = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE, ._p = -MAX_VALUE};
        vec3_p high_right = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE, ._p = -MAX_VALUE};
        vec3_p low_left   = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE, ._p = MAX_VALUE};
        vec3_p low_right  = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE, ._p = MAX_VALUE};

        for (int k = 1; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          update_bounds_of_bins(bins + k - 1, &high_left, &low_left);
          fit_bounds_of_bins(bins + k, OBJECT_SPLIT_BIN_COUNT - k, &high_right, &low_right);

          left_energy += bins[k - 1].energy;
          right_energy -= bins[k - 1].energy;

          vec3_p diff_left = {
            .x = high_left.x - low_left.x, .y = high_left.y - low_left.y, .z = high_left.z - low_left.z, ._p = high_left._p - low_left._p};

          vec3_p diff_right = {
            .x  = high_right.x - low_right.x,
            .y  = high_right.y - low_right.y,
            .z  = high_right.z - low_right.z,
            ._p = high_right._p - low_right._p};

          const double left_area  = diff_left.x * diff_left.y + diff_left.x * diff_left.z + diff_left.y * diff_left.z;
          const double right_area = diff_right.x * diff_right.y + diff_right.x * diff_right.z + diff_right.y * diff_right.z;

          left += bins[k - 1].entry;

          if (left == 0 || left == fragments_count)
            continue;

          const double total_cost = interval_cost * (left_energy * left_area + right_energy * right_area) / parent_surface_area;

          if (total_cost < optimal_cost) {
            optimal_cost            = total_cost;
            optimal_split           = left;
            optimal_splitting_plane = low_split + k * interval;
            found_split             = 1;
            axis                    = a;
            optimal_left_energy     = left_energy;
            optimal_right_energy    = right_energy;
          }
        }
      }

      if (found_split) {
        divide_middles_along_axis(optimal_splitting_plane, axis, fragments + fragments_ptr, fragments_count);
      }
      else if (fragments_count > THRESHOLD_TRIANGLES) {
        // We didn't find a split but we have too many triangles so we need to do a simply list split.
        optimal_split = fragments_count / 2;

        optimal_left_energy  = 0.0f;
        optimal_right_energy = 0.0f;

        uint32_t frag_id = 0;

        for (; frag_id < optimal_split; frag_id++) {
          optimal_left_energy += fragments[fragments_ptr + frag_id].power;
        }

        for (; frag_id < fragments_count; frag_id++) {
          optimal_right_energy += fragments[fragments_ptr + frag_id].power;
        }

        warn_message("List Split with %u triangles. This warning is here until I have verified that this works!", fragments_count);
      }
      else {
        // We didn't find a split but there are few enough triangles so we will leave this as a leaf node.
        continue;
      }

      // At this point we are committed to split, hence we need to make sure that we have enough memory allocated for that.
      if (write_ptr + 2 >= nodes_length) {
        nodes_length *= 2;
        nodes = safe_realloc(nodes, sizeof(LightTreeBinaryNode) * nodes_length);
      }

      node.left_energy  = optimal_left_energy;
      node.right_energy = optimal_right_energy;

      fit_bounds(fragments + fragments_ptr, optimal_split, &high, &low);

      node.left_high.x   = high.x;
      node.left_high.y   = high.y;
      node.left_high.z   = high.z;
      node.left_low.x    = low.x;
      node.left_low.y    = low.y;
      node.left_low.z    = low.z;
      node.child_address = write_ptr;

      LightTreeBinaryNode node_left = {
        .triangle_count    = optimal_split,
        .triangles_address = fragments_ptr,
        .type              = LIGHT_TREE_NODE_TYPE_LEAF,
        .surface_area = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z),
        .self_high.x  = high.x,
        .self_high.y  = high.y,
        .self_high.z  = high.z,
        .self_low.x   = low.x,
        .self_low.y   = low.y,
        .self_low.z   = low.z,
        .path         = node.path | (1 << node.depth),
        .depth        = node.depth + 1,
      };

      nodes[write_ptr] = node_left;

      write_ptr++;

      fit_bounds(fragments + fragments_ptr + optimal_split, node.triangle_count - optimal_split, &high, &low);

      node.right_high.x = high.x;
      node.right_high.y = high.y;
      node.right_high.z = high.z;
      node.right_low.x  = low.x;
      node.right_low.y  = low.y;
      node.right_low.z  = low.z;

      LightTreeBinaryNode node_right = {
        .triangle_count    = node.triangle_count - optimal_split,
        .triangles_address = fragments_ptr + optimal_split,
        .type              = LIGHT_TREE_NODE_TYPE_LEAF,
        .surface_area = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z),
        .self_high.x  = high.x,
        .self_high.y  = high.y,
        .self_high.z  = high.z,
        .self_low.x   = low.x,
        .self_low.y   = low.y,
        .self_low.z   = low.z,
        .path         = node.path,
        .depth        = node.depth + 1,
      };

      nodes[write_ptr] = node_right;

      write_ptr++;

      node.type = LIGHT_TREE_NODE_TYPE_INTERNAL;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;
  }

  nodes_length = write_ptr;

  nodes = safe_realloc(nodes, sizeof(LightTreeBinaryNode) * nodes_length);

  free(bins);

  work->binary_nodes = nodes;
  work->nodes_count  = nodes_length;
}

static void _lights_get_ref_point_and_dist(LightTreeWork* work, LightTreeBinaryNode node, float energy, vec3* ref_point, float* ref_dist) {
  const float inverse_total_energy = 1.0f / energy;

  vec3 p = {.x = 0.0f, .y = 0.0f, .z = 0.0f};
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    Fragment frag = work->fragments[node.triangles_address + i];

    const float weight = frag.power * inverse_total_energy;

    p.x += weight * frag.middle.x;
    p.y += weight * frag.middle.y;
    p.z += weight * frag.middle.z;
  }

  float weighted_dist = 0.0f;
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    Fragment frag = work->fragments[node.triangles_address + i];

    const vec3 diff = {.x = frag.middle.x - p.x, .y = frag.middle.y - p.y, .z = frag.middle.z - p.z};

    const float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    const float weight = frag.power * inverse_total_energy;

    weighted_dist += weight * dist;
  }

  weighted_dist = fmaxf(weighted_dist, 0.0001f);

  *ref_point = p;
  *ref_dist  = weighted_dist;
}

// Set the reference point in each node to be the energy weighted mean of the centers of the lights.
// Then, based on that reference point, compute the smallest distance to any light center.
// This is our spatial confidence that we use to clamp the distance with when evaluating the importance during traversal.
static void _lights_tree_build_traversal_structure(LightTreeWork* work) {
  LightTreeNode* nodes = malloc(sizeof(LightTreeNode) * work->nodes_count);

  for (uint32_t i = 0; i < work->nodes_count; i++) {
    LightTreeBinaryNode binary_node = work->binary_nodes[i];

    LightTreeNode node;

    node.left_energy  = binary_node.left_energy;
    node.right_energy = binary_node.right_energy;

    switch (binary_node.type) {
      case LIGHT_TREE_NODE_TYPE_INTERNAL:
        node.light_count = 0;
        node.ptr         = binary_node.child_address;

        _lights_get_ref_point_and_dist(
          work, work->binary_nodes[binary_node.child_address + 0], node.left_energy, &node.left_ref_point, &node.left_confidence);
        _lights_get_ref_point_and_dist(
          work, work->binary_nodes[binary_node.child_address + 1], node.right_energy, &node.right_ref_point, &node.right_confidence);
        break;
      case LIGHT_TREE_NODE_TYPE_LEAF:
        node.light_count = binary_node.triangle_count;
        node.ptr         = binary_node.triangles_address;
        break;
      default:
        crash_message("Encountered illegal node type!");
        break;
    }

    nodes[i] = node;
  }

  work->nodes = nodes;
}

static void _lights_tree_collapse(LightTreeWork* work) {
  const uint32_t fragments_count = work->triangles_count;

  LightTreeNode* binary_nodes       = work->nodes;
  const uint32_t binary_nodes_count = work->nodes_count;

  uint32_t node_count = binary_nodes_count;

  LightTreeNode8Packed* nodes = (LightTreeNode8Packed*) malloc(sizeof(LightTreeNode8Packed) * node_count);
  memset(nodes, 0, sizeof(LightTreeNode8Packed) * node_count);

  uint64_t* node_paths  = (uint64_t*) malloc(sizeof(uint64_t) * node_count);
  uint32_t* node_depths = (uint32_t*) malloc(sizeof(uint32_t) * node_count);

  uint32_t* new_fragments  = (uint32_t*) malloc(sizeof(uint32_t) * fragments_count);
  uint64_t* fragment_paths = (uint64_t*) malloc(sizeof(uint64_t) * fragments_count);

  memset(new_fragments, 0xFF, sizeof(uint32_t) * fragments_count);
  memset(fragment_paths, 0xFF, sizeof(uint64_t) * fragments_count);

  nodes[0].child_ptr = 0;
  nodes[0].light_ptr = 0;

  node_paths[0]  = 0;
  node_depths[0] = 0;

  uint32_t begin_of_current_nodes = 0;
  uint32_t end_of_current_nodes   = 1;
  uint32_t write_ptr              = 1;
  uint32_t triangles_ptr          = 0;

  while (begin_of_current_nodes != end_of_current_nodes) {
    for (uint32_t node_ptr = begin_of_current_nodes; node_ptr < end_of_current_nodes; node_ptr++) {
      LightTreeNode8Packed node = nodes[node_ptr];

      const uint32_t binary_index = node.child_ptr;

      if (binary_index == 0xFFFFFFFF)
        continue;

      const uint64_t current_node_path  = node_paths[node_ptr];
      const uint32_t current_node_depth = node_depths[node_ptr];

      node.child_ptr = write_ptr;
      node.light_ptr = triangles_ptr;

      LightTreeChildNode children[8];
      uint32_t child_binary_index[8];
      uint32_t child_leaf_light_ptr[8];
      uint32_t child_leaf_light_count[8];

      for (int i = 0; i < 8; i++) {
        child_binary_index[i]     = 0xFFFFFFFF;
        child_leaf_light_ptr[i]   = 0xFFFFFFFF;
        child_leaf_light_count[i] = 0;
      }

      uint32_t child_count      = 0;
      uint32_t child_leaf_count = 0;

      if (binary_nodes[binary_index].light_count == 0) {
        const LightTreeNode binary_node = binary_nodes[binary_index];

        LightTreeChildNode left_child;
        memset(&left_child, 0, sizeof(LightTreeChildNode));

        left_child.point      = binary_node.left_ref_point;
        left_child.energy     = binary_node.left_energy;
        left_child.confidence = binary_node.left_confidence;

        child_binary_index[child_count] = binary_node.ptr;

        children[child_count++] = left_child;

        LightTreeChildNode right_child;
        memset(&right_child, 0, sizeof(LightTreeChildNode));

        right_child.point      = binary_node.right_ref_point;
        right_child.energy     = binary_node.right_energy;
        right_child.confidence = binary_node.right_confidence;

        child_binary_index[child_count] = binary_node.ptr + 1;

        children[child_count++] = right_child;
      }
      else {
        // This case implies that the whole tree was just a leaf.
        // Hence we fill in some basic information and that is it.
        LightTreeChildNode child;
        memset(&child, 0, sizeof(LightTreeChildNode));

        child.energy = 1.0f;

        child_leaf_light_ptr[child_count]   = binary_nodes[binary_index].ptr;
        child_leaf_light_count[child_count] = binary_nodes[binary_index].light_count;

        children[child_count++] = child;

        child_leaf_count++;
      }

      while (child_count < 8 && child_count > child_leaf_count) {
        uint32_t loop_end = child_count;

        for (uint64_t child_ptr = 0; child_ptr < loop_end; child_ptr++) {
          const uint32_t binary_index_of_child = child_binary_index[child_ptr];

          // If this child does not point to another binary node, then skip.
          if (binary_index_of_child == 0xFFFFFFFF)
            continue;

          const LightTreeNode binary_node = binary_nodes[binary_index_of_child];

          if (binary_node.light_count == 0) {
            LightTreeChildNode left_child;
            memset(&left_child, 0, sizeof(LightTreeChildNode));

            left_child.point      = binary_node.left_ref_point;
            left_child.energy     = binary_node.left_energy;
            left_child.confidence = binary_node.left_confidence;

            child_binary_index[child_ptr] = binary_node.ptr;

            children[child_ptr] = left_child;

            LightTreeChildNode right_child;
            memset(&right_child, 0, sizeof(LightTreeChildNode));

            right_child.point      = binary_node.right_ref_point;
            right_child.energy     = binary_node.right_energy;
            right_child.confidence = binary_node.right_confidence;

            child_binary_index[child_count] = binary_node.ptr + 1;

            children[child_count++] = right_child;
          }
          else {
            child_leaf_light_ptr[child_ptr]   = binary_node.ptr;
            child_leaf_light_count[child_ptr] = binary_node.light_count;
            child_binary_index[child_ptr]     = 0xFFFFFFFF;
            child_leaf_count++;
          }

          // Child list may have run full, terminate early.
          if (child_count == 8)
            break;
        }
      }

      // The above logic has a flaw in that that leaf nodes that were inserted during the last
      // loop would not be identified as such. Hence I loop again through the children to mark
      // all children correctly as leafs.
      for (uint64_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
        const uint32_t binary_index_of_child = child_binary_index[child_ptr];

        // If this child does not point to another binary node, then skip.
        if (binary_index_of_child == 0xFFFFFFFF)
          continue;

        const LightTreeNode binary_node = binary_nodes[binary_index_of_child];

        if (binary_node.light_count > 0) {
          child_leaf_light_ptr[child_ptr]   = binary_node.ptr;
          child_leaf_light_count[child_ptr] = binary_node.light_count;
          child_binary_index[child_ptr]     = 0xFFFFFFFF;
          child_leaf_count++;
        }
      }

      uint32_t child_light_ptr = 0;

      // Now we copy the triangles for all of our leaf child nodes.
      for (uint64_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
        const uint32_t light_ptr   = child_leaf_light_ptr[child_ptr];
        const uint32_t light_count = child_leaf_light_count[child_ptr];

        // If this child is not a leaf, then skip.
        if (light_count == 0)
          continue;

        for (uint32_t i = 0; i < light_count; i++) {
          new_fragments[triangles_ptr + child_light_ptr + i]  = light_ptr + i;
          fragment_paths[triangles_ptr + child_light_ptr + i] = current_node_path | (child_ptr << (3 * current_node_depth));
        }

        children[child_ptr].light_count = light_count;

        child_light_ptr += light_count;
      }

      vec3 min_point       = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};
      vec3 max_point       = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};
      float max_energy     = 0.0f;
      float max_confidence = 0.0f;

      for (uint32_t i = 0; i < child_count; i++) {
        const vec3 point = children[i].point;

        min_point.x = min(min_point.x, point.x);
        min_point.y = min(min_point.y, point.y);
        min_point.z = min(min_point.z, point.z);

        max_point.x = max(max_point.x, point.x);
        max_point.y = max(max_point.y, point.y);
        max_point.z = max(max_point.z, point.z);

        max_energy     = fmaxf(max_energy, children[i].energy);
        max_confidence = fmaxf(max_confidence, children[i].confidence);
      }

      node.base_point = min_point;

      node.exp_x = (int8_t) (max_point.x != min_point.x) ? ceilf(log2f((max_point.x - min_point.x) * 1.0 / 255.0)) : 0;
      node.exp_y = (int8_t) (max_point.y != min_point.y) ? ceilf(log2f((max_point.y - min_point.y) * 1.0 / 255.0)) : 0;
      node.exp_z = (int8_t) (max_point.z != min_point.z) ? ceilf(log2f((max_point.z - min_point.z) * 1.0 / 255.0)) : 0;

      node.exp_confidence = ((int8_t) ceilf(log2f(max_confidence * 1.0 / 63.0)));

      const float compression_x = 1.0f / exp2f(node.exp_x);
      const float compression_y = 1.0f / exp2f(node.exp_y);
      const float compression_z = 1.0f / exp2f(node.exp_z);
      const float compression_c = 1.0f / exp2f(node.exp_confidence);

      uint64_t rel_point_x      = 0;
      uint64_t rel_point_y      = 0;
      uint64_t rel_point_z      = 0;
      uint64_t rel_energy       = 0;
      uint64_t confidence_light = 0;

      for (uint32_t i = 0; i < child_count; i++) {
        const LightTreeChildNode child_node = children[i];

        uint64_t child_rel_point_x      = (uint64_t) floorf((child_node.point.x - node.base_point.x) * compression_x);
        uint64_t child_rel_point_y      = (uint64_t) floorf((child_node.point.y - node.base_point.y) * compression_y);
        uint64_t child_rel_point_z      = (uint64_t) floorf((child_node.point.z - node.base_point.z) * compression_z);
        uint64_t child_rel_energy       = (uint64_t) floorf(255.0f * child_node.energy / max_energy);
        uint64_t child_confidence_light = (((uint64_t) (child_node.confidence * compression_c)) << 2) | ((uint64_t) child_node.light_count);

        if (child_rel_point_x > 255 || child_rel_point_y > 255 || child_rel_point_z > 255 || (child_confidence_light >> 2) > 63)
          crash_message(
            "ChildRelPoint is too large: %u %u %u %f %d %f %f %f", child_rel_point_x, child_rel_point_y, child_rel_point_z, compression_y,
            node.exp_y, child_node.point.y, node.base_point.y, child_node.point.y - node.base_point.y);

        child_rel_energy = max(1, child_rel_energy);
        rel_point_x |= child_rel_point_x << (i * 8);
        rel_point_y |= child_rel_point_y << (i * 8);
        rel_point_z |= child_rel_point_z << (i * 8);
        rel_energy |= child_rel_energy << (i * 8);
        confidence_light |= child_confidence_light << (i * 8);
      }

      node.rel_point_x[0]      = (uint32_t) (rel_point_x & 0xFFFFFFFF);
      node.rel_point_x[1]      = (uint32_t) ((rel_point_x >> 32) & 0xFFFFFFFF);
      node.rel_point_y[0]      = (uint32_t) (rel_point_y & 0xFFFFFFFF);
      node.rel_point_y[1]      = (uint32_t) ((rel_point_y >> 32) & 0xFFFFFFFF);
      node.rel_point_z[0]      = (uint32_t) (rel_point_z & 0xFFFFFFFF);
      node.rel_point_z[1]      = (uint32_t) ((rel_point_z >> 32) & 0xFFFFFFFF);
      node.rel_energy[0]       = (uint32_t) (rel_energy & 0xFFFFFFFF);
      node.rel_energy[1]       = (uint32_t) ((rel_energy >> 32) & 0xFFFFFFFF);
      node.confidence_light[0] = (uint32_t) (confidence_light & 0xFFFFFFFF);
      node.confidence_light[1] = (uint32_t) ((confidence_light >> 32) & 0xFFFFFFFF);

      if (write_ptr + child_count >= node_count) {
        node_count += binary_nodes_count;
        nodes = safe_realloc(nodes, sizeof(LightTreeNode8Packed) * node_count);
      }

      // Prepare the next nodes to be constructed from the respective binary nodes.
      for (uint64_t i = 0; i < child_count; i++) {
        node_paths[write_ptr]        = current_node_path | (i << (3 * current_node_depth));
        node_depths[write_ptr]       = current_node_depth + 1;
        nodes[write_ptr++].child_ptr = child_binary_index[i];
      }

      triangles_ptr += child_light_ptr;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;
    end_of_current_nodes   = write_ptr;
  }

  // Check
  for (uint32_t i = 0; i < fragments_count; i++) {
    if (new_fragments[i] == 0xFFFFFFFF)
      crash_message("That didn't work!");
  }

  Fragment* fragments_swap = malloc(sizeof(Fragment) * fragments_count);
  memcpy(fragments_swap, work->fragments, sizeof(Fragment) * fragments_count);

  work->paths = (uint2*) malloc(sizeof(uint2) * fragments_count);

  for (uint32_t i = 0; i < fragments_count; i++) {
    work->fragments[i] = fragments_swap[new_fragments[i]];

    uint64_t fragment_path = fragment_paths[i];

    uint2 light_path;

    light_path.x = (uint32_t) ((fragment_path >> 0) & 0x3FFFFFFF);
    light_path.y = (uint32_t) ((fragment_path >> 30) & 0x3FFFFFFF);

    work->paths[i] = light_path;
  }

  free(node_paths);
  free(node_depths);
  free(fragments_swap);
  free(new_fragments);
  free(fragment_paths);

  node_count = write_ptr;

  nodes = safe_realloc(nodes, sizeof(LightTreeNode8Packed) * node_count);

  work->nodes8_packed = nodes;
  work->nodes_8_count = node_count;
}

static void _lights_tree_finalize(LightTreeWork* work) {
  TriangleLight* reordered_lights = (TriangleLight*) malloc(sizeof(TriangleLight) * work->triangles_count);

  for (uint32_t i = 0; i < work->triangles_count; i++) {
    Fragment frag       = work->fragments[i];
    reordered_lights[i] = work->triangles[frag.id];
  }

  memcpy(work->triangles, reordered_lights, sizeof(TriangleLight) * work->triangles_count);

  free(reordered_lights);

  // TODO: Clean up this mess, this is the same as in bvh.c but I would like this to be cleaned, we need this anyway for instance support.
  void* paths;
  device_malloc(&paths, sizeof(uint2) * work->triangles_count);
  device_upload(paths, work->paths, sizeof(uint2) * work->triangles_count);

  void* light_tree_nodes;
  device_malloc(&light_tree_nodes, sizeof(LightTreeNode) * work->nodes_count);
  device_upload(light_tree_nodes, work->nodes, sizeof(LightTreeNode) * work->nodes_count);

  void* light_tree_nodes_8;
  device_malloc(&light_tree_nodes_8, sizeof(LightTreeNode8Packed) * work->nodes_8_count);
  device_upload(light_tree_nodes_8, work->nodes8_packed, sizeof(LightTreeNode8Packed) * work->nodes_8_count);

  device_update_symbol(light_tree_nodes, light_tree_nodes);
  device_update_symbol(light_tree_nodes_8, light_tree_nodes_8);
  device_update_symbol(light_tree_paths, paths);
}

static void _lights_tree_clear_work(LightTreeWork* work) {
  free(work->fragments);
  free(work->paths);
  free(work->binary_nodes);
  free(work->nodes);
  free(work->nodes8_packed);
}

#ifdef LIGHT_TREE_DEBUG_OUTPUT
static void _lights_tree_debug_output_export_binary_node(
  FILE* obj_file, FILE* mtl_file, LightTreeWork* work, uint32_t id, uint32_t* vertex_offset) {
  LightTreeBinaryNode node = work->binary_nodes[id];

  char buffer[4096];
  int buffer_offset = 0;

  uint32_t v_offset = *vertex_offset;

  buffer_offset += sprintf(buffer + buffer_offset, "o Node%u\n", id);

  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_low.x, node.self_low.y, node.self_low.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_low.x, node.self_low.y, node.self_high.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_low.x, node.self_high.y, node.self_low.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_low.x, node.self_high.y, node.self_high.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_high.x, node.self_low.y, node.self_low.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_high.x, node.self_low.y, node.self_high.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_high.x, node.self_high.y, node.self_low.z);
  buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", node.self_high.x, node.self_high.y, node.self_high.z);

  buffer_offset += sprintf(buffer + buffer_offset, "usemtl NodeMTL%u\n", id);

  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 0, v_offset + 1, v_offset + 2);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 3, v_offset + 1, v_offset + 2);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 0, v_offset + 4, v_offset + 1);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 5, v_offset + 4, v_offset + 1);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 0, v_offset + 4, v_offset + 2);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 6, v_offset + 4, v_offset + 2);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 1, v_offset + 5, v_offset + 3);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 7, v_offset + 5, v_offset + 3);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 2, v_offset + 6, v_offset + 3);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 7, v_offset + 6, v_offset + 3);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 4, v_offset + 5, v_offset + 6);
  buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 7, v_offset + 5, v_offset + 6);

  fwrite(buffer, buffer_offset, 1, obj_file);

  v_offset += 8;
  buffer_offset = 0;

  buffer_offset += sprintf(buffer + buffer_offset, "newmtl NodeMTL%u\n", id);
  buffer_offset +=
    sprintf(buffer + buffer_offset, "Kd %f %f %f\n", (id & 0b100) ? 1.0f : 0.0f, (id & 0b10) ? 1.0f : 0.0f, (id & 0b1) ? 1.0f : 0.0f);
  buffer_offset += sprintf(buffer + buffer_offset, "d %f\n", 0.1f);

  fwrite(buffer, buffer_offset, 1, mtl_file);

  buffer_offset = 0;

  if (node.type == LIGHT_TREE_NODE_TYPE_LEAF) {
    for (uint32_t i = 0; i < node.triangle_count; i++) {
      const uint32_t light_id = node.triangles_address + i;

      TriangleLight light = work->triangles[light_id];

      buffer_offset += sprintf(buffer + buffer_offset, "o Node%u - Tri%u\n", id, light_id);

      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", light.vertex.x, light.vertex.y, light.vertex.z);
      buffer_offset += sprintf(
        buffer + buffer_offset, "v %f %f %f\n", light.vertex.x + light.edge1.x, light.vertex.y + light.edge1.y,
        light.vertex.z + light.edge1.z);
      buffer_offset += sprintf(
        buffer + buffer_offset, "v %f %f %f\n", light.vertex.x + light.edge2.x, light.vertex.y + light.edge2.y,
        light.vertex.z + light.edge2.z);

      buffer_offset += sprintf(buffer + buffer_offset, "usemtl TriMTL%u\n", light_id);

      buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 0, v_offset + 1, v_offset + 2);

      fwrite(buffer, buffer_offset, 1, obj_file);

      v_offset += 3;
      buffer_offset = 0;

      buffer_offset += sprintf(buffer + buffer_offset, "newmtl TriMTL%u\n", light_id);
      buffer_offset += sprintf(
        buffer + buffer_offset, "Kd %f %f %f\n", (light_id & 0b100) ? 1.0f : 0.0f, (light_id & 0b10) ? 1.0f : 0.0f,
        (light_id & 0b1) ? 1.0f : 0.0f);
      buffer_offset += sprintf(buffer + buffer_offset, "d %f\n", 1.0f);

      fwrite(buffer, buffer_offset, 1, mtl_file);

      buffer_offset = 0;
    }
  }

  *vertex_offset = v_offset;
}

static void _lights_tree_debug_output(LightTreeWork* work) {
  FILE* obj_file = fopen("LuminaryLightTree.obj", "wb");
  FILE* mtl_file = fopen("LuminaryLightTree.mtl", "wb");

  fwrite("LuminaryLightTree.mtl\n", 22, 1, mtl_file);

  uint32_t vertex_offset = 1;

  for (uint32_t i = 1; i < work->nodes_count; i++) {
    _lights_tree_debug_output_export_binary_node(obj_file, mtl_file, work, i, &vertex_offset);
  }

  fclose(obj_file);
  fclose(mtl_file);
}
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

static void _lights_build_light_tree(Scene* scene) {
  LightTreeWork work;
  memset(&work, 0, sizeof(LightTreeWork));

  _lights_tree_create_fragments(scene, &work);
  _lights_tree_build_binary_bvh(&work);
  _lights_tree_build_traversal_structure(&work);
  _lights_tree_collapse(&work);
  _lights_tree_finalize(&work);
#ifdef LIGHT_TREE_DEBUG_OUTPUT
  _lights_tree_debug_output(&work);
#endif /* LIGHT_TREE_DEBUG_OUTPUT */
  _lights_tree_clear_work(&work);
}

static float _uint16_t_to_float(const uint16_t v) {
  const uint32_t i = 0x3F800000u | (((uint32_t) v) << 7);

  return *(float*) (&i) - 1.0f;
}

void lights_process(Scene* scene, int dmm_active) {
  bench_tic("Processing Lights");

  ////////////////////////////////////////////////////////////////////
  // Iterate over all triangles and find all light candidates.
  ////////////////////////////////////////////////////////////////////

  uint32_t candidate_lights_length = 16;
  uint32_t candidate_lights_count  = 0;
  TriangleLight* candidate_lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * candidate_lights_length);

  uint32_t lights_length = 16;
  uint32_t lights_count  = 0;
  TriangleLight* lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * candidate_lights_length);

  for (uint32_t i = 0; i < scene->triangle_data.triangle_count; i++) {
    const Triangle triangle = scene->triangles[i];

    const PackedMaterial material = scene->materials[triangle.material_id];
    const uint16_t tex_index      = material.luminance_map;

    // Triangles with displacement can't be light sources.
    if (dmm_active && scene->materials[triangle.material_id].normal_map)
      continue;

    const int is_textured_light = (tex_index != TEXTURE_NONE);

    RGBF constant_emission;

    constant_emission.r = _uint16_t_to_float(material.emission_r);
    constant_emission.g = _uint16_t_to_float(material.emission_g);
    constant_emission.b = _uint16_t_to_float(material.emission_b);

    constant_emission.r *= material.emission_scale;
    constant_emission.g *= material.emission_scale;
    constant_emission.b *= material.emission_scale;

    // Triangle is a light if it has a light texture with non-zero value at some point on the triangle's surface or it
    // has no light texture but a non-zero constant emission.
    const int is_light = is_textured_light || constant_emission.r || constant_emission.g || constant_emission.b;

    if (is_light) {
      TriangleLight light;

      light.vertex      = triangle.vertex;
      light.edge1       = triangle.edge1;
      light.edge2       = triangle.edge2;
      light.triangle_id = i;
      light.material_id = triangle.material_id;

      if (is_textured_light) {
        light.power = 0.0f;  // To be determined...

        candidate_lights[candidate_lights_count++] = light;
        if (candidate_lights_count == candidate_lights_length) {
          candidate_lights_length *= 2;
          candidate_lights = (TriangleLight*) safe_realloc(candidate_lights, sizeof(TriangleLight) * candidate_lights_length);
        }
      }
      else {
        light.power = fmaxf(constant_emission.r, fmaxf(constant_emission.g, constant_emission.b));

        scene->triangles[i].light_id = lights_count;

        lights[lights_count++] = light;
        if (lights_count == lights_length) {
          lights_length *= 2;
          lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_length);
        }
      }
    }
  }

  log_message("Number of untextured lights: %u", lights_count);
  log_message("Number of textured candidate lights: %u", candidate_lights_count);

  candidate_lights = (TriangleLight*) safe_realloc(candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  ////////////////////////////////////////////////////////////////////
  // Iterate over all light candidates and compute their power.
  ////////////////////////////////////////////////////////////////////

  float* power_dst;
  device_malloc(&power_dst, sizeof(float) * candidate_lights_count);

  TriangleLight* device_candidate_lights;
  device_malloc(&device_candidate_lights, sizeof(TriangleLight) * candidate_lights_count);
  device_upload(device_candidate_lights, candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  lights_compute_power_host(device_candidate_lights, candidate_lights_count, power_dst);

  float* power = (float*) malloc(sizeof(float) * candidate_lights_count);

  device_download(power, power_dst, sizeof(float) * candidate_lights_count);

  device_free(power_dst, sizeof(float) * candidate_lights_count);
  device_free(device_candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  float max_power = 0.0f;

  for (uint32_t i = 0; i < candidate_lights_count; i++) {
    const float candidate_light_power = power[i];

    if (candidate_light_power > 1e-6f) {
      TriangleLight light = candidate_lights[i];

      max_power = max(max_power, candidate_light_power);

      light.power = candidate_light_power;

      lights[lights_count++] = light;
      if (lights_count == lights_length) {
        lights_length *= 2;
        lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_length);
      }
    }
  }

  free(power);
  free(candidate_lights);

  log_message("Number of textured lights: %u", lights_count);
  log_message("Highest encountered light power: %f", max_power);

  lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_count);

  scene->triangle_lights       = lights;
  scene->triangle_lights_count = lights_count;

  ////////////////////////////////////////////////////////////////////
  // Create light tree.
  ////////////////////////////////////////////////////////////////////

  _lights_build_light_tree(scene);

  ////////////////////////////////////////////////////////////////////
  // Setup light ptrs in geometry.
  ////////////////////////////////////////////////////////////////////

  for (uint32_t light_id = 0; light_id < scene->triangle_lights_count; light_id++) {
    TriangleLight light                          = scene->triangle_lights[light_id];
    scene->triangles[light.triangle_id].light_id = light_id;
  }

  ////////////////////////////////////////////////////////////////////
  // Create vertex and index buffer for BVH creation.
  ////////////////////////////////////////////////////////////////////

  TriangleGeomData tri_data;

  tri_data.vertex_count   = lights_count * 3;
  tri_data.index_count    = lights_count * 3;
  tri_data.triangle_count = lights_count;

  const size_t vertex_buffer_size = sizeof(float) * 4 * 3 * lights_count;
  const size_t index_buffer_size  = sizeof(uint32_t) * 4 * lights_count;

  float* vertex_buffer   = (float*) malloc(vertex_buffer_size);
  uint32_t* index_buffer = (uint32_t*) malloc(index_buffer_size);

  for (uint32_t i = 0; i < lights_count; i++) {
    const TriangleLight l = lights[i];

    vertex_buffer[3 * 4 * i + 4 * 0 + 0] = l.vertex.x;
    vertex_buffer[3 * 4 * i + 4 * 0 + 1] = l.vertex.y;
    vertex_buffer[3 * 4 * i + 4 * 0 + 2] = l.vertex.z;
    vertex_buffer[3 * 4 * i + 4 * 0 + 3] = 1.0f;
    vertex_buffer[3 * 4 * i + 4 * 1 + 0] = l.vertex.x + l.edge1.x;
    vertex_buffer[3 * 4 * i + 4 * 1 + 1] = l.vertex.y + l.edge1.y;
    vertex_buffer[3 * 4 * i + 4 * 1 + 2] = l.vertex.z + l.edge1.z;
    vertex_buffer[3 * 4 * i + 4 * 1 + 3] = 1.0f;
    vertex_buffer[3 * 4 * i + 4 * 2 + 0] = l.vertex.x + l.edge2.x;
    vertex_buffer[3 * 4 * i + 4 * 2 + 1] = l.vertex.y + l.edge2.y;
    vertex_buffer[3 * 4 * i + 4 * 2 + 2] = l.vertex.z + l.edge2.z;
    vertex_buffer[3 * 4 * i + 4 * 2 + 3] = 1.0f;

    index_buffer[4 * i + 0] = 3 * i + 0;
    index_buffer[4 * i + 1] = 3 * i + 1;
    index_buffer[4 * i + 2] = 3 * i + 2;
    index_buffer[4 * i + 3] = 0;
  }

  device_malloc(&tri_data.vertex_buffer, vertex_buffer_size);
  device_upload(tri_data.vertex_buffer, vertex_buffer, vertex_buffer_size);

  device_malloc(&tri_data.index_buffer, index_buffer_size);
  device_upload(tri_data.index_buffer, index_buffer, index_buffer_size);

  scene->triangle_lights_data = tri_data;

  bench_toc();
}
