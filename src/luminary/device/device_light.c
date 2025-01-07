#include "device_light.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceb.h"
#include "device.h"
#include "internal_error.h"
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

struct LightTreeNode {
  vec3 left_ref_point;
  vec3 right_ref_point;
  float left_confidence;
  float right_confidence;
  float left_energy;
  float right_energy;
  uint32_t ptr;
  uint32_t light_count;
} typedef LightTreeNode;
static_assert(sizeof(LightTreeNode) == 0x30, "Incorrect packing size.");

struct LightTreeChildNode {
  vec3 point;
  float energy;
  float confidence;
  uint32_t light_count;
} typedef LightTreeChildNode;

struct LightTreeWork {
  LightTreeFragment* fragments;
  uint32_t fragments_count;
  uint2* paths;
  ARRAY LightTreeBinaryNode* binary_nodes;
  LightTreeNode* nodes;
  LightTreeNode8Packed* nodes8_packed;
  uint32_t nodes_count;
  uint32_t nodes_8_count;
} typedef LightTreeWork;

struct Bin {
  Vec128 high;
  Vec128 low;
  int32_t entry;
  int32_t exit;
  float energy;
  uint32_t padding;
} typedef Bin;

#define OBJECT_SPLIT_BIN_COUNT_LOG (5)
#define OBJECT_SPLIT_BIN_COUNT (1 << 5)

// We need to bound the dimensions, the number must be large but still much smaller than FLT_MAX
#define MAX_VALUE 1e10f

#define FRAGMENT_ERROR_COMP (FLT_EPSILON * 4.0f)

inline void _light_tree_fit_bounds(
  const LightTreeFragment* fragments, const uint32_t fragments_count, Vec128* restrict high_out, Vec128* restrict low_out) {
  Vec128 high = vec128_set_1(-MAX_VALUE);
  Vec128 low  = vec128_set_1(MAX_VALUE);

  for (uint32_t i = 0; i < fragments_count; i++) {
    // TODO: Split fragment array memory layout so that bounds fitting can be done in a more cache friendly way
    const Vec128 high_frag = vec128_load((float*) &(fragments[i].high));
    const Vec128 low_frag  = vec128_load((float*) &(fragments[i].low));

    high = vec128_max(high, high_frag);
    low  = vec128_min(low, low_frag);
  }

  if (high_out)
    vec128_store((float*) high_out, high);
  if (low_out)
    vec128_store((float*) low_out, low);
}

inline void _light_tree_fit_bounds_of_bins(
  const Bin* bins, const uint32_t bins_count, Vec128* restrict high_out, Vec128* restrict low_out) {
  Vec128 high = vec128_set_1(-MAX_VALUE);
  Vec128 low  = vec128_set_1(MAX_VALUE);

  for (uint32_t i = 0; i < bins_count; i++) {
    // TODO: Split bin array memory layout so that bounds fitting can be done in a more cache friendly way
    const Vec128 high_frag = vec128_load((float*) &(bins[i].high));
    const Vec128 low_frag  = vec128_load((float*) &(bins[i].low));

    high = vec128_max(high, high_frag);
    low  = vec128_min(low, low_frag);
  }

  if (high_out)
    vec128_store((float*) high_out, high);
  if (low_out)
    vec128_store((float*) low_out, low);
}

#define _light_tree_update_bounds_of_bins(__macro_in_bins, __macro_in_high, __macro_in_low) \
  {                                                                                         \
    const float* __macro_baseptr  = (float*) (__macro_in_bins);                             \
    const Vec128 __macro_high_bin = vec128_load(__macro_baseptr);                           \
    const Vec128 __macro_low_bin  = vec128_load(__macro_baseptr + 4);                       \
    __macro_in_high               = vec128_max(__macro_in_high, __macro_high_bin);          \
    __macro_in_low                = vec128_min(__macro_in_low, __macro_low_bin);            \
  }

#define _light_tree_construct_bins_kernel(                                                                                                \
  __macro_in_fragments, __macro_in_fragments_count, __macro_in_bins, __macro_in_low_axis, __macro_in_interval, __macro_in_axis_)          \
  {                                                                                                                                       \
    const double __macro_inv_interval = 1.0 / __macro_in_interval;                                                                        \
    for (uint32_t __macro_i = 0; __macro_i < __macro_in_fragments_count; __macro_i++) {                                                   \
      const double __macro_value = vec128_get_1(vec128_load((const float*) &(__macro_in_fragments[__macro_i].middle)), __macro_in_axis_); \
      int32_t __macro_pos        = ((int32_t) ceil((__macro_value - __macro_in_low_axis) * __macro_inv_interval)) - 1;                    \
      if (__macro_pos < 0)                                                                                                                \
        __macro_pos = 0;                                                                                                                  \
                                                                                                                                          \
      __macro_in_bins[__macro_pos].entry++;                                                                                               \
      __macro_in_bins[__macro_pos].exit++;                                                                                                \
      __macro_in_bins[__macro_pos].energy += __macro_in_fragments[__macro_i].power;                                                       \
                                                                                                                                          \
      Vec128 __macro_high_bin        = vec128_load((const float*) &(__macro_in_bins[__macro_pos].high));                                  \
      Vec128 __macro_low_bin         = vec128_load((const float*) &(__macro_in_bins[__macro_pos].low));                                   \
      const Vec128 __macro_high_frag = vec128_load((const float*) &(__macro_in_fragments[__macro_i].high));                               \
      const Vec128 __macro_low_frag  = vec128_load((const float*) &(__macro_in_fragments[__macro_i].low));                                \
                                                                                                                                          \
      __macro_high_bin = vec128_max(__macro_high_bin, __macro_high_frag);                                                                 \
      __macro_low_bin  = vec128_min(__macro_low_bin, __macro_low_frag);                                                                   \
                                                                                                                                          \
      vec128_store((float*) &(__macro_in_bins[__macro_pos].high), __macro_high_bin);                                                      \
      vec128_store((float*) &(__macro_in_bins[__macro_pos].low), __macro_low_bin);                                                        \
    }                                                                                                                                     \
  }

static double _light_tree_construct_bins(
  Bin* bins, const LightTreeFragment* fragments, const uint32_t fragments_count, const LightTreeSweepAxis axis, double* offset) {
  Vec128 high, low;
  _light_tree_fit_bounds(fragments, fragments_count, &high, &low);

  const double high_axis = high.data[axis];
  const double low_axis  = low.data[axis];

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

  // Axis must be a compile time constant here.
  switch (axis) {
    case LIGHT_TREE_SWEEP_AXIS_X:
      _light_tree_construct_bins_kernel(fragments, fragments_count, bins, low_axis, interval, LIGHT_TREE_SWEEP_AXIS_X);
      break;
    case LIGHT_TREE_SWEEP_AXIS_Y:
      _light_tree_construct_bins_kernel(fragments, fragments_count, bins, low_axis, interval, LIGHT_TREE_SWEEP_AXIS_Y);
      break;
    case LIGHT_TREE_SWEEP_AXIS_Z:
      _light_tree_construct_bins_kernel(fragments, fragments_count, bins, low_axis, interval, LIGHT_TREE_SWEEP_AXIS_Z);
      break;
  }

  return interval;
}

static void _light_tree_divide_middles_along_axis(
  const double split, const LightTreeSweepAxis axis, LightTreeFragment* fragments, const uint32_t fragments_count) {
  uint32_t left  = 0;
  uint32_t right = 0;

  while (left + right < fragments_count) {
    const LightTreeFragment frag = fragments[left];

    const double middle = frag.middle.data[axis];

    if (middle > split) {
      const uint32_t swap_index = fragments_count - 1 - right;

      LightTreeFragment temp = fragments[swap_index];
      fragments[swap_index]  = frag;
      fragments[left]        = temp;

      right++;
    }
    else {
      left++;
    }
  }
}

static LuminaryResult _light_tree_build_binary_bvh(LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  LightTreeFragment* fragments = work->fragments;
  uint32_t fragments_count     = work->fragments_count;

  ARRAY LightTreeBinaryNode* nodes;
  __FAILURE_HANDLE(array_create(&nodes, sizeof(LightTreeBinaryNode), 1 + fragments_count));

  if (fragments_count == 0) {
    work->binary_nodes = nodes;
    work->nodes_count  = 0;
    return LUMINARY_SUCCESS;
  }

  {
    LightTreeBinaryNode root_node;
    memset(&root_node, 0, sizeof(LightTreeBinaryNode));

    root_node.triangles_address = 0;
    root_node.triangle_count    = fragments_count;
    root_node.type              = LIGHT_TREE_NODE_TYPE_LEAF;
    root_node.path              = 0;
    root_node.depth             = 0;

    __FAILURE_HANDLE(array_push(&nodes, &root_node));
  }

  Bin* bins;
  __FAILURE_HANDLE(host_malloc(&bins, sizeof(Bin) * OBJECT_SPLIT_BIN_COUNT));

  uint32_t begin_of_current_nodes = 0;
  uint32_t end_of_current_nodes   = 1;

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
      Vec128 high_parent, low_parent;
      _light_tree_fit_bounds(fragments + fragments_ptr, fragments_count, &high_parent, &low_parent);

      const Vec128 diff               = vec128_set_w_to_0(vec128_sub(high_parent, low_parent));
      const float max_axis_interval   = vec128_hmax(diff);
      const float parent_surface_area = vec128_box_area(diff);

      Vec128 high, low;
      double optimal_cost = DBL_MAX;
      LightTreeSweepAxis axis;
      double optimal_splitting_plane;
      bool found_split = false;
      uint32_t optimal_split;
      float optimal_left_energy, optimal_right_energy;

      // For each axis, perform a greedy search for an optimal split.
      for (uint32_t a = 0; a < 3; a++) {
        double low_split;
        const double interval =
          _light_tree_construct_bins(bins, fragments + fragments_ptr, fragments_count, (LightTreeSweepAxis) a, &low_split);

        if (interval == 0.0)
          continue;

        const double interval_cost = max_axis_interval / interval;

        uint32_t left = 0;

        float left_energy  = 0.0f;
        float right_energy = 0.0f;

        for (uint32_t k = 0; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          right_energy += bins[k].energy;
        }

        Vec128 high_left  = vec128_set_1(-MAX_VALUE);
        Vec128 high_right = vec128_set_1(-MAX_VALUE);
        Vec128 low_left   = vec128_set_1(MAX_VALUE);
        Vec128 low_right  = vec128_set_1(MAX_VALUE);

        for (uint32_t k = 1; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          _light_tree_update_bounds_of_bins(bins + k - 1, high_left, low_left);
          _light_tree_fit_bounds_of_bins(bins + k, OBJECT_SPLIT_BIN_COUNT - k, &high_right, &low_right);

          left_energy += bins[k - 1].energy;
          right_energy -= bins[k - 1].energy;

          const Vec128 diff_left  = vec128_sub(high_left, low_left);
          const Vec128 diff_right = vec128_sub(high_right, low_right);

          const float left_area  = vec128_box_area(vec128_set_w_to_0(diff_left));
          const float right_area = vec128_box_area(vec128_set_w_to_0(diff_right));

          left += bins[k - 1].entry;

          if (left == 0 || left == fragments_count)
            continue;

          const double total_cost = interval_cost * (left_energy * left_area + right_energy * right_area) / parent_surface_area;

          if (total_cost < optimal_cost) {
            optimal_cost            = total_cost;
            optimal_split           = left;
            optimal_splitting_plane = low_split + k * interval;
            found_split             = true;
            axis                    = a;
            optimal_left_energy     = left_energy;
            optimal_right_energy    = right_energy;
          }
        }
      }

      if (found_split) {
        _light_tree_divide_middles_along_axis(optimal_splitting_plane, axis, fragments + fragments_ptr, fragments_count);
      }
      else {
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
      }

      node.left_energy  = optimal_left_energy;
      node.right_energy = optimal_right_energy;

      _light_tree_fit_bounds(fragments + fragments_ptr, optimal_split, &high, &low);

      node.left_high.x = high.x;
      node.left_high.y = high.y;
      node.left_high.z = high.z;
      node.left_low.x  = low.x;
      node.left_low.y  = low.y;
      node.left_low.z  = low.z;

      __FAILURE_HANDLE(array_get_num_elements(nodes, &node.child_address));

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

      __FAILURE_HANDLE(array_push(&nodes, &node_left));

      _light_tree_fit_bounds(fragments + fragments_ptr + optimal_split, node.triangle_count - optimal_split, &high, &low);

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

      __FAILURE_HANDLE(array_push(&nodes, &node_right));

      node.type = LIGHT_TREE_NODE_TYPE_INTERNAL;

      nodes[node_ptr] = node;
    }

    begin_of_current_nodes = end_of_current_nodes;

    __FAILURE_HANDLE(array_get_num_elements(nodes, &end_of_current_nodes));
  }

  __FAILURE_HANDLE(host_free(&bins));

  work->binary_nodes = nodes;

  __FAILURE_HANDLE(array_get_num_elements(work->binary_nodes, &work->nodes_count));

  return LUMINARY_SUCCESS;
}

static void _lights_get_ref_point_and_dist(LightTreeWork* work, LightTreeBinaryNode node, float energy, vec3* ref_point, float* ref_dist) {
  const float inverse_total_energy = 1.0f / energy;

  vec3 p = {.x = 0.0f, .y = 0.0f, .z = 0.0f};
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    const LightTreeFragment frag = work->fragments[node.triangles_address + i];

    const float weight = frag.power * inverse_total_energy;

    p.x += weight * frag.middle.x;
    p.y += weight * frag.middle.y;
    p.z += weight * frag.middle.z;
  }

  float weighted_dist = 0.0f;
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    const LightTreeFragment frag = work->fragments[node.triangles_address + i];

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
static LuminaryResult _light_tree_build_traversal_structure(LightTreeWork* work) {
  LightTreeNode* nodes;
  __FAILURE_HANDLE(host_malloc(&nodes, sizeof(LightTreeNode) * work->nodes_count));

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
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Encountered illegal node type!");
    }

    nodes[i] = node;
  }

  work->nodes = nodes;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collapse(LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  if (work->nodes_count == 0) {
    __FAILURE_HANDLE(host_malloc(&work->paths, 0));
    __FAILURE_HANDLE(host_malloc(&work->nodes8_packed, 0));
    work->nodes_8_count = 0;
    return LUMINARY_SUCCESS;
  }

  const uint32_t fragments_count = work->fragments_count;

  LightTreeNode* binary_nodes       = work->nodes;
  const uint32_t binary_nodes_count = work->nodes_count;

  uint32_t node_count = binary_nodes_count;

  LightTreeNode8Packed* nodes;
  __FAILURE_HANDLE(host_malloc(&nodes, sizeof(LightTreeNode8Packed) * node_count));

  uint64_t* node_paths;
  __FAILURE_HANDLE(host_malloc(&node_paths, sizeof(uint64_t) * node_count));

  uint32_t* node_depths;
  __FAILURE_HANDLE(host_malloc(&node_depths, sizeof(uint32_t) * node_count));

  uint32_t* new_fragments;
  __FAILURE_HANDLE(host_malloc(&new_fragments, sizeof(uint32_t) * fragments_count));

  uint64_t* fragment_paths;
  __FAILURE_HANDLE(host_malloc(&fragment_paths, sizeof(uint64_t) * fragments_count));

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

      for (uint32_t i = 0; i < 8; i++) {
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

#ifdef LIGHT_TREE_DEBUG_OUTPUT
      info_message("==== %u ====", node_ptr);
      info_message("Meta:  %u (%u) %u", node.child_ptr, child_count, node.light_ptr);
      info_message("Point: (%f, %f, %f)", node.base_point.x, node.base_point.y, node.base_point.z);
      info_message(
        "Exponents: %d %d %d => %f %f %f | %d => %f", node.exp_x, node.exp_y, node.exp_z, 1.0f / compression_x, 1.0f / compression_y,
        1.0f / compression_z, node.exp_confidence, 1.0f / compression_c);
#endif

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

        if (child_rel_point_x > 255 || child_rel_point_y > 255 || child_rel_point_z > 255 || (child_confidence_light >> 2) > 63) {
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Fatal error during light tree compression. Value exceeded bit limit.");
        }

        child_rel_energy = max(1, child_rel_energy);

#ifdef LIGHT_TREE_DEBUG_OUTPUT
        info_message(
          "[%u] %llX %llX %llX %llX %llX", i, child_rel_point_x, child_rel_point_y, child_rel_point_z, child_rel_energy,
          child_confidence_light);

        {
          // Check error of compressed point

          const float decompression_x = exp2f(node.exp_x);
          const float decompression_y = exp2f(node.exp_y);
          const float decompression_z = exp2f(node.exp_z);

          const float decompressed_point_x = child_rel_point_x * decompression_x + node.base_point.x;
          const float decompressed_point_y = child_rel_point_y * decompression_y + node.base_point.y;
          const float decompressed_point_z = child_rel_point_z * decompression_z + node.base_point.z;

          const float diff_x = decompressed_point_x - child_node.point.x;
          const float diff_y = decompressed_point_y - child_node.point.y;
          const float diff_z = decompressed_point_z - child_node.point.z;

          const float error = sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

          info_message(
            "    (%f, %f, %f) => (%f, %f, %f) [Err: %f]", child_node.point.x, child_node.point.y, child_node.point.z, decompressed_point_x,
            decompressed_point_y, decompressed_point_z, error);
        }

        {
          // Check error of confidence

          const float decompression_c = exp2f(node.exp_confidence);

          const float decompressed_confidence = (child_confidence_light >> 2) * decompression_c;

          info_message(
            "    %f => %f [Err: %f]", child_node.confidence, decompressed_confidence, child_node.confidence - decompressed_confidence);
        }
#endif

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
    if (new_fragments[i] == 0xFFFFFFFF || fragment_paths[i] == 0xFFFFFFFFFFFFFFFF) {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Fatal error during light tree compression. Light was lost.");
    }
  }

  LightTreeFragment* fragments_swap;
  __FAILURE_HANDLE(host_malloc(&fragments_swap, sizeof(LightTreeFragment) * fragments_count));

  memcpy(fragments_swap, work->fragments, sizeof(LightTreeFragment) * fragments_count);

  __FAILURE_HANDLE(host_malloc(&work->paths, sizeof(uint2) * fragments_count));

  for (uint32_t i = 0; i < fragments_count; i++) {
    work->fragments[i] = fragments_swap[new_fragments[i]];

    uint64_t fragment_path = fragment_paths[i];

    uint2 light_path;

    light_path.x = (uint32_t) ((fragment_path >> 0) & 0x3FFFFFFF);
    light_path.y = (uint32_t) ((fragment_path >> 30) & 0x3FFFFFFF);

    work->paths[i] = light_path;
  }

  __FAILURE_HANDLE(host_free(&node_paths));
  __FAILURE_HANDLE(host_free(&node_depths));
  __FAILURE_HANDLE(host_free(&fragments_swap));
  __FAILURE_HANDLE(host_free(&new_fragments));
  __FAILURE_HANDLE(host_free(&fragment_paths));

  node_count = write_ptr;

  __FAILURE_HANDLE(host_realloc(&nodes, sizeof(LightTreeNode8Packed) * node_count));

  work->nodes8_packed = nodes;
  work->nodes_8_count = node_count;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_finalize(LightTree* tree, LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(work);

  ////////////////////////////////////////////////////////////////////
  // Apply permutation obtained in tree construction
  ////////////////////////////////////////////////////////////////////

  TriangleHandle* tri_handle_map;
  __FAILURE_HANDLE(host_malloc(&tri_handle_map, sizeof(TriangleHandle) * work->fragments_count));

  LightTreeBVHTriangle* bvh_triangles;
  __FAILURE_HANDLE(host_malloc(&bvh_triangles, sizeof(LightTreeBVHTriangle) * work->fragments_count));

  for (uint32_t id = 0; id < work->fragments_count; id++) {
    const LightTreeFragment frag = work->fragments[id];

    const LightTreeCacheInstance* instance = tree->cache.instances + frag.instance_id;

    bvh_triangles[id] = instance->bvh_triangles[frag.instance_cache_tri_id];

    const TriangleHandle handle = (TriangleHandle){.instance_id = frag.instance_id, .tri_id = frag.tri_id};

    tri_handle_map[id] = handle;
  }

  ////////////////////////////////////////////////////////////////////
  // Assign light tree data
  ////////////////////////////////////////////////////////////////////

  tree->nodes_size = sizeof(LightTreeNode8Packed) * work->nodes_8_count;
  tree->nodes_data = (void*) work->nodes8_packed;

  tree->paths_size = sizeof(uint2) * work->fragments_count;
  tree->paths_data = (void*) work->paths;

  tree->tri_handle_map_size = sizeof(TriangleHandle) * work->fragments_count;
  tree->tri_handle_map_data = (void*) tri_handle_map;

  tree->bvh_vertex_buffer_data = (void*) bvh_triangles;
  tree->light_count            = work->fragments_count;

  work->nodes8_packed = (LightTreeNode8Packed*) 0;
  work->paths         = (uint2*) 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_clear_work(LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  __FAILURE_HANDLE(host_free(&work->fragments));
  __FAILURE_HANDLE(array_destroy(&work->binary_nodes));
  __FAILURE_HANDLE(host_free(&work->nodes));

  return LUMINARY_SUCCESS;
}

#ifdef LIGHT_TREE_DEBUG_OUTPUT
static void _light_tree_debug_output_export_binary_node(
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

      // This is crashing, no idea why. This is due to changes in the host rework project.
      LightTreeBVHTriangle light = work->bvh_triangles[light_id];

      buffer_offset += sprintf(buffer + buffer_offset, "o Node%u - Tri%u\n", id, light_id);

      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", light.vertex.x, light.vertex.y, light.vertex.z);
      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", light.vertex1.x, light.vertex1.y, light.vertex1.z);
      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", light.vertex2.x, light.vertex2.y, light.vertex2.z);

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

static void _light_tree_debug_output(LightTreeWork* work) {
  FILE* obj_file = fopen("LuminaryLightTree.obj", "wb");
  FILE* mtl_file = fopen("LuminaryLightTree.mtl", "wb");

  fwrite("LuminaryLightTree.mtl\n", 22, 1, mtl_file);

  uint32_t vertex_offset = 1;

  for (uint32_t i = 1; i < work->nodes_count; i++) {
    _light_tree_debug_output_export_binary_node(obj_file, mtl_file, work, i, &vertex_offset);
  }

  fclose(obj_file);
  fclose(mtl_file);
}
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

// TODO: Remove if no longer needed
static float _uint16_t_to_float(const uint16_t v) {
  const uint32_t i = 0x3F800000u | (((uint32_t) v) << 7);

  return *(float*) (&i) - 1.0f;
}

////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////

LuminaryResult light_tree_create(LightTree** tree) {
  __CHECK_NULL_ARGUMENT(tree);

  __FAILURE_HANDLE(host_malloc(tree, sizeof(LightTree)));
  memset(*tree, 0, sizeof(LightTree));

  ////////////////////////////////////////////////////////////////////
  // Initialize cache
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(array_create(&(*tree)->cache.meshes, sizeof(LightTreeCacheMesh), 4));
  __FAILURE_HANDLE(array_create(&(*tree)->cache.instances, sizeof(LightTreeCacheInstance), 4));
  __FAILURE_HANDLE(array_create(&(*tree)->cache.materials, sizeof(LightTreeCacheMaterial), 4));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Mesh updating
////////////////////////////////////////////////////////////////////

static LuminaryResult _light_tree_update_cache_mesh_has_emission(LightTreeCacheMesh* mesh, const ARRAY LightTreeCacheMaterial* materials) {
  __CHECK_NULL_ARGUMENT(mesh);
  __CHECK_NULL_ARGUMENT(materials);

  uint32_t num_materials;
  __FAILURE_HANDLE(array_get_num_elements(mesh->materials, &num_materials));

  mesh->has_emission = false;

  uint32_t num_cached_materials;
  __FAILURE_HANDLE(array_get_num_elements(materials, &num_cached_materials));

  for (uint32_t material_slot_id = 0; material_slot_id < num_materials; material_slot_id++) {
    const uint16_t material_id = mesh->materials[material_slot_id];

    if (material_id < num_cached_materials && materials[material_id].has_emission) {
      mesh->has_emission = true;
      break;
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_update_cache_mesh(
  LightTreeCacheMesh* cache, const ARRAY LightTreeCacheMaterial* materials, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(mesh);

  ////////////////////////////////////////////////////////////////////
  // Clear all the triangle related cache, modifying meshes is not supposed to be fast
  ////////////////////////////////////////////////////////////////////

  if (cache->materials) {
    uint32_t num_materials_allocated;
    __FAILURE_HANDLE(array_get_num_elements(cache->materials, &num_materials_allocated));

    for (uint32_t material_id = 0; material_id < num_materials_allocated; material_id++) {
      __FAILURE_HANDLE(array_destroy(cache->material_triangles + material_id));
    }

    __FAILURE_HANDLE(array_clear(&cache->materials));
  }

  if (cache->material_triangles) {
    __FAILURE_HANDLE(array_clear(&cache->material_triangles));
  }

  __FAILURE_HANDLE(array_create(&cache->materials, sizeof(uint16_t), 16));
  __FAILURE_HANDLE(array_create(&cache->material_triangles, sizeof(LightTreeCacheTriangle*), 16));

  ////////////////////////////////////////////////////////////////////
  // Generate triangle caches
  ////////////////////////////////////////////////////////////////////

  uint32_t num_materials = 0;

  for (uint32_t tri_id = 0; tri_id < mesh->data.triangle_count; tri_id++) {
    const Triangle* triangle = mesh->triangles + tri_id;

    const uint16_t material_id = triangle->material_id;

    uint32_t material_slot = 0xFFFFFFFF;

    for (uint32_t material_slot_id = 0; material_slot_id < num_materials; material_slot_id++) {
      if (cache->materials[material_slot_id] == material_id) {
        material_slot = material_slot_id;
        break;
      }
    }

    // Material is not present in the cache yet, so add it.
    if (material_slot == 0xFFFFFFFF) {
      material_slot = num_materials;
      __FAILURE_HANDLE(array_push(&cache->materials, &material_id));
      num_materials++;

      LightTreeCacheTriangle* material_triangles;
      __FAILURE_HANDLE(array_create(&material_triangles, sizeof(LightTreeCacheTriangle), 16));

      __FAILURE_HANDLE(array_push(&cache->material_triangles, &material_triangles));
    }

    const Vec128 vertex = vec128_set(triangle->vertex.x, triangle->vertex.y, triangle->vertex.z, 0.0f);
    const Vec128 edge1  = vec128_set(triangle->edge1.x, triangle->edge1.y, triangle->edge1.z, 0.0f);
    const Vec128 edge2  = vec128_set(triangle->edge2.x, triangle->edge2.y, triangle->edge2.z, 0.0f);

    LightTreeCacheTriangle cache_triangle;
    cache_triangle.tri_id  = tri_id;
    cache_triangle.vertex  = vertex;
    cache_triangle.vertex1 = vec128_add(vertex, edge1);
    cache_triangle.vertex2 = vec128_add(vertex, edge2);
    cache_triangle.cross   = vec128_cross(edge1, edge2);

    __FAILURE_HANDLE(array_push(&cache->material_triangles[material_slot], &cache_triangle));
  }

  // Some materials might not be cached at this point in time but it is the duty of the materials to update whether the mesh has emission
  __FAILURE_HANDLE(_light_tree_update_cache_mesh_has_emission(cache, materials));

  if (cache->has_emission) {
    cache->is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_update_cache_mesh(LightTree* tree, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(mesh);

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.meshes, &num_meshes));

  if (mesh->id >= num_meshes) {
    __FAILURE_HANDLE(array_set_num_elements(&tree->cache.meshes, mesh->id + 1));
  }

  __FAILURE_HANDLE(_light_tree_update_cache_mesh(tree->cache.meshes + mesh->id, tree->cache.materials, mesh));

  if (tree->cache.meshes[mesh->id].is_dirty) {
    tree->cache.is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Instance updating
////////////////////////////////////////////////////////////////////

static LuminaryResult _light_tree_update_cache_instance(
  LightTreeCacheInstance* cache, ARRAY LightTreeCacheMesh* cache_meshes, const MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(instance);

  bool instance_is_dirty          = false;
  bool previous_mesh_has_emission = false;

  if (cache->active != instance->active) {
    cache->active     = instance->active;
    instance_is_dirty = true;
  }
  else if (!cache->active) {
    // Skip further processing if this instance is inactive.
    return LUMINARY_SUCCESS;
  }

  if (cache->mesh_id != instance->mesh_id) {
    uint32_t num_meshes;
    __FAILURE_HANDLE(array_get_num_elements(cache_meshes, &num_meshes));

    if (cache->mesh_id != MESH_ID_INVALID) {
      if (num_meshes <= cache->mesh_id || num_meshes <= instance->mesh_id) {
        __RETURN_ERROR(
          LUMINARY_ERROR_API_EXCEPTION, "MeshID [%u/%u] was out of range [%u].", cache->mesh_id, instance->mesh_id, num_meshes);
      }

      cache_meshes[cache->mesh_id].instance_count--;

      if (cache_meshes[cache->mesh_id].has_emission) {
        previous_mesh_has_emission = true;
      }
    }

    cache_meshes[instance->mesh_id].instance_count++;

    cache->mesh_id    = instance->mesh_id;
    instance_is_dirty = true;
  }

  const bool current_mesh_has_emission = cache_meshes[cache->mesh_id].has_emission;

  if (memcmp(&cache->rotation, &instance->rotation, sizeof(Quaternion))) {
    cache->rotation   = instance->rotation;
    instance_is_dirty = true;
  }

  if (memcmp(&cache->scale, &instance->scale, sizeof(vec3))) {
    cache->scale      = instance->scale;
    instance_is_dirty = true;
  }

  if (memcmp(&cache->translation, &instance->translation, sizeof(vec3))) {
    cache->translation = instance->translation;
    instance_is_dirty  = true;
  }

  // Only set the dirty flag if the referenced mesh is present or was present in the light tree.
  if (instance_is_dirty && (current_mesh_has_emission || (current_mesh_has_emission != previous_mesh_has_emission))) {
    cache->is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_update_cache_instance(LightTree* tree, const MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(instance);

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.instances, &num_instances));

  if (instance->id >= num_instances) {
    __FAILURE_HANDLE(array_set_num_elements(&tree->cache.instances, instance->id + 1));

    // Invalidate the mesh ID for proper reference counting.
    for (uint32_t instance_id = num_instances; instance_id < instance->id + 1; instance_id++) {
      tree->cache.instances[instance_id].mesh_id = MESH_ID_INVALID;
    }
  }

  __FAILURE_HANDLE(_light_tree_update_cache_instance(tree->cache.instances + instance->id, tree->cache.meshes, instance));

  if (tree->cache.instances[instance->id].is_dirty) {
    tree->cache.is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Material updating
////////////////////////////////////////////////////////////////////

static LuminaryResult _light_tree_update_cache_material(LightTreeCacheMaterial* cache, const Material* material, bool* meshes_need_update) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(material);

  bool has_emission      = false;
  bool material_is_dirty = false;
  *meshes_need_update    = false;

  const bool has_textured_emission = material->luminance_tex != TEXTURE_NONE;
  if (cache->has_textured_emission != has_textured_emission) {
    cache->has_textured_emission = has_textured_emission;
    material_is_dirty            = true;
  }

  float intensity;
  if (has_textured_emission) {
    // TODO: Check if triangle_has_textured_emission && (emission texture hash are not equal) then reintegrate.
    intensity = material->emission_scale;
  }
  else {
    intensity = fmaxf(material->emission.r, fmaxf(material->emission.g, material->emission.b));
  }

  if (cache->constant_emission_intensity != intensity) {
    cache->constant_emission_intensity = intensity;
    material_is_dirty                  = true;
  }

  if (intensity > 0.0f) {
    has_emission = true;
  }

  if (cache->has_emission != has_emission) {
    cache->has_emission = has_emission;
    material_is_dirty   = true;
    *meshes_need_update = true;
  }

  if (has_emission && material_is_dirty) {
    cache->is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_update_cache_material(LightTree* tree, const Material* material) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(material);

  uint32_t num_materials;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.materials, &num_materials));

  if (material->id >= num_materials) {
    __FAILURE_HANDLE(array_set_num_elements(&tree->cache.materials, material->id + 1));
  }

  bool meshes_need_update = false;
  __FAILURE_HANDLE(_light_tree_update_cache_material(tree->cache.materials + material->id, material, &meshes_need_update));

  if (tree->cache.materials[material->id].is_dirty) {
    tree->cache.is_dirty = true;

    tree->cache.materials[material->id].is_dirty = false;
  }

  if (meshes_need_update) {
    uint32_t num_meshes;
    __FAILURE_HANDLE(array_get_num_elements(tree->cache.meshes, &num_meshes));

    for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
      __FAILURE_HANDLE(_light_tree_update_cache_mesh_has_emission(tree->cache.meshes + mesh_id, tree->cache.materials));
      tree->cache.meshes[mesh_id].is_dirty = true;
    }
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Light Tree construction
////////////////////////////////////////////////////////////////////

static LuminaryResult _light_tree_compute_instance_fragments(LightTree* tree, const uint32_t instance_id) {
  __CHECK_NULL_ARGUMENT(tree);

  LightTreeCacheInstance* instance = tree->cache.instances + instance_id;

  if (instance->fragments) {
    __FAILURE_HANDLE(array_destroy(&instance->fragments));
  }

  if (instance->bvh_triangles) {
    __FAILURE_HANDLE(array_destroy(&instance->bvh_triangles));
  }

  if (instance->mesh_id == MESH_ID_INVALID)
    return LUMINARY_SUCCESS;

  const LightTreeCacheMesh* mesh = tree->cache.meshes + instance->mesh_id;

  if (!mesh->has_emission)
    return LUMINARY_SUCCESS;

  uint32_t num_materials;
  __FAILURE_HANDLE(array_get_num_elements(mesh->materials, &num_materials));

  uint32_t num_cached_materials;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.materials, &num_cached_materials));

  __FAILURE_HANDLE(array_create(&instance->fragments, sizeof(LightTreeFragment), 16));
  __FAILURE_HANDLE(array_create(&instance->bvh_triangles, sizeof(LightTreeBVHTriangle), 16));

  for (uint32_t material_slot_id = 0; material_slot_id < num_materials; material_slot_id++) {
    const uint16_t material_id = mesh->materials[material_slot_id];

    // Material is not cached, consider it non existent.
    if (material_id >= num_cached_materials)
      continue;

    const LightTreeCacheMaterial* material = tree->cache.materials + material_id;

    if (!material->has_emission)
      continue;

    const ARRAY LightTreeCacheTriangle* material_triangles = mesh->material_triangles[material_slot_id];

    uint32_t num_material_triangles;
    __FAILURE_HANDLE(array_get_num_elements(material_triangles, &num_material_triangles));

    for (uint32_t tri_id = 0; tri_id < num_material_triangles; tri_id++) {
      const LightTreeCacheTriangle* triangle = material_triangles + tri_id;

      // TODO: Apply instance transformation.

      const float area = 0.5f * vec128_norm2(triangle->cross);

      if (area == 0.0f)
        continue;

      LightTreeFragment fragment;
      fragment.low         = vec128_min(triangle->vertex, vec128_min(triangle->vertex1, triangle->vertex2));
      fragment.high        = vec128_max(triangle->vertex, vec128_max(triangle->vertex1, triangle->vertex2));
      fragment.middle      = vec128_mul(vec128_add(fragment.low, fragment.high), vec128_set_1(0.5f));
      fragment.power       = material->constant_emission_intensity * area;
      fragment.instance_id = instance_id;
      fragment.tri_id      = triangle->tri_id;

      __FAILURE_HANDLE(array_get_num_elements(instance->bvh_triangles, &fragment.instance_cache_tri_id));

      __FAILURE_HANDLE(array_push(&instance->fragments, &fragment));

      LightTreeBVHTriangle bvh_triangle;
      bvh_triangle.vertex  = triangle->vertex;
      bvh_triangle.vertex1 = triangle->vertex1;
      bvh_triangle.vertex2 = triangle->vertex2;

      __FAILURE_HANDLE(array_push(&instance->bvh_triangles, &bvh_triangle));
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_handle_dirty_states(LightTree* tree) {
  __CHECK_NULL_ARGUMENT(tree);

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.instances, &num_instances));

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    bool instance_is_dirty = tree->cache.instances[instance_id].is_dirty;

    const uint32_t mesh_id = tree->cache.instances[instance_id].mesh_id;

    if (mesh_id != MESH_ID_INVALID) {
      instance_is_dirty |= tree->cache.meshes[mesh_id].is_dirty;
    }

    if (instance_is_dirty) {
      __FAILURE_HANDLE(_light_tree_compute_instance_fragments(tree, instance_id));
    }

    tree->cache.instances[instance_id].is_dirty = false;
  }

  // Reset mesh dirty flags
  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.meshes, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    tree->cache.meshes[mesh_id].is_dirty = false;
  }

  tree->cache.is_dirty = false;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collect_fragments(LightTree* tree, LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(work);

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.instances, &num_instances));

  uint32_t total_fragments = 0;

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    LightTreeCacheInstance* instance = tree->cache.instances + instance_id;

    if (!instance->active)
      continue;

    uint32_t num_fragments;
    __FAILURE_HANDLE(array_get_num_elements(instance->fragments, &num_fragments));

    total_fragments += num_fragments;
  }

  work->fragments_count = total_fragments;

  __FAILURE_HANDLE(host_malloc(&work->fragments, sizeof(LightTreeFragment) * work->fragments_count));

  uint32_t fragment_offset = 0;

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    LightTreeCacheInstance* instance = tree->cache.instances + instance_id;

    if (!instance->active)
      continue;

    uint32_t num_fragments;
    __FAILURE_HANDLE(array_get_num_elements(instance->fragments, &num_fragments));

    memcpy(work->fragments + fragment_offset, instance->fragments, sizeof(LightTreeFragment) * num_fragments);
    fragment_offset += num_fragments;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_free_data(LightTree* tree) {
  __CHECK_NULL_ARGUMENT(tree);

  if (tree->nodes_data) {
    __FAILURE_HANDLE(host_free(&tree->nodes_data));
  }

  if (tree->paths_data) {
    __FAILURE_HANDLE(host_free(&tree->paths_data));
  }

  if (tree->tri_handle_map_data) {
    __FAILURE_HANDLE(host_free(&tree->tri_handle_map_data));
  }

  if (tree->bvh_vertex_buffer_data) {
    __FAILURE_HANDLE(host_free(&tree->bvh_vertex_buffer_data));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_build(LightTree* tree, Device* device) {
  __CHECK_NULL_ARGUMENT(tree);

  // Only build if the cache is dirty.
  if (!tree->cache.is_dirty)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(_light_tree_free_data(tree));

  __FAILURE_HANDLE(_light_tree_handle_dirty_states(tree));

  LightTreeWork work;
  memset(&work, 0, sizeof(LightTreeWork));

  __FAILURE_HANDLE(_light_tree_collect_fragments(tree, &work));
  __FAILURE_HANDLE(_light_tree_build_binary_bvh(&work));
  __FAILURE_HANDLE(_light_tree_build_traversal_structure(&work));
  __FAILURE_HANDLE(_light_tree_collapse(&work));
  __FAILURE_HANDLE(_light_tree_finalize(tree, &work));
#ifdef LIGHT_TREE_DEBUG_OUTPUT
  _light_tree_debug_output(&work);
#endif /* LIGHT_TREE_DEBUG_OUTPUT */
  __FAILURE_HANDLE(_light_tree_clear_work(&work));

  tree->build_id++;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////

LuminaryResult light_tree_destroy(LightTree** tree) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(*tree);

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements((*tree)->cache.meshes, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    LightTreeCacheMesh* mesh = (*tree)->cache.meshes + mesh_id;

    if (mesh->materials) {
      __FAILURE_HANDLE(array_destroy(&mesh->materials));
    }

    if (mesh->material_triangles) {
      uint32_t num_materials;
      __FAILURE_HANDLE(array_get_num_elements(mesh->material_triangles, &num_materials));

      for (uint32_t material_slot_id = 0; material_slot_id < num_materials; material_slot_id++) {
        __FAILURE_HANDLE(array_destroy(&mesh->material_triangles[material_slot_id]));
      }

      __FAILURE_HANDLE(array_destroy(&mesh->material_triangles));
    }
  }

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements((*tree)->cache.instances, &num_instances));

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    LightTreeCacheInstance* instance = (*tree)->cache.instances + instance_id;

    if (instance->fragments) {
      __FAILURE_HANDLE(array_destroy(&instance->fragments));
    }

    if (instance->bvh_triangles) {
      __FAILURE_HANDLE(array_destroy(&instance->bvh_triangles));
    }
  }

  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.meshes));
  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.instances));
  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.materials));

  __FAILURE_HANDLE(_light_tree_free_data(*tree));

  __FAILURE_HANDLE(host_free(tree));

  return LUMINARY_SUCCESS;
}

#if 0
void lights_process(Scene* scene, int dmm_active) {
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
    const uint16_t tex_index      = material.luminance_tex;

    // Triangles with displacement can't be light sources.
    if (dmm_active && scene->materials[triangle.material_id].normal_tex)
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

  if (scene->triangle_lights_count > 0) {
    _lights_build_light_tree(scene);
  }

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
}
#endif
