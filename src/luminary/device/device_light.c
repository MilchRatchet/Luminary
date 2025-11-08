#include "device_light.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceb.h"
#include "device.h"
#include "device_packing.h"
#include "host_math.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "texture.h"
#include "utils.h"

// #define LIGHT_TREE_DEBUG_OUTPUT
// #define LIGHT_COMPUTE_VMF_DISTRIBUTIONS

// #define LIGHT_TREE_UNIFORM_WEIGHTING
// #define LIGHT_TREE_USE_BOUND_AS_VARIANCE

#define LIGHT_TREE_MAX_LEAF_TRIANGLE_COUNT (1)
#define LIGHT_TREE_BINARY_INDEX_NULL (0xFFFFFFFF)

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
  float left_power;
  float right_power;
  float left_area;
  float right_area;
  uint32_t path;
  uint32_t depth;
} typedef LightTreeBinaryNode;

struct LightTreeNode {
#ifdef LIGHT_COMPUTE_VMF_DISTRIBUTIONS
  vec3 left_vmf_dir;
  vec3 right_vmf_dir;
  float left_vmf_sharpness;
  float right_vmf_sharpness;
#endif /* LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
  vec3 left_mean;
  vec3 right_mean;
  float left_variance;
  float right_variance;
  float left_power;
  float right_power;
  uint32_t child_ptr;
  uint32_t light_ptr;
  uint32_t light_count;
} typedef LightTreeNode;

struct LightTreeChildNode {
  vec3 mean;
  float variance;
  float power;
  bool is_leaf;
} typedef LightTreeChildNode;

struct LightTreeWork {
  LightTreeFragment* fragments;
  uint32_t fragments_count;
  ARRAY LightTreeBinaryNode* binary_nodes;
  LightTreeNode* nodes;
  uint32_t nodes_count;
  ARRAY DeviceLightTreeRootHeader* root_output;
  ARRAY DeviceLightTreeNode* nodes_output;
} typedef LightTreeWork;

struct Bin {
  Vec128 high;
  Vec128 low;
  int32_t entry;
  int32_t exit;
  float power;
  uint32_t padding;
} typedef Bin;

#define OBJECT_SPLIT_BIN_COUNT_LOG (5)
#define OBJECT_SPLIT_BIN_COUNT (1 << OBJECT_SPLIT_BIN_COUNT_LOG)

// We need to bound the dimensions, the number must be large but still much smaller than FLT_MAX
#define MAX_VALUE 1e10f

#define FRAGMENT_ERROR_COMP (FLT_EPSILON * 16.0f)

#define _light_tree_swap(__internal_macro_a, __internal_macro_b, __internal_macro_type) \
  {                                                                                     \
    const __internal_macro_type __internal_macro_swap = __internal_macro_a;             \
    __internal_macro_a                                = __internal_macro_b;             \
    __internal_macro_b                                = __internal_macro_swap;          \
  }

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
      if (__macro_pos >= OBJECT_SPLIT_BIN_COUNT)                                                                                          \
        __macro_pos = OBJECT_SPLIT_BIN_COUNT - 1;                                                                                         \
                                                                                                                                          \
      __macro_in_bins[__macro_pos].entry++;                                                                                               \
      __macro_in_bins[__macro_pos].exit++;                                                                                                \
      __macro_in_bins[__macro_pos].power += __macro_in_fragments[__macro_i].power;                                                        \
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
    .power  = 0.0f};
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

      __DEBUG_ASSERT(fragments_count > 0);

      // Node has few enough triangles, finalize it as a leaf node.
      if (fragments_count == 1)
        continue;

      // Compute surface area of current node.

      Vec128 high_parent, low_parent;
      _light_tree_fit_bounds(fragments + fragments_ptr, fragments_count, &high_parent, &low_parent);

      const Vec128 diff             = vec128_set_w_to_0(vec128_sub(high_parent, low_parent));
      const float max_axis_interval = vec128_hmax(diff);

      Vec128 high, low;
      double optimal_cost = DBL_MAX;
      LightTreeSweepAxis axis;
      double optimal_splitting_plane;
      bool found_split = false;
      uint32_t optimal_split;
      float optimal_left_power, optimal_right_power;

      // For each axis, perform a greedy search for an optimal split.
      for (uint32_t a = 0; a < 3; a++) {
        double low_split;
        const double interval =
          _light_tree_construct_bins(bins, fragments + fragments_ptr, fragments_count, (LightTreeSweepAxis) a, &low_split);

        if (interval == 0.0)
          continue;

        const double interval_cost = max_axis_interval / interval;

        uint32_t left = 0;

        float left_power  = 0.0f;
        float right_power = 0.0f;

        for (uint32_t k = 0; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          right_power += bins[k].power;
        }

        Vec128 high_left  = vec128_set_1(-MAX_VALUE);
        Vec128 high_right = vec128_set_1(-MAX_VALUE);
        Vec128 low_left   = vec128_set_1(MAX_VALUE);
        Vec128 low_right  = vec128_set_1(MAX_VALUE);

        for (uint32_t k = 1; k < OBJECT_SPLIT_BIN_COUNT; k++) {
          _light_tree_update_bounds_of_bins(bins + k - 1, high_left, low_left);
          _light_tree_fit_bounds_of_bins(bins + k, OBJECT_SPLIT_BIN_COUNT - k, &high_right, &low_right);

          left_power += bins[k - 1].power;
          right_power -= bins[k - 1].power;

          const Vec128 diff_left  = vec128_sub(high_left, low_left);
          const Vec128 diff_right = vec128_sub(high_right, low_right);

          const float left_area  = vec128_box_area(vec128_set_w_to_0(diff_left));
          const float right_area = vec128_box_area(vec128_set_w_to_0(diff_right));

          const double total_cost = interval_cost * (left_power * left_area + right_power * right_area);

          left += bins[k - 1].entry;

          if (left == 0 || left == fragments_count)
            continue;

          if (total_cost < optimal_cost) {
            optimal_cost            = total_cost;
            optimal_split           = left;
            optimal_splitting_plane = low_split + k * interval;
            found_split             = true;
            axis                    = a;
            optimal_left_power      = left_power;
            optimal_right_power     = right_power;
          }
        }
      }

      if (found_split) {
        _light_tree_divide_middles_along_axis(optimal_splitting_plane, axis, fragments + fragments_ptr, fragments_count);
      }
      else {
        // We didn't find a split but we have too many triangles so we need to do a simply list split.
        optimal_split       = fragments_count / 2;
        optimal_left_power  = 0.0f;
        optimal_right_power = 0.0f;

        uint32_t frag_id = 0;

        for (; frag_id < optimal_split; frag_id++) {
          optimal_left_power += fragments[fragments_ptr + frag_id].power;
        }

        for (; frag_id < fragments_count; frag_id++) {
          optimal_right_power += fragments[fragments_ptr + frag_id].power;
        }
      }

      node.left_power  = optimal_left_power;
      node.right_power = optimal_right_power;

      _light_tree_fit_bounds(fragments + fragments_ptr, optimal_split, &high, &low);

      node.left_high.x = high.x;
      node.left_high.y = high.y;
      node.left_high.z = high.z;
      node.left_low.x  = low.x;
      node.left_low.y  = low.y;
      node.left_low.z  = low.z;
      node.left_area   = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z);

      __FAILURE_HANDLE(array_get_num_elements(nodes, &node.child_address));

      LightTreeBinaryNode node_left = {
        .triangle_count    = optimal_split,
        .triangles_address = fragments_ptr,
        .type              = LIGHT_TREE_NODE_TYPE_LEAF,
        .surface_area      = node.left_area,
        .self_high.x       = high.x,
        .self_high.y       = high.y,
        .self_high.z       = high.z,
        .self_low.x        = low.x,
        .self_low.y        = low.y,
        .self_low.z        = low.z,
        .path              = node.path | (1 << node.depth),
        .depth             = node.depth + 1,
      };

      __FAILURE_HANDLE(array_push(&nodes, &node_left));

      _light_tree_fit_bounds(fragments + fragments_ptr + optimal_split, node.triangle_count - optimal_split, &high, &low);

      node.right_high.x = high.x;
      node.right_high.y = high.y;
      node.right_high.z = high.z;
      node.right_low.x  = low.x;
      node.right_low.y  = low.y;
      node.right_low.z  = low.z;
      node.right_area   = (high.x - low.x) * (high.y - low.y) + (high.x - low.x) * (high.z - low.z) + (high.y - low.y) * (high.z - low.z);

      LightTreeBinaryNode node_right = {
        .triangle_count    = node.triangle_count - optimal_split,
        .triangles_address = fragments_ptr + optimal_split,
        .type              = LIGHT_TREE_NODE_TYPE_LEAF,
        .surface_area      = node.right_area,
        .self_high.x       = high.x,
        .self_high.y       = high.y,
        .self_high.z       = high.z,
        .self_low.x        = low.x,
        .self_low.y        = low.y,
        .self_low.z        = low.z,
        .path              = node.path,
        .depth             = node.depth + 1,
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

static LuminaryResult _lights_get_vmf_and_mean_and_variance(
  const LightTreeWork* work, const LightTreeBinaryNode node, const float parent_power, float* power, vec3* mean, float* variance) {
  // TODO: This could instead be done in a bottom up fashion like in the paper which would be much faster, however, this here might be more
  // accurate

  // There is a good chance that the power is numerically unstable, recompute it.
  if (*power < parent_power * 1e-5f) {
    float new_power = 0.0f;

    for (uint32_t i = 0; i < node.triangle_count; i++) {
      const LightTreeFragment frag = work->fragments[node.triangles_address + i];
      new_power += frag.power;
    }

    *power = new_power;
  }

  __DEBUG_ASSERT(*power > 0.0f);

#ifdef LIGHT_TREE_UNIFORM_WEIGHTING
  const float inverse_total_power = 1.0f / node.triangle_count;
#else
  const float inverse_total_power = 1.0f / *power;
#endif

#ifdef LIGHT_COMPUTE_VMF_DISTRIBUTIONS
  Vec128 average_direction = vec128_set_1(0.0f);
#endif /* LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
  Vec128 p = vec128_set_1(0.0f);
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    const LightTreeFragment frag = work->fragments[node.triangles_address + i];

#ifdef LIGHT_TREE_UNIFORM_WEIGHTING
    const float weight = inverse_total_power;
#else
    const float weight = frag.power * inverse_total_power;
#endif

    const Vec128 weight_vector = vec128_set_1(weight);

#ifdef LIGHT_COMPUTE_VMF_DISTRIBUTIONS
    average_direction = vec128_fmadd(frag.average_direction, weight_vector, average_direction);
#endif /* LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
    p = vec128_fmadd(frag.middle, weight_vector, p);
  }

  float spatial_variance = 0.0f;
  for (uint32_t i = 0; i < node.triangle_count; i++) {
    const LightTreeFragment frag = work->fragments[node.triangles_address + i];

#ifdef LIGHT_TREE_USE_BOUND_AS_VARIANCE
    const Vec128 diff0 = vec128_sub(frag.v0, p);
    spatial_variance   = fmaxf(spatial_variance, vec128_dot(diff0, diff0));

    const Vec128 diff1 = vec128_sub(frag.v1, p);
    spatial_variance   = fmaxf(spatial_variance, vec128_dot(diff1, diff1));

    const Vec128 diff2 = vec128_sub(frag.v2, p);
    spatial_variance   = fmaxf(spatial_variance, vec128_dot(diff2, diff2));
#else
    // Each triangle contributed 3 times to the spatial variance
#ifdef LIGHT_TREE_UNIFORM_WEIGHTING
    const float weight = (1.0f / 3.0f) * inverse_total_power;
#else
    const float weight = (1.0f / 3.0f) * frag.power * inverse_total_power;
#endif

    const Vec128 diff0 = vec128_sub(frag.v0, p);
    spatial_variance += weight * vec128_dot(diff0, diff0);

    const Vec128 diff1 = vec128_sub(frag.v1, p);
    spatial_variance += weight * vec128_dot(diff1, diff1);

    const Vec128 diff2 = vec128_sub(frag.v2, p);
    spatial_variance += weight * vec128_dot(diff2, diff2);
#endif
  }

  __DEBUG_ASSERT(isnan(p.x) == false);
  __DEBUG_ASSERT(isnan(p.y) == false);
  __DEBUG_ASSERT(isnan(p.z) == false);

#ifdef LIGHT_COMPUTE_VMF_DISTRIBUTIONS
  const float avg_dir_norm = vec128_norm2(average_direction);

  *vmf_direction =
    (vec3) {.x = average_direction.x / avg_dir_norm, .y = average_direction.y / avg_dir_norm, .z = average_direction.z / avg_dir_norm};
  *vmf_sharpness = (3.0f * avg_dir_norm - avg_dir_norm * avg_dir_norm * avg_dir_norm) / (1.0f - avg_dir_norm * avg_dir_norm);
#endif /* LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
  *mean     = (vec3) {.x = p.x, .y = p.y, .z = p.z};
  *variance = spatial_variance;

  return LUMINARY_SUCCESS;
}

// Set the reference point in each node to be the power weighted mean of the centers of the lights.
// Then, based on that reference point, compute the smallest distance to any light center.
// This is our spatial confidence that we use to clamp the distance with when evaluating the importance during traversal.
static LuminaryResult _light_tree_build_traversal_structure(LightTreeWork* work) {
  LightTreeNode* nodes;
  __FAILURE_HANDLE(host_malloc(&nodes, sizeof(LightTreeNode) * work->nodes_count));

  for (uint32_t i = 0; i < work->nodes_count; i++) {
    LightTreeBinaryNode binary_node = work->binary_nodes[i];
    LightTreeBinaryNode* children   = work->binary_nodes + binary_node.child_address;

    LightTreeNode node;

    node.left_power  = binary_node.left_power;
    node.right_power = binary_node.right_power;
    node.light_count = binary_node.triangle_count;
    node.light_ptr   = binary_node.triangles_address;

    __DEBUG_ASSERT(node.light_count > 0);

    switch (binary_node.type) {
      case LIGHT_TREE_NODE_TYPE_INTERNAL:
        const float parent_power = binary_node.left_power + binary_node.right_power;

        __DEBUG_ASSERT(parent_power > 0.0f);

        node.child_ptr = binary_node.child_address;

#ifdef LIGHT_COMPUTE_VMF_DISTRIBUTIONS
        __FAILURE_HANDLE(_lights_get_vmf_and_mean_and_variance(
          work, children[0], &node.left_power, &node.left_vmf_dir, &node.left_vmf_sharpness, &node.left_mean, &node.left_variance));
        __FAILURE_HANDLE(_lights_get_vmf_and_mean_and_variance(
          work, children[1], &node.right_power, &node.right_vmf_dir, &node.right_vmf_sharpness, &node.right_mean, &node.right_variance));
#else  /* LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
        __FAILURE_HANDLE(
          _lights_get_vmf_and_mean_and_variance(work, children[0], parent_power, &node.left_power, &node.left_mean, &node.left_variance));
        __FAILURE_HANDLE(_lights_get_vmf_and_mean_and_variance(
          work, children[1], parent_power, &node.right_power, &node.right_mean, &node.right_variance));
#endif /* !LIGHT_COMPUTE_VMF_DISTRIBUTIONS */
        break;
      case LIGHT_TREE_NODE_TYPE_LEAF:
        node.child_ptr = LIGHT_TREE_BINARY_INDEX_NULL;
        break;
      default:
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Encountered illegal node type!");
    }

    nodes[i] = node;
  }

  work->nodes = nodes;

  return LUMINARY_SUCCESS;
}

struct LightTreeCollapseWork {
  ARRAY uint32_t* binary_node_indices;
  ARRAY uint32_t* node_offset;
  ARRAY DeviceLightTreeRootHeader* root;
  ARRAY DeviceLightTreeNode* nodes;
  uint32_t num_node_jobs;
  uint64_t* node_paths;
  uint32_t* node_depths;
  uint32_t* new_fragments;
  uint64_t* fragment_paths;
  uint32_t triangles_ptr;
} typedef LightTreeCollapseWork;

static LuminaryResult _light_tree_subset_append(LightTreeCollapseWork* cwork, const LightTreeNode node, uint32_t* subset_index) {
  __CHECK_NULL_ARGUMENT(cwork);

  if (*subset_index == LIGHT_TREE_LIGHT_ID_NULL) {
    *subset_index = cwork->triangles_ptr;
  }

  cwork->new_fragments[cwork->triangles_ptr++] = node.light_ptr;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collapse_binary_node(
  LightTreeCollapseWork* cwork, const LightTreeNode base_node, const LightTreeNode* binary_nodes, LightTreeChildNode* children,
  uint32_t* child_binary_index, const uint32_t max_child_count, const bool is_root, uint32_t* light_ptr, uint32_t* child_count_out,
  uint32_t* leaf_count_out) {
  __CHECK_NULL_ARGUMENT(cwork);
  __CHECK_NULL_ARGUMENT(binary_nodes);
  __CHECK_NULL_ARGUMENT(children);
  __CHECK_NULL_ARGUMENT(child_binary_index);
  __CHECK_NULL_ARGUMENT(light_ptr);
  __CHECK_NULL_ARGUMENT(child_count_out);

  uint32_t child_count       = 0;
  bool children_require_work = false;

  // We must be able to split.
  __DEBUG_ASSERT(max_child_count > 1);

  if (base_node.light_count > 1) {
    LightTreeChildNode left_child;
    memset(&left_child, 0, sizeof(LightTreeChildNode));

    left_child.mean     = base_node.left_mean;
    left_child.variance = base_node.left_variance;
    left_child.power    = base_node.left_power;

    child_binary_index[child_count] = base_node.child_ptr;

    children[child_count++] = left_child;

    LightTreeChildNode right_child;
    memset(&right_child, 0, sizeof(LightTreeChildNode));

    right_child.mean     = base_node.right_mean;
    right_child.variance = base_node.right_variance;
    right_child.power    = base_node.right_power;

    child_binary_index[child_count] = base_node.child_ptr + 1;

    children[child_count++] = right_child;

    children_require_work = (child_count < max_child_count);
  }
  else {
    __DEBUG_ASSERT(is_root);

    // This case only happens if there is only 1 light in the scene
    LightTreeChildNode child;
    memset(&child, 0, sizeof(LightTreeChildNode));

    child.is_leaf = true;
    child.power   = 1.0f;

    child_binary_index[child_count] = 0;
    children[child_count++]         = child;
  }

  while (children_require_work) {
    children_require_work = false;

    float optimal_cost      = 0.0f;
    uint32_t selected_child = 0;

    for (uint32_t child_ptr = 0; child_ptr < max_child_count; child_ptr++) {
      const uint32_t binary_index_of_child = child_binary_index[child_ptr];

      if (binary_index_of_child == LIGHT_TREE_BINARY_INDEX_NULL)
        continue;

      const LightTreeNode binary_node = binary_nodes[binary_index_of_child];

      __DEBUG_ASSERT(binary_node.light_count > 0);

      // This is a leaf node
      if (binary_node.light_count == 1)
        continue;

      const float cost_this_node =
        (binary_node.left_power + binary_node.right_power) * (binary_node.left_variance + binary_node.right_variance);

      if (cost_this_node > optimal_cost) {
        optimal_cost          = cost_this_node;
        selected_child        = child_ptr;
        children_require_work = true;
      }
    }

    if (children_require_work == false)
      break;

    const LightTreeNode binary_node = binary_nodes[child_binary_index[selected_child]];

    __DEBUG_ASSERT(binary_node.light_count > 1);
    __DEBUG_ASSERT(binary_node.child_ptr != LIGHT_TREE_BINARY_INDEX_NULL);

    LightTreeChildNode left_child;
    memset(&left_child, 0, sizeof(LightTreeChildNode));

    left_child.mean     = binary_node.left_mean;
    left_child.variance = binary_node.left_variance;
    left_child.power    = binary_node.left_power;

    child_binary_index[selected_child] = binary_node.child_ptr;
    children[selected_child]           = left_child;

    LightTreeChildNode right_child;
    memset(&right_child, 0, sizeof(LightTreeChildNode));

    right_child.mean     = binary_node.right_mean;
    right_child.variance = binary_node.right_variance;
    right_child.power    = binary_node.right_power;

    uint32_t child_slot = 0;
    for (; child_slot < max_child_count; child_slot++) {
      if (child_binary_index[child_slot] == LIGHT_TREE_BINARY_INDEX_NULL)
        break;
    }

    child_binary_index[child_slot] = binary_node.child_ptr + 1;
    children[child_slot]           = right_child;

    child_count++;

    if (child_count == max_child_count)
      break;
  }

  if (child_count < max_child_count) {
    // The non-null children must come first
    for (uint32_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
      const uint32_t binary_index_of_child = child_binary_index[child_ptr];

      // Child is null node, search for a mis-positioned non-null node and swap
      if (binary_index_of_child == LIGHT_TREE_BINARY_INDEX_NULL) {
        uint32_t child_swap_ptr = child_count;
        for (; child_swap_ptr < max_child_count; child_swap_ptr++) {
          if (child_binary_index[child_swap_ptr] != LIGHT_TREE_BINARY_INDEX_NULL)
            break;
        }

        _light_tree_swap(children[child_ptr], children[child_swap_ptr], LightTreeChildNode);
        _light_tree_swap(child_binary_index[child_ptr], child_binary_index[child_swap_ptr], uint32_t);
      }
    }
  }

  uint32_t num_leaf_nodes = 0;

  // Mark all leaf nodes
  for (uint32_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
    const uint32_t binary_index_of_child = child_binary_index[child_ptr];

    __DEBUG_ASSERT(binary_index_of_child != LIGHT_TREE_BINARY_INDEX_NULL);

    const LightTreeNode binary_node = binary_nodes[binary_index_of_child];

    if (binary_node.light_count == 1) {
      __FAILURE_HANDLE(_light_tree_subset_append(cwork, binary_node, light_ptr));
      children[child_ptr].is_leaf   = true;
      child_binary_index[child_ptr] = LIGHT_TREE_BINARY_INDEX_NULL;
      num_leaf_nodes++;
    }
  }

  // Sort children so that leaf nodes come first
  for (uint32_t child_ptr = 0; child_ptr < num_leaf_nodes; child_ptr++) {
    const LightTreeChildNode child = children[child_ptr];

    // Child is not a leaf but is in the part of the list that contains leaves
    if (child.is_leaf == false) {
      // It is important that the order of the leaves is not modified, else we would have to apply this reordering to the fragments too
      uint32_t child_swap_ptr = child_ptr + 1;
      for (; child_swap_ptr < child_count; child_swap_ptr++) {
        if (children[child_swap_ptr].is_leaf)
          break;
      }

      _light_tree_swap(children[child_ptr], children[child_swap_ptr], LightTreeChildNode);
      _light_tree_swap(child_binary_index[child_ptr], child_binary_index[child_swap_ptr], uint32_t);
    }
  }

  *child_count_out = child_count;
  *leaf_count_out  = num_leaf_nodes;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collapse_root(LightTreeCollapseWork* cwork, const LightTreeNode* binary_nodes) {
  __CHECK_NULL_ARGUMENT(cwork);
  __CHECK_NULL_ARGUMENT(binary_nodes);

  const LightTreeNode binary_root = binary_nodes[0];

  LightTreeChildNode children[LIGHT_TREE_ROOT_MAX_CHILD_COUNT];
  uint32_t child_binary_index[LIGHT_TREE_ROOT_MAX_CHILD_COUNT];

  for (uint32_t i = 0; i < LIGHT_TREE_ROOT_MAX_CHILD_COUNT; i++) {
    child_binary_index[i] = LIGHT_TREE_BINARY_INDEX_NULL;
  }

  uint32_t light_ptr   = LIGHT_TREE_LIGHT_ID_NULL;
  uint32_t child_count = 0;
  uint32_t num_lights  = 0;

  __FAILURE_HANDLE(_light_tree_collapse_binary_node(
    cwork, binary_root, binary_nodes, children, child_binary_index, LIGHT_TREE_ROOT_MAX_CHILD_COUNT, true, &light_ptr, &child_count,
    &num_lights));

  vec3 min_mean      = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};
  vec3 max_mean      = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};
  float max_variance = 0.0f;
  float max_power    = 0.0f;

  for (uint32_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
    const vec3 mean = children[child_ptr].mean;

    min_mean.x = fminf(min_mean.x, mean.x);
    min_mean.y = fminf(min_mean.y, mean.y);
    min_mean.z = fminf(min_mean.z, mean.z);

    max_mean.x = fmaxf(max_mean.x, mean.x);
    max_mean.y = fmaxf(max_mean.y, mean.y);
    max_mean.z = fmaxf(max_mean.z, mean.z);

    max_variance = fmaxf(max_variance, children[child_ptr].variance);
    max_power    = fmaxf(max_power, children[child_ptr].power);
  }

  const float max_std_dev = sqrtf(max_variance);

  __DEBUG_ASSERT(child_count > 0);

  DeviceLightTreeRootHeader header;
  memset(&header, 0, sizeof(DeviceLightTreeRootHeader));

  header.x = device_pack_float(min_mean.x, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);
  header.y = device_pack_float(min_mean.y, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);
  header.z = device_pack_float(min_mean.z, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);

  min_mean.x = device_unpack_float(header.x);
  min_mean.y = device_unpack_float(header.y);
  min_mean.z = device_unpack_float(header.z);

  header.exp_x       = (int8_t) (max_mean.x != min_mean.x) ? ceilf(log2f((max_mean.x - min_mean.x) * 1.0f / 255.0f)) : 0;
  header.exp_y       = (int8_t) (max_mean.y != min_mean.y) ? ceilf(log2f((max_mean.y - min_mean.y) * 1.0f / 255.0f)) : 0;
  header.exp_z       = (int8_t) (max_mean.z != min_mean.z) ? ceilf(log2f((max_mean.z - min_mean.z) * 1.0f / 255.0f)) : 0;
  header.exp_std_dev = (int8_t) ceilf(log2f(max_std_dev * 1.0f / 255.0f));

  const float compression_x = 1.0f / exp2f(header.exp_x);
  const float compression_y = 1.0f / exp2f(header.exp_y);
  const float compression_z = 1.0f / exp2f(header.exp_z);
  const float compression_v = 1.0f / exp2f(header.exp_std_dev);

  header.num_sections        = (child_count + LIGHT_TREE_MAX_CHILDREN_PER_SECTION - 1) / LIGHT_TREE_MAX_CHILDREN_PER_SECTION;
  header.num_root_lights     = num_lights;
  header.power_normalization = device_pack_float(max_power, DEVICE_PACK_FLOAT_ROUNDING_MODE_CEIL);

#ifdef LIGHT_TREE_DEBUG_OUTPUT
  info_message("======= ROOT =======");
  info_message("Sections: %u", header.num_sections);
  info_message("Lights: %u", header.num_root_lights);
  info_message("Min: (%f, %f, %f)", min_mean.x, min_mean.y, min_mean.z);
  info_message("Max: (%f, %f, %f)", max_mean.x, max_mean.y, max_mean.z);
  info_message(
    "Exponents: %d %d %d => %f %f %f | %d => %f", header.exp_x, header.exp_y, header.exp_z, 1.0f / compression_x, 1.0f / compression_y,
    1.0f / compression_z, header.exp_std_dev, 1.0f / compression_v);
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

  __FAILURE_HANDLE(array_push(&cwork->root, &header));

  for (uint32_t section_id = 0; section_id < header.num_sections; section_id++) {
    DeviceLightTreeRootSection section;
    memset(&section, 0, sizeof(DeviceLightTreeRootSection));

    const uint32_t child_id_start = section_id * LIGHT_TREE_MAX_CHILDREN_PER_SECTION;
    const uint32_t child_id_end   = min(child_count, (section_id + 1) * LIGHT_TREE_MAX_CHILDREN_PER_SECTION);

    for (uint32_t child_id = child_id_start; child_id < child_id_end; child_id++) {
      const LightTreeChildNode child_node = children[child_id];

      uint64_t child_rel_mean_x  = (uint64_t) floorf((child_node.mean.x - min_mean.x) * compression_x + 0.5f);
      uint64_t child_rel_mean_y  = (uint64_t) floorf((child_node.mean.y - min_mean.y) * compression_y + 0.5f);
      uint64_t child_rel_mean_z  = (uint64_t) floorf((child_node.mean.z - min_mean.z) * compression_z + 0.5f);
      uint64_t child_rel_std_dev = (uint64_t) (sqrtf(child_node.variance) * compression_v + 0.5f);
      uint64_t child_rel_power   = (uint64_t) floorf(0xFFFF * child_node.power / max_power + 0.5f);

      __DEBUG_ASSERT((child_rel_mean_x & 0xFF) == child_rel_mean_x);
      __DEBUG_ASSERT((child_rel_mean_y & 0xFF) == child_rel_mean_y);
      __DEBUG_ASSERT((child_rel_mean_z & 0xFF) == child_rel_mean_z);
      __DEBUG_ASSERT((child_rel_std_dev & 0xFF) == child_rel_std_dev);
      __DEBUG_ASSERT((child_rel_power & 0xFFFF) == child_rel_power);

      // Power may not be zero as zero implies NULL node and a node with 0 power cannot be sampled.
      child_rel_std_dev = max(child_rel_std_dev, 1);
      child_rel_power   = max(child_rel_power, 1);

#ifdef LIGHT_TREE_DEBUG_OUTPUT
      info_message(
        "[%u] %llX %llX %llX %llX %llX %s", child_id, child_rel_mean_x, child_rel_mean_y, child_rel_mean_z, child_rel_power,
        child_rel_std_dev, child_node.is_leaf ? "LEAF" : "");

      {
        // Check error of compressed mean

        const float decompression_x = exp2f(header.exp_x);
        const float decompression_y = exp2f(header.exp_y);
        const float decompression_z = exp2f(header.exp_z);

        const float decompressed_mean_x = child_rel_mean_x * decompression_x + min_mean.x;
        const float decompressed_mean_y = child_rel_mean_y * decompression_y + min_mean.y;
        const float decompressed_mean_z = child_rel_mean_z * decompression_z + min_mean.z;

        const float diff_x = decompressed_mean_x - child_node.mean.x;
        const float diff_y = decompressed_mean_y - child_node.mean.y;
        const float diff_z = decompressed_mean_z - child_node.mean.z;

        const float error = sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

        info_message(
          "    (%f, %f, %f) => (%f, %f, %f) [Err: %f]", child_node.mean.x, child_node.mean.y, child_node.mean.z, decompressed_mean_x,
          decompressed_mean_y, decompressed_mean_z, error);
      }

      {
        // Check error of standard deviation
        const float decompression_v      = exp2f(header.exp_std_dev);
        const float decompressed_std_dev = child_rel_std_dev * decompression_v;

        info_message(
          "    %f => %f [Err: %f]", sqrtf(child_node.variance), decompressed_std_dev, sqrtf(child_node.variance) - decompressed_std_dev);
      }

      {
        // Check error of power
        const float decompressed_power = child_rel_power * (1.0f / 0xFFFF) * max_power;

        info_message("    %f => %f [Err: %f]", child_node.power, decompressed_power, child_node.power - decompressed_power);
      }
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

      const uint32_t section_child_id = child_id - section_id * LIGHT_TREE_MAX_CHILDREN_PER_SECTION;

      section.rel_mean_x[section_child_id]  = child_rel_mean_x;
      section.rel_mean_y[section_child_id]  = child_rel_mean_y;
      section.rel_mean_z[section_child_id]  = child_rel_mean_z;
      section.rel_std_dev[section_child_id] = child_rel_std_dev;
      section.rel_power[section_child_id]   = child_rel_power;
    }

    for (uint32_t section_part = 0; section_part < LIGHT_TREE_NODE_SECTION_REL_SIZE; section_part++) {
      __FAILURE_HANDLE(array_push(&cwork->root, ((DeviceLightTreeRootHeader*) &section) + section_part));
    }
  }

  // Prepare the next nodes to be constructed from the respective binary nodes.
  for (uint32_t i = 0; i < child_count; i++) {
    // There is nothing left to construct if this is a leaf.
    if (child_binary_index[i] == LIGHT_TREE_BINARY_INDEX_NULL)
      continue;

    __FAILURE_HANDLE(array_push(&cwork->binary_node_indices, &child_binary_index[i]));
    cwork->num_node_jobs++;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collapse_nodes(LightTreeCollapseWork* cwork, const LightTreeNode* binary_nodes) {
  __CHECK_NULL_ARGUMENT(cwork);
  __CHECK_NULL_ARGUMENT(binary_nodes);

  for (uint32_t job_id = 0; job_id < cwork->num_node_jobs; job_id++) {
    const uint32_t binary_index = cwork->binary_node_indices[job_id];

    __DEBUG_ASSERT(binary_index != LIGHT_TREE_BINARY_INDEX_NULL);

    LightTreeChildNode children[LIGHT_TREE_CHILDREN_PER_NODE];
    uint32_t child_binary_index[LIGHT_TREE_CHILDREN_PER_NODE];

    for (uint32_t i = 0; i < LIGHT_TREE_CHILDREN_PER_NODE; i++) {
      child_binary_index[i] = LIGHT_TREE_BINARY_INDEX_NULL;
    }

    uint32_t light_ptr   = LIGHT_TREE_LIGHT_ID_NULL;
    uint32_t child_count = 0;
    uint32_t num_lights  = 0;

    __FAILURE_HANDLE(_light_tree_collapse_binary_node(
      cwork, binary_nodes[binary_index], binary_nodes, children, child_binary_index, LIGHT_TREE_CHILDREN_PER_NODE, false, &light_ptr,
      &child_count, &num_lights));

    vec3 min_mean      = {.x = MAX_VALUE, .y = MAX_VALUE, .z = MAX_VALUE};
    vec3 max_mean      = {.x = -MAX_VALUE, .y = -MAX_VALUE, .z = -MAX_VALUE};
    float max_variance = 0.0f;
    float max_power    = 0.0f;

    for (uint32_t child_ptr = 0; child_ptr < child_count; child_ptr++) {
      const vec3 mean = children[child_ptr].mean;

      min_mean.x = fminf(min_mean.x, mean.x);
      min_mean.y = fminf(min_mean.y, mean.y);
      min_mean.z = fminf(min_mean.z, mean.z);

      max_mean.x = fmaxf(max_mean.x, mean.x);
      max_mean.y = fmaxf(max_mean.y, mean.y);
      max_mean.z = fmaxf(max_mean.z, mean.z);

      max_variance = fmaxf(max_variance, children[child_ptr].variance);
      max_power    = fmaxf(max_power, children[child_ptr].power);
    }

    const float max_std_dev = sqrtf(max_variance);

    __DEBUG_ASSERT(child_count > 0);

    DeviceLightTreeNode node;
    memset(&node, 0, sizeof(DeviceLightTreeNode));

    node.child_ptr  = cwork->num_node_jobs;
    node.light_ptr  = light_ptr;
    node.num_lights = num_lights;

    node.x = device_pack_float(min_mean.x, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);
    node.y = device_pack_float(min_mean.y, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);
    node.z = device_pack_float(min_mean.z, DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR);

    min_mean.x = device_unpack_float(node.x);
    min_mean.y = device_unpack_float(node.y);
    min_mean.z = device_unpack_float(node.z);

    node.exp_x       = (int8_t) (max_mean.x != min_mean.x) ? ceilf(log2f((max_mean.x - min_mean.x) * 1.0f / 255.0f)) : 0;
    node.exp_y       = (int8_t) (max_mean.y != min_mean.y) ? ceilf(log2f((max_mean.y - min_mean.y) * 1.0f / 255.0f)) : 0;
    node.exp_z       = (int8_t) (max_mean.z != min_mean.z) ? ceilf(log2f((max_mean.z - min_mean.z) * 1.0f / 255.0f)) : 0;
    node.exp_std_dev = (int8_t) ceilf(log2f(max_std_dev * 1.0f / 255.0f));

    const float compression_x = 1.0f / exp2f(node.exp_x);
    const float compression_y = 1.0f / exp2f(node.exp_y);
    const float compression_z = 1.0f / exp2f(node.exp_z);
    const float compression_v = 1.0f / exp2f(node.exp_std_dev);

    for (uint32_t child_id = 0; child_id < child_count; child_id++) {
      const LightTreeChildNode child_node = children[child_id];

      uint64_t child_rel_mean_x  = (uint64_t) floorf((child_node.mean.x - min_mean.x) * compression_x + 0.5f);
      uint64_t child_rel_mean_y  = (uint64_t) floorf((child_node.mean.y - min_mean.y) * compression_y + 0.5f);
      uint64_t child_rel_mean_z  = (uint64_t) floorf((child_node.mean.z - min_mean.z) * compression_z + 0.5f);
      uint64_t child_rel_std_dev = (uint64_t) (sqrtf(child_node.variance) * compression_v + 0.5f);
      uint64_t child_rel_power   = (uint64_t) floorf(0xFF * child_node.power / max_power + 0.5f);

      // Power may not be zero as zero implies NULL node and a node with 0 power cannot be sampled.
      child_rel_std_dev = max(child_rel_std_dev, 1);
      child_rel_power   = max(child_rel_power, 1);

      __DEBUG_ASSERT((child_rel_mean_x & 0xFF) == child_rel_mean_x);
      __DEBUG_ASSERT((child_rel_mean_y & 0xFF) == child_rel_mean_y);
      __DEBUG_ASSERT((child_rel_mean_z & 0xFF) == child_rel_mean_z);
      __DEBUG_ASSERT((child_rel_std_dev & 0xFF) == child_rel_std_dev);
      __DEBUG_ASSERT((child_rel_power & 0xFF) == child_rel_power);

      node.rel_mean_x[child_id]  = child_rel_mean_x;
      node.rel_mean_y[child_id]  = child_rel_mean_y;
      node.rel_mean_z[child_id]  = child_rel_mean_z;
      node.rel_std_dev[child_id] = child_rel_std_dev;
      node.rel_power[child_id]   = child_rel_power;
    }

#ifdef LIGHT_TREE_DEBUG_OUTPUT
    info_message("======= Node:0x%08X =======", job_id);
    info_message("Meta: 0x%08X (%u) 0x%08X", node.child_ptr, child_count, node.light_ptr);
    info_message("Min: (%f, %f, %f)", min_mean.x, min_mean.y, min_mean.z);
    info_message("Max: (%f, %f, %f)", max_mean.x, max_mean.y, max_mean.z);
    info_message(
      "Exponents: %d %d %d => %f %f %f | %d => %f", node.exp_x, node.exp_y, node.exp_z, 1.0f / compression_x, 1.0f / compression_y,
      1.0f / compression_z, node.exp_std_dev, 1.0f / compression_v);
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

    __FAILURE_HANDLE(array_push(&cwork->nodes, &node));

    // Prepare the next nodes to be constructed from the respective binary nodes.
    for (uint32_t i = 0; i < child_count; i++) {
      // There is nothing left to construct if this is a leaf.
      if (child_binary_index[i] == LIGHT_TREE_BINARY_INDEX_NULL)
        continue;

      __FAILURE_HANDLE(array_push(&cwork->binary_node_indices, &child_binary_index[i]));
      cwork->num_node_jobs++;
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_collapse(LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  if (work->nodes_count == 0) {
    __FAILURE_HANDLE(array_create(&work->root_output, sizeof(DeviceLightTreeRootHeader), 0));
    __FAILURE_HANDLE(array_create(&work->nodes_output, sizeof(DeviceLightTreeNode), 0));

    return LUMINARY_SUCCESS;
  }

  const uint32_t fragments_count = work->fragments_count;

  LightTreeNode* binary_nodes       = work->nodes;
  const uint32_t binary_nodes_count = work->nodes_count;

  uint32_t node_count = binary_nodes_count;

  LightTreeCollapseWork cwork;
  memset(&cwork, 0, sizeof(LightTreeCollapseWork));

  __FAILURE_HANDLE(array_create(&cwork.binary_node_indices, sizeof(uint32_t), node_count));
  __FAILURE_HANDLE(array_create(&cwork.node_offset, sizeof(uint32_t), node_count));
  __FAILURE_HANDLE(
    array_create(&cwork.root, sizeof(DeviceLightTreeRootHeader), 1 + LIGHT_TREE_ROOT_MAX_CHILD_COUNT * LIGHT_TREE_NODE_SECTION_REL_SIZE));
  __FAILURE_HANDLE(array_create(&cwork.nodes, sizeof(DeviceLightTreeNode), node_count));
  __FAILURE_HANDLE(host_malloc(&cwork.node_paths, sizeof(uint64_t) * node_count));
  __FAILURE_HANDLE(host_malloc(&cwork.node_depths, sizeof(uint32_t) * node_count));
  __FAILURE_HANDLE(host_malloc(&cwork.new_fragments, sizeof(uint32_t) * fragments_count));
  __FAILURE_HANDLE(host_malloc(&cwork.fragment_paths, sizeof(uint64_t) * fragments_count));

  memset(cwork.new_fragments, 0xFF, sizeof(uint32_t) * fragments_count);
  memset(cwork.fragment_paths, 0xFF, sizeof(uint64_t) * fragments_count);

  __FAILURE_HANDLE(_light_tree_collapse_root(&cwork, binary_nodes));
  __FAILURE_HANDLE(_light_tree_collapse_nodes(&cwork, binary_nodes));

  __DEBUG_ASSERT(cwork.triangles_ptr == fragments_count);

// Check
#ifdef LIGHT_TREE_DEBUG_OUTPUT
  for (uint32_t i = 0; i < fragments_count; i++) {
    if (cwork.new_fragments[i] == 0xFFFFFFFF) {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Fatal error during light tree compression. Light was lost.");
    }
  }
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

  LightTreeFragment* fragments_swap;
  __FAILURE_HANDLE(host_malloc(&fragments_swap, sizeof(LightTreeFragment) * fragments_count));

  memcpy(fragments_swap, work->fragments, sizeof(LightTreeFragment) * fragments_count);

  for (uint32_t i = 0; i < fragments_count; i++) {
    work->fragments[i] = fragments_swap[cwork.new_fragments[i]];
  }

  __FAILURE_HANDLE(host_free(&fragments_swap));

  __FAILURE_HANDLE(host_free(&cwork.node_paths));
  __FAILURE_HANDLE(host_free(&cwork.node_depths));
  __FAILURE_HANDLE(host_free(&cwork.new_fragments));
  __FAILURE_HANDLE(host_free(&cwork.fragment_paths));
  __FAILURE_HANDLE(array_destroy(&cwork.binary_node_indices));
  __FAILURE_HANDLE(array_destroy(&cwork.node_offset));

  work->root_output  = cwork.root;
  work->nodes_output = cwork.nodes;

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

    const LightTreeCacheMesh* mesh         = tree->cache.meshes + instance->mesh_id;
    const LightTreeCacheTriangle* triangle = mesh->material_triangles[frag.material_slot_id] + frag.material_tri_id;

    const TriangleHandle handle = (TriangleHandle) {.instance_id = frag.instance_id, .tri_id = triangle->tri_id};

    tri_handle_map[id] = handle;
    bvh_triangles[id]  = instance->bvh_triangles[frag.instance_cache_tri_id];
  }

  // TODO: This is inefficient, I take the array data and turn it into generic memory.
  uint32_t num_roots_as_16bytes;
  __FAILURE_HANDLE(array_get_num_elements(work->root_output, &num_roots_as_16bytes));

  DeviceLightTreeRootHeader* root_data;
  __FAILURE_HANDLE(host_malloc(&root_data, sizeof(DeviceLightTreeRootHeader) * num_roots_as_16bytes));

  memcpy(root_data, work->root_output, sizeof(DeviceLightTreeRootHeader) * num_roots_as_16bytes);

  uint32_t num_nodes_as_16bytes;
  __FAILURE_HANDLE(array_get_num_elements(work->nodes_output, &num_nodes_as_16bytes));

  DeviceLightTreeNode* nodes_data;
  __FAILURE_HANDLE(host_malloc(&nodes_data, sizeof(DeviceLightTreeNode) * num_nodes_as_16bytes));

  memcpy(nodes_data, work->nodes_output, sizeof(DeviceLightTreeNode) * num_nodes_as_16bytes);

  ////////////////////////////////////////////////////////////////////
  // Assign light tree data
  ////////////////////////////////////////////////////////////////////

  tree->root_size = sizeof(DeviceLightTreeRootHeader) * num_roots_as_16bytes;
  tree->root_data = (void*) root_data;

  tree->nodes_size = sizeof(DeviceLightTreeNode) * num_nodes_as_16bytes;
  tree->nodes_data = (void*) nodes_data;

  tree->tri_handle_map_size = sizeof(TriangleHandle) * work->fragments_count;
  tree->tri_handle_map_data = (void*) tri_handle_map;

  tree->bvh_vertex_buffer_size = sizeof(LightTreeBVHTriangle) * work->fragments_count;
  tree->bvh_vertex_buffer_data = (void*) bvh_triangles;

  tree->light_count = work->fragments_count;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_clear_work(LightTreeWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  __FAILURE_HANDLE(host_free(&work->fragments));
  __FAILURE_HANDLE(array_destroy(&work->binary_nodes));
  __FAILURE_HANDLE(host_free(&work->nodes));
  __FAILURE_HANDLE(array_destroy(&work->root_output));
  __FAILURE_HANDLE(array_destroy(&work->nodes_output));

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

  *vertex_offset = v_offset;
}

static LuminaryResult _light_tree_debug_output_export_device_node(
  FILE* obj_file, FILE* mtl_file, LightTreeWork* work, uint32_t* node_offset, uint32_t* vertex_offset) {
  __CHECK_NULL_ARGUMENT(obj_file);
  __CHECK_NULL_ARGUMENT(mtl_file);
  __CHECK_NULL_ARGUMENT(work);
  __CHECK_NULL_ARGUMENT(node_offset);
  __CHECK_NULL_ARGUMENT(vertex_offset);
#if 0
  const uint32_t offset = *node_offset;

  const DeviceLightTreeNodeHeader header = work->nodes8_packed[offset];

  const uint32_t child_ptr = ((uint32_t) header.child_and_light_ptr[0]) | (((uint32_t) header.child_and_light_ptr[1] & 0x00FF) << 16);
  const uint32_t light_ptr = ((uint32_t) header.child_and_light_ptr[2]) | (((uint32_t) header.child_and_light_ptr[1] & 0xFF00) << 8);

  char* buffer;
  __FAILURE_HANDLE(host_malloc(&buffer, 1024 * 1024));

  int buffer_offset = 0;

  buffer_offset += sprintf(buffer + buffer_offset, "o Node_0x%08X\n", offset);
  buffer_offset += sprintf(buffer + buffer_offset, "usemtl NodeMTL_0x%08X\n", offset);

  uint32_t section_offset = offset + 1;

  if (child_ptr != 0) {
    vec3 base;
    base.x = device_unpack_float(header.x);
    base.y = device_unpack_float(header.y);
    base.z = device_unpack_float(header.z);

    vec3 exp;
    exp.x = exp2f(header.exp_x);
    exp.y = exp2f(header.exp_y);
    exp.z = exp2f(header.exp_z);

    float exp_v;
    exp_v = exp2f(header.exp_variance);

    bool has_next = true;

    while (has_next) {
      const DeviceLightTreeNodeSection section = *((const DeviceLightTreeNodeSection*) (work->nodes8_packed + section_offset));
      section_offset += LIGHT_TREE_NODE_SECTION_REL_SIZE;

      has_next = (section.meta & LIGHT_TREE_META_HAS_NEXT) != 0;

      for (uint32_t rel_child_id = 0; rel_child_id < 3; rel_child_id++) {
        if (section.rel_power[rel_child_id] == 0)
          continue;

        const float variance = section.rel_variance[rel_child_id] * exp_v;

        vec3 mean;
        mean.x = section.rel_mean_x[rel_child_id] * exp.x + base.x;
        mean.y = section.rel_mean_y[rel_child_id] * exp.y + base.y;
        mean.z = section.rel_mean_z[rel_child_id] * exp.z + base.z;

        vec3 low;
        low.x = mean.x - 2.0f * variance;
        low.y = mean.y - 2.0f * variance;
        low.z = mean.z - 2.0f * variance;

        vec3 high;
        high.x = mean.x + 2.0f * variance;
        high.y = mean.y + 2.0f * variance;
        high.z = mean.z + 2.0f * variance;

        uint32_t v_offset = *vertex_offset;

        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", low.x, low.y, low.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", low.x, low.y, high.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", low.x, high.y, low.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", low.x, high.y, high.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", high.x, low.y, low.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", high.x, low.y, high.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", high.x, high.y, low.z);
        buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", high.x, high.y, high.z);

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

        v_offset += 8;

        *vertex_offset = v_offset;
      }

      __DEBUG_ASSERT(buffer_offset < 1024 * 1024);
      fwrite(buffer, buffer_offset, 1, obj_file);
      buffer_offset = 0;
    }

    const uint32_t num_sections        = section_offset - offset - 1;
    const uint32_t strided_num_section = ((num_sections + LIGHT_TREE_CHILD_OFFSET_STRIDE - 1) >> LIGHT_TREE_CHILD_OFFSET_STRIDE_LOG)
                                         << LIGHT_TREE_CHILD_OFFSET_STRIDE_LOG;

    section_offset = offset + 1 + strided_num_section;
  }

  *node_offset = section_offset;

#if 0
  if (light_ptr != LIGHT_TREE_LIGHT_ID_NULL) {
    DeviceLightSubset subset = work->subsets[light_ptr];

    for (uint32_t light_id = subset.index; light_id < subset.index + subset.count; light_id++) {
      LightTreeFragment fragment = work->fragments[light_id];

      uint32_t v_offset = *vertex_offset;

      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", fragment.v0.x, fragment.v0.y, fragment.v0.z);
      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", fragment.v1.x, fragment.v1.y, fragment.v1.z);
      buffer_offset += sprintf(buffer + buffer_offset, "v %f %f %f\n", fragment.v2.x, fragment.v2.y, fragment.v2.z);

      buffer_offset += sprintf(buffer + buffer_offset, "f %u %u %u\n", v_offset + 0, v_offset + 1, v_offset + 2);

      v_offset += 3;

      *vertex_offset = v_offset;
    }

    __DEBUG_ASSERT(buffer_offset < 1024 * 1024);
    fwrite(buffer, buffer_offset, 1, obj_file);
    buffer_offset = 0;
  }
#endif

  buffer_offset += sprintf(buffer + buffer_offset, "newmtl NodeMTL_0x%08X\n", offset);
  buffer_offset += sprintf(
    buffer + buffer_offset, "Kd %f %f %f\n", (offset & 0b100) ? 1.0f : 0.0f, (offset & 0b10) ? 1.0f : 0.0f, (offset & 0b1) ? 1.0f : 0.0f);
  buffer_offset += sprintf(buffer + buffer_offset, "d %f\n", 0.1f);

  __DEBUG_ASSERT(buffer_offset < 1024 * 1024);
  fwrite(buffer, buffer_offset, 1, mtl_file);

  __FAILURE_HANDLE(host_free(&buffer));
#endif
  *node_offset = 0xFFFFFFFF;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_debug_output(LightTreeWork* work) {
  FILE* obj_file = fopen("LuminaryLightTree.obj", "wb");

  if (!obj_file) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file LuminaryLightTree.obj.");
  }

  FILE* mtl_file = fopen("LuminaryLightTree.mtl", "wb");

  if (!mtl_file) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file LuminaryLightTree.mtl.");
  }

  fwrite("LuminaryLightTree.mtl\n", 22, 1, mtl_file);

  uint32_t vertex_offset = 1;

  for (uint32_t i = 1; i < work->nodes_count; i++) {
    _light_tree_debug_output_export_binary_node(obj_file, mtl_file, work, i, &vertex_offset);
  }

  fclose(obj_file);
  fclose(mtl_file);

  obj_file = fopen("LuminaryDeviceLightTree.obj", "wb");

  if (!obj_file) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file LuminaryDeviceLightTree.obj.");
  }

  mtl_file = fopen("LuminaryDeviceLightTree.mtl", "wb");

  if (!mtl_file) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file LuminaryDeviceLightTree.mtl.");
  }

  uint32_t num_nodes_as_16bytes;
  __FAILURE_HANDLE(array_get_num_elements(work->nodes_output, &num_nodes_as_16bytes));

  vertex_offset        = 1;
  uint32_t node_offset = 0;

  while (node_offset < num_nodes_as_16bytes) {
    __FAILURE_HANDLE(_light_tree_debug_output_export_device_node(obj_file, mtl_file, work, &node_offset, &vertex_offset))
  }

  fclose(obj_file);
  fclose(mtl_file);

  return LUMINARY_SUCCESS;
}
#endif /* LIGHT_TREE_DEBUG_OUTPUT */

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

  ////////////////////////////////////////////////////////////////////
  // Initialize integrator
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(array_create(&(*tree)->integrator.tasks, sizeof(LightTreeIntegratorTask), 1024));

  ////////////////////////////////////////////////////////////////////
  // Optix BVH
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(optix_bvh_create(&(*tree)->optix_bvh));

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
    cache_triangle.tri_id            = tri_id;
    cache_triangle.vertex            = vertex;
    cache_triangle.vertex1           = vec128_add(vertex, edge1);
    cache_triangle.vertex2           = vec128_add(vertex, edge2);
    cache_triangle.cross             = vec128_cross(edge1, edge2);
    cache_triangle.average_intensity = 1.0f;

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

  const Quaternion rotation = rotation_euler_angles_to_quaternion(instance->rotation);

  if (memcmp(&cache->rotation, &rotation, sizeof(Quaternion))) {
    cache->rotation   = rotation;
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

  bool has_emission   = false;
  *meshes_need_update = false;

  float intensity = 0.0f;

  if (material->emission_active) {
    const bool has_textured_emission = material->luminance_tex != TEXTURE_NONE;
    if (cache->has_textured_emission != has_textured_emission) {
      cache->has_textured_emission = has_textured_emission;
      cache->is_dirty              = true;
      cache->needs_reintegration   = true;
    }

    if (has_textured_emission) {
      // TODO: Check if triangle_has_textured_emission && (emission texture hash are not equal) then reintegrate.
      intensity = material->emission_scale;
    }
    else {
      intensity = fmaxf(material->emission.r, fmaxf(material->emission.g, material->emission.b));
    }
  }

  if (cache->constant_emission_intensity != intensity) {
    cache->constant_emission_intensity = intensity;
    cache->is_dirty                    = true;
  }

  if (intensity > 0.0f) {
    has_emission = true;
  }

  if (cache->has_emission != has_emission) {
    cache->has_emission = has_emission;
    cache->is_dirty     = true;
    *meshes_need_update = true;
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

static LuminaryResult _light_tree_queue_texture_integrations(LightTree* tree, const uint32_t mesh_id) {
  __CHECK_NULL_ARGUMENT(tree);

  if (mesh_id == MESH_ID_INVALID)
    return LUMINARY_SUCCESS;

  const LightTreeCacheMesh* mesh = tree->cache.meshes + mesh_id;

  if (!mesh->has_emission)
    return LUMINARY_SUCCESS;

  uint32_t num_materials;
  __FAILURE_HANDLE(array_get_num_elements(mesh->materials, &num_materials));

  uint32_t num_cached_materials;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.materials, &num_cached_materials));

  for (uint32_t material_slot_id = 0; material_slot_id < num_materials; material_slot_id++) {
    const uint16_t material_id = mesh->materials[material_slot_id];

    // Material is not cached, consider it non existent.
    if (material_id >= num_cached_materials)
      continue;

    const LightTreeCacheMaterial* material = tree->cache.materials + material_id;

    if (material->has_emission == false || material->needs_reintegration == false)
      continue;

    const ARRAY LightTreeCacheTriangle* material_triangles = mesh->material_triangles[material_slot_id];

    uint32_t num_material_triangles;
    __FAILURE_HANDLE(array_get_num_elements(material_triangles, &num_material_triangles));

    for (uint32_t tri_id = 0; tri_id < num_material_triangles; tri_id++) {
      const LightTreeCacheTriangle* triangle = material_triangles + tri_id;

      LightTreeIntegratorTask task;
      task.mesh_id              = mesh_id;
      task.material_slot_id     = material_slot_id;
      task.material_slot_tri_id = tri_id;
      task.triangle_id          = triangle->tri_id;

      __FAILURE_HANDLE(array_push(&tree->integrator.tasks, &task));
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_integrate(LightTree* tree, Device* device) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(device);

  uint32_t num_tasks;
  __FAILURE_HANDLE(array_get_num_elements(tree->integrator.tasks, &num_tasks));

  if (num_tasks == 0)
    return LUMINARY_SUCCESS;

  // TODO: Optimize, this will only rarely run so it does not really matter for now

  if (num_tasks > tree->integrator.allocated_tasks) {
    __FAILURE_HANDLE(light_tree_unload_integrator(tree));

    __FAILURE_HANDLE(host_malloc(&tree->integrator.mesh_ids, sizeof(uint32_t) * num_tasks));
    __FAILURE_HANDLE(host_malloc(&tree->integrator.triangle_ids, sizeof(uint32_t) * num_tasks));
    __FAILURE_HANDLE(host_malloc(&tree->integrator.intensities, sizeof(float) * num_tasks));

    __FAILURE_HANDLE(device_malloc(&tree->integrator.device_mesh_ids, sizeof(uint32_t) * num_tasks));
    __FAILURE_HANDLE(device_malloc(&tree->integrator.device_triangle_ids, sizeof(uint32_t) * num_tasks));
    __FAILURE_HANDLE(device_malloc(&tree->integrator.device_intensities, sizeof(float) * num_tasks));

    tree->integrator.allocated_tasks = num_tasks;
  }

  for (uint32_t task_id = 0; task_id < num_tasks; task_id++) {
    LightTreeIntegratorTask* task = tree->integrator.tasks + task_id;

    tree->integrator.mesh_ids[task_id]     = task->mesh_id;
    tree->integrator.triangle_ids[task_id] = task->triangle_id;
  }

  __FAILURE_HANDLE(
    device_upload(tree->integrator.device_mesh_ids, tree->integrator.mesh_ids, 0, sizeof(uint32_t) * num_tasks, device->stream_main));
  __FAILURE_HANDLE(device_upload(
    tree->integrator.device_triangle_ids, tree->integrator.triangle_ids, 0, sizeof(uint32_t) * num_tasks, device->stream_main));

  KernelArgsLightComputeIntensity args;
  args.mesh_ids        = DEVICE_PTR(tree->integrator.device_mesh_ids);
  args.triangle_ids    = DEVICE_PTR(tree->integrator.device_triangle_ids);
  args.dst_intensities = DEVICE_PTR(tree->integrator.device_intensities);
  args.lights_count    = num_tasks;

  // Every thread handles two microtriangles. Every warp handles 1 light.
  const uint32_t num_blocks = (num_tasks * (LIGHT_NUM_MICROTRIANGLES >> 1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  __FAILURE_HANDLE(kernel_execute_custom(
    device->cuda_kernels[CUDA_KERNEL_TYPE_LIGHT_COMPUTE_INTENSITY], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1, &args, device->stream_main));

  __FAILURE_HANDLE(
    device_download(tree->integrator.intensities, tree->integrator.device_intensities, 0, sizeof(float) * num_tasks, device->stream_main));

  for (uint32_t task_id = 0; task_id < num_tasks; task_id++) {
    LightTreeIntegratorTask* task = tree->integrator.tasks + task_id;

    LightTreeCacheMesh* mesh = tree->cache.meshes + task->mesh_id;

    ARRAY LightTreeCacheTriangle* material_triangles = mesh->material_triangles[task->material_slot_id];

    LightTreeCacheTriangle* triangle = material_triangles + task->material_slot_tri_id;

    triangle->average_intensity = tree->integrator.intensities[task_id];
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_compute_instance_fragments(LightTree* tree, const uint32_t instance_id) {
  __CHECK_NULL_ARGUMENT(tree);

  LightTreeCacheInstance* instance = tree->cache.instances + instance_id;

  const Vec128 offset   = vec128_set(instance->translation.x, instance->translation.y, instance->translation.z, 0.0f);
  const Vec128 scale    = vec128_set(instance->scale.x, instance->scale.y, instance->scale.z, 1.0f);
  const Vec128 rotation = vec128_set(-instance->rotation.x, -instance->rotation.y, -instance->rotation.z, instance->rotation.w);

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

      const Vec128 vertex  = vec128_add(vec128_mul(vec128_rotate_quaternion(triangle->vertex, rotation), scale), offset);
      const Vec128 vertex1 = vec128_add(vec128_mul(vec128_rotate_quaternion(triangle->vertex1, rotation), scale), offset);
      const Vec128 vertex2 = vec128_add(vec128_mul(vec128_rotate_quaternion(triangle->vertex2, rotation), scale), offset);

      const Vec128 cross = vec128_cross(vec128_sub(vertex1, vertex), vec128_sub(vertex2, vertex));

      const float area = 0.5f * vec128_norm2(cross);

      if (area == 0.0f || triangle->average_intensity == 0.0f)
        continue;

      LightTreeFragment fragment;
      fragment.low               = vec128_min(vertex, vec128_min(vertex1, vertex2));
      fragment.high              = vec128_max(vertex, vec128_max(vertex1, vertex2));
      fragment.middle            = vec128_scale(vec128_add(vertex, vec128_add(vertex1, vertex2)), 1.0f / 3.0f);
      fragment.average_direction = vec128_scale(cross, 0.5f / area);
      fragment.v0                = vertex;
      fragment.v1                = vertex1;
      fragment.v2                = vertex2;
      fragment.power             = material->constant_emission_intensity * area * triangle->average_intensity;
      fragment.intensity         = material->constant_emission_intensity * triangle->average_intensity;
      fragment.instance_id       = instance_id;
      fragment.material_slot_id  = material_slot_id;
      fragment.material_tri_id   = tri_id;

      __FAILURE_HANDLE(array_get_num_elements(instance->bvh_triangles, &fragment.instance_cache_tri_id));

      __FAILURE_HANDLE(array_push(&instance->fragments, &fragment));

      LightTreeBVHTriangle bvh_triangle;
      bvh_triangle.vertex  = vertex;
      bvh_triangle.vertex1 = vertex1;
      bvh_triangle.vertex2 = vertex2;

      __FAILURE_HANDLE(array_push(&instance->bvh_triangles, &bvh_triangle));
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_handle_dirty_states(LightTree* tree, Device* device) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(array_clear(tree->integrator.tasks));

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.meshes, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    LightTreeCacheMesh* mesh = tree->cache.meshes + mesh_id;

    if (mesh->is_dirty) {
      __FAILURE_HANDLE(_light_tree_queue_texture_integrations(tree, mesh_id));
    }
  }

  __FAILURE_HANDLE(_light_tree_integrate(tree, device));

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
  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    tree->cache.meshes[mesh_id].is_dirty = false;
  }

  uint32_t num_materials;
  __FAILURE_HANDLE(array_get_num_elements(tree->cache.materials, &num_materials));

  for (uint32_t material_id = 0; material_id < num_materials; material_id++) {
    tree->cache.materials[material_id].needs_reintegration = false;
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

    if (!instance->active || (instance->fragments == (LightTreeFragment*) 0))
      continue;

    uint32_t num_fragments;
    __FAILURE_HANDLE(array_get_num_elements(instance->fragments, &num_fragments));

    total_fragments += num_fragments;
  }

  work->fragments_count = total_fragments;

  __FAILURE_HANDLE(host_malloc(&work->fragments, sizeof(LightTreeFragment) * work->fragments_count));

  uint32_t fragment_offset = 0;
  float total_power        = 0.0f;

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    LightTreeCacheInstance* instance = tree->cache.instances + instance_id;

    if (!instance->active || (instance->fragments == (LightTreeFragment*) 0))
      continue;

    uint32_t num_fragments;
    __FAILURE_HANDLE(array_get_num_elements(instance->fragments, &num_fragments));

    for (uint32_t fragment_id = 0; fragment_id < num_fragments; fragment_id++) {
      const LightTreeFragment fragment = instance->fragments[fragment_id];

      total_power += fragment.power;

      work->fragments[fragment_offset + fragment_id] = fragment;
    }

    fragment_offset += num_fragments;
  }

  Vec128 mean = vec128_set_1(0.0f);

  for (uint32_t fragment_id = 0; fragment_id < total_fragments; fragment_id++) {
    const LightTreeFragment fragment = work->fragments[fragment_id];

    mean = vec128_add(mean, vec128_scale(fragment.middle, fragment.power / total_power));
  }

  float sum_one_over_distance_sq = 0.0f;

  for (uint32_t fragment_id = 0; fragment_id < total_fragments; fragment_id++) {
    const LightTreeFragment fragment = work->fragments[fragment_id];

    const Vec128 diff   = vec128_sub(fragment.middle, mean);
    const float dist_sq = vec128_dot(diff, diff);

    sum_one_over_distance_sq += (1.0f / dist_sq) * (fragment.power / total_power);
  }

  tree->scene_data.total_power = total_power * sum_one_over_distance_sq;
  tree->scene_data.num_lights  = total_fragments;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_tree_free_data(LightTree* tree) {
  __CHECK_NULL_ARGUMENT(tree);

  if (tree->root_data) {
    __FAILURE_HANDLE(host_free(&tree->root_data));
    tree->root_size = 0;
  }

  if (tree->nodes_data) {
    __FAILURE_HANDLE(host_free(&tree->nodes_data));
    tree->nodes_size = 0;
  }

  if (tree->tri_handle_map_data) {
    __FAILURE_HANDLE(host_free(&tree->tri_handle_map_data));
    tree->tri_handle_map_size = 0;
  }

  if (tree->bvh_vertex_buffer_data) {
    __FAILURE_HANDLE(host_free(&tree->bvh_vertex_buffer_data));
    tree->bvh_vertex_buffer_size = 0;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_build(LightTree* tree, Device* device) {
  __CHECK_NULL_ARGUMENT(tree);

  LUM_UNUSED(device);

  // Only build if the cache is dirty.
  if (!tree->cache.is_dirty)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(_light_tree_free_data(tree));

  __FAILURE_HANDLE(_light_tree_handle_dirty_states(tree, device));

  LightTreeWork work;
  memset(&work, 0, sizeof(LightTreeWork));

  __FAILURE_HANDLE(_light_tree_collect_fragments(tree, &work));
  __FAILURE_HANDLE(_light_tree_build_binary_bvh(&work));
  __FAILURE_HANDLE(_light_tree_build_traversal_structure(&work));
  __FAILURE_HANDLE(_light_tree_collapse(&work));
  __FAILURE_HANDLE(_light_tree_finalize(tree, &work));
#ifdef LIGHT_TREE_DEBUG_OUTPUT
  __FAILURE_HANDLE(_light_tree_debug_output(&work));
#endif /* LIGHT_TREE_DEBUG_OUTPUT */
  __FAILURE_HANDLE(_light_tree_clear_work(&work));

  __FAILURE_HANDLE(optix_bvh_light_build(tree->optix_bvh, device, tree));

  tree->build_id++;

  return LUMINARY_SUCCESS;
}

LuminaryResult light_tree_unload_integrator(LightTree* tree) {
  __CHECK_NULL_ARGUMENT(tree);

  if (tree->integrator.allocated_tasks > 0) {
    __FAILURE_HANDLE(host_free(&tree->integrator.mesh_ids));
    __FAILURE_HANDLE(host_free(&tree->integrator.triangle_ids));
    __FAILURE_HANDLE(host_free(&tree->integrator.intensities));
    __FAILURE_HANDLE(device_free(&tree->integrator.device_mesh_ids));
    __FAILURE_HANDLE(device_free(&tree->integrator.device_triangle_ids));
    __FAILURE_HANDLE(device_free(&tree->integrator.device_intensities));

    tree->integrator.allocated_tasks = 0;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////

LuminaryResult light_tree_destroy(LightTree** tree) {
  __CHECK_NULL_ARGUMENT(tree);
  __CHECK_NULL_ARGUMENT(*tree);

  // Tasks are allocated by the GPU so they must be freed in a DEVICE_CTX_FUNC.
  __DEBUG_ASSERT((*tree)->integrator.allocated_tasks == 0);

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
  }

  __FAILURE_HANDLE(optix_bvh_destroy(&(*tree)->optix_bvh));

  __FAILURE_HANDLE(array_destroy(&(*tree)->integrator.tasks));

  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.meshes));
  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.instances));
  __FAILURE_HANDLE(array_destroy(&(*tree)->cache.materials));

  __FAILURE_HANDLE(_light_tree_free_data(*tree));

  __FAILURE_HANDLE(host_free(tree));

  return LUMINARY_SUCCESS;
}
