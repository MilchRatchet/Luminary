#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "bsdf.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "mis.cuh"
#include "restir.cuh"
#include "state.cuh"
#include "texture_utils.cuh"

__device__ vec3 geometry_compute_normal(
  vec3 v_normal, vec3 e1_normal, vec3 e2_normal, vec3 ray, vec3 e1, vec3 e2, UV e1_tex, UV e2_tex, uint16_t normal_map, float2 coords,
  UV tex_coords, bool& is_inside) {
  vec3 face_normal = normalize_vector(cross_product(e1, e2));
  vec3 normal      = lerp_normals(v_normal, e1_normal, e2_normal, coords, face_normal);
  is_inside        = dot_product(face_normal, ray) > 0.0f;

  if (normal_map != TEXTURE_NONE) {
    const float4 normal_f = texture_load(device.ptrs.normal_atlas[normal_map], tex_coords);

    vec3 map_normal = get_vector(normal_f.x, normal_f.y, normal_f.z);

    map_normal = scale_vector(map_normal, 2.0f);
    map_normal = sub_vector(map_normal, get_vector(1.0f, 1.0f, 1.0f));

    Mat3x3 tangent_space = cotangent_frame(normal, e1, e2, e1_tex, e2_tex);

    normal = normalize_vector(transform_vec3(tangent_space, map_normal));

    if (dot_product(normal, ray) > 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }
  }
  else {
    if (dot_product(face_normal, normal) < 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }

    if (dot_product(face_normal, ray) > 0.0f) {
      face_normal = scale_vector(face_normal, -1.0f);
      normal      = scale_vector(normal, -1.0f);
    }

    /*
     * I came up with this quickly, problem is that we need to rotate the normal
     * towards the face normal with our ray is "behind" the shading normal
     */
    if (dot_product(normal, ray) > 0.0f) {
      const float a = sqrtf(1.0f - dot_product(face_normal, ray));
      const float b = dot_product(face_normal, normal);
      const float t = a / (a + b + eps);
      normal        = normalize_vector(add_vector(scale_vector(face_normal, t), scale_vector(normal, 1.0f - t)));
    }
  }

  return normal;
}

__device__ GBufferData geometry_generate_g_buffer(const GeometryTask task, const int pixel) {
  const float4 t1 = __ldg((float4*) triangle_get_entry_address(0, 0, task.hit_id));
  const float4 t2 = __ldg((float4*) triangle_get_entry_address(1, 0, task.hit_id));
  const float4 t3 = __ldg((float4*) triangle_get_entry_address(2, 0, task.hit_id));
  const float4 t4 = __ldg((float4*) triangle_get_entry_address(3, 0, task.hit_id));
  const float4 t5 = __ldg((float4*) triangle_get_entry_address(4, 0, task.hit_id));
  const float4 t6 = __ldg((float4*) triangle_get_entry_address(5, 0, task.hit_id));

  const uint32_t material_id = load_triangle_material_id(task.hit_id);

  const vec3 vertex = get_vector(t1.x, t1.y, t1.z);
  const vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
  const vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

  const float2 coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

  const UV vertex_texture = get_UV(t5.z, t5.w);
  const UV edge1_texture  = get_UV(t6.x, t6.y);
  const UV edge2_texture  = get_UV(t6.z, t6.w);

  const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);

  const Material mat = load_material(device.materials, material_id);

  const vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
  const vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
  const vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

  bool is_inside;
  const vec3 normal = geometry_compute_normal(
    vertex_normal, edge1_normal, edge2_normal, task.ray, edge1, edge2, edge1_texture, edge2_texture, mat.normal_map, coords, tex_coords,
    is_inside);

  RGBAF albedo = mat.albedo;
  if (mat.albedo_map != TEXTURE_NONE) {
    const float4 albedo_f = texture_load(device.ptrs.albedo_atlas[mat.albedo_map], tex_coords);
    albedo.r              = albedo_f.x;
    albedo.g              = albedo_f.y;
    albedo.b              = albedo_f.z;
    albedo.a              = albedo_f.w;
  }

  if (albedo.a < device.scene.material.alpha_cutoff)
    albedo.a = 0.0f;

  RGBF emission = (device.scene.material.lights_active) ? mat.emission : get_color(0.0f, 0.0f, 0.0f);
  if (mat.luminance_map != TEXTURE_NONE && device.scene.material.lights_active) {
    const float4 luminance_f = texture_load(device.ptrs.luminance_atlas[mat.luminance_map], tex_coords);

    emission = get_color(luminance_f.x, luminance_f.y, luminance_f.z);
    emission = scale_color(emission, luminance_f.w * albedo.a);
  }
  emission = scale_color(emission, device.scene.material.default_material.b);

  if (device.scene.material.light_side_mode != LIGHT_SIDE_MODE_BOTH) {
    const vec3 face_normal = cross_product(edge1, edge2);
    const float side       = (device.scene.material.light_side_mode == LIGHT_SIDE_MODE_ONE_CW) ? 1.0f : -1.0f;

    if (dot_product(face_normal, task.ray) * side > 0.0f) {
      emission = get_color(0.0f, 0.0f, 0.0f);
    }
  }

  float roughness = mat.roughness;
  float metallic  = mat.metallic;
  if (mat.material_map != TEXTURE_NONE) {
    const float4 material_f = texture_load(device.ptrs.material_atlas[mat.material_map], tex_coords);

    roughness = material_f.x;
    metallic  = material_f.y;
  }
  else if (device.scene.material.override_materials) {
    roughness = 1.0f - device.scene.material.default_material.r;
    metallic  = device.scene.material.default_material.g;
  }

  if ((!device.scene.material.override_materials || mat.material_map != TEXTURE_NONE) && device.scene.material.invert_roughness) {
    roughness = 1.0f - roughness;
  }

  uint32_t flags = G_BUFFER_REQUIRES_SAMPLING;

  if (is_inside) {
    flags |= G_BUFFER_REFRACTION_IS_INSIDE;
  }

  const IORStackMethod ior_stack_method =
    (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior = ior_stack_interact(mat.refraction_index, pixel, ior_stack_method);

  GBufferData data;
  data.hit_id             = task.hit_id;
  data.albedo             = albedo;
  data.emission           = emission;
  data.normal             = normal;
  data.position           = task.position;
  data.V                  = scale_vector(task.ray, -1.0f);
  data.roughness          = roughness;
  data.metallic           = metallic;
  data.flags              = flags;
  data.ior_in             = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? mat.refraction_index : ray_ior;
  data.ior_out            = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ray_ior : mat.refraction_index;
  data.colored_dielectric = device.scene.material.colored_transparency;

  return data;
}

LUMINARY_KERNEL void process_geometry_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = geometry_generate_g_buffer(task, pixel);

    RGBF record = load_RGBF(device.records + pixel);

    if (data.albedo.a > 0.0f && color_any(data.emission)) {
      write_albedo_buffer(add_color(data.emission, opaque_color(data.albedo)), pixel);

      RGBF emission = mul_color(data.emission, record);

      if (device.iteration_type == TYPE_BOUNCE) {
        const float mis_weight = mis_weight_bsdf_sampled(data, pixel);
        emission               = scale_color(emission, mis_weight);
      }

      write_beauty_buffer(emission, pixel);
    }

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    uint32_t light_history_buffer_entry = LIGHT_ID_ANY;
    float light_sample_marginal;
    LightSample light = restir_sample_reservoir(data, record, task.index, light_sample_marginal);

    if (light.weight > 0.0f) {
      RGBF light_weight;
      bool is_transparent_pass;
      const vec3 light_ray = restir_apply_sample_shading(data, light, task.index, light_weight, is_transparent_pass);

      const RGBF light_record = mul_color(record, light_weight);

      const float shift           = (is_transparent_pass) ? -eps : eps;
      const vec3 shifted_position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

      TraceTask light_task;
      light_task.origin = shifted_position;
      light_task.ray    = light_ray;
      light_task.index  = task.index;

      const float light_mis_weight = mis_weight_light_sampled(data, light_ray, bounce_info, light_sample_marginal);
      store_RGBF(device.ptrs.light_records + pixel, scale_color(light_record, light_mis_weight));
      light_history_buffer_entry = light.id;
      store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
    }

    device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    const float shift           = (bounce_info.is_transparent_pass) ? -eps : eps;
    const vec3 shifted_position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_in, pixel, ior_stack_method);
    }

    TraceTask bounce_task;
    bounce_task.origin = shifted_position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);

      MISData mis_data;
      mis_data.light_sampled_technique = light.weight - light.sample_weight;
      mis_data.bsdf_marginal           = bsdf_marginal;

      mis_store_data(data, record, mis_data, bounce_ray, pixel);
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

LUMINARY_KERNEL void process_geometry_light_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = geometry_generate_g_buffer(task, pixel);

    RGBF record = load_RGBF(device.records + pixel);

    if (color_any(data.emission)) {
      const uint32_t light             = device.ptrs.light_sample_history[pixel];
      const uint32_t triangle_light_id = load_triangle_light_id(data.hit_id);

      if (proper_light_sample(light, triangle_light_id)) {
        write_beauty_buffer(mul_color(data.emission, record), pixel);
      }
    }
  }
}

LUMINARY_KERNEL void process_debug_geometry_tasks() {
  const int task_count = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(i));
    const int pixel   = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      write_beauty_buffer(add_color(opaque_color(data.albedo), data.emission), pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      const vec3 normal = data.normal;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_HEAT) {
      const float cost  = device.ptrs.trace_result_buffer[pixel].depth;
      const float value = 0.1f * cost;
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      write_beauty_buffer(get_color(red, green, blue), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0, task.hit_id);

      const uint16_t r = v & 0x7ff;
      const uint16_t g = (v >> 10) & 0x7ff;
      const uint16_t b = (v >> 20) & 0x7ff;

      const float cr = ((float) r) / 0x7ff;
      const float cg = ((float) g) / 0x7ff;
      const float cb = ((float) b) / 0x7ff;

      const RGBF color = get_color(cr, cg, cb);

      write_beauty_buffer(color, pixel, true);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      RGBF color = add_color(scale_color(opaque_color(data.albedo), 0.5f), scale_color(data.emission, 100.0f));

      write_beauty_buffer(color, pixel, true);
    }
  }
}

#endif /* CU_GEOMETRY_H */
