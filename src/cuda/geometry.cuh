#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"
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

__global__ void geometry_generate_g_buffer() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    const float4* hit_address = (float4*) (device.scene.triangles + task.hit_id);

    const float4 t1 = __ldg(hit_address);
    const float4 t2 = __ldg(hit_address + 1);
    const float4 t3 = __ldg(hit_address + 2);
    const float4 t4 = __ldg(hit_address + 3);
    const float4 t5 = __ldg(hit_address + 4);
    const float4 t6 = __ldg(hit_address + 5);
    const float t7  = __ldg((float*) (hit_address + 6));

    const vec3 vertex = get_vector(t1.x, t1.y, t1.z);
    const vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
    const vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

    const float2 coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

    const UV vertex_texture = get_UV(t5.z, t5.w);
    const UV edge1_texture  = get_UV(t6.x, t6.y);
    const UV edge2_texture  = get_UV(t6.z, t6.w);

    const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);

    const int material_id = __float_as_int(t7);

    const Material mat = load_material(device.materials, material_id);

    const vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
    const vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
    const vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

    bool is_inside;
    const vec3 normal = geometry_compute_normal(
      vertex_normal, edge1_normal, edge2_normal, task.ray, edge1, edge2, edge1_texture, edge2_texture, mat.normal_map, coords, tex_coords,
      is_inside);

    RGBAF albedo;

    if (mat.albedo_map != TEXTURE_NONE) {
      const float4 albedo_f = texture_load(device.ptrs.albedo_atlas[mat.albedo_map], tex_coords);
      albedo.r              = albedo_f.x;
      albedo.g              = albedo_f.y;
      albedo.b              = albedo_f.z;
      albedo.a              = albedo_f.w;
    }
    else {
      albedo.r = 0.9f;
      albedo.g = 0.9f;
      albedo.b = 0.9f;
      albedo.a = 1.0f;
    }

    if (albedo.a < device.scene.material.alpha_cutoff)
      albedo.a = 0.0f;

    RGBF emission = get_color(0.0f, 0.0f, 0.0f);

    if (mat.illuminance_map != TEXTURE_NONE && device.scene.material.lights_active) {
      const float4 illuminance_f = texture_load(device.ptrs.illuminance_atlas[mat.illuminance_map], tex_coords);

      emission = get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z);
      emission = scale_color(emission, device.scene.material.default_material.b * illuminance_f.w * albedo.a);
    }

    float roughness;
    float metallic;

    if (mat.material_map != TEXTURE_NONE) {
      const float4 material_f = texture_load(device.ptrs.material_atlas[mat.material_map], tex_coords);

      roughness = (1.0f - material_f.x);
      metallic  = material_f.y;
    }
    else {
      roughness = (1.0f - device.scene.material.default_material.r);
      metallic  = device.scene.material.default_material.g;
    }

    uint32_t flags = 0;

    const QuasiRandomTarget random_target =
      (device.iteration_type == TYPE_LIGHT) ? QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY : QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY;

    if (albedo.a < 1.0f && quasirandom_sequence_1D(random_target, pixel) > albedo.a) {
      flags |= G_BUFFER_TRANSPARENT_PASS;
    }

    if (!(flags & G_BUFFER_TRANSPARENT_PASS) && !state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      flags |= G_BUFFER_REQUIRES_SAMPLING;
    }

    if (flags & G_BUFFER_TRANSPARENT_PASS) {
      task.position = add_vector(task.position, scale_vector(task.ray, eps * get_length(task.position)));
    }
    else {
      task.position = add_vector(task.position, scale_vector(task.ray, -eps * get_length(task.position)));
    }

    if (is_inside) {
      flags |= G_BUFFER_REFRACTION_IS_INSIDE;
    }

    GBufferData data;
    data.hit_id           = task.hit_id;
    data.albedo           = albedo;
    data.emission         = emission;
    data.normal           = normal;
    data.position         = task.position;
    data.V                = scale_vector(task.ray, -1.0f);
    data.roughness        = roughness;
    data.metallic         = metallic;
    data.flags            = flags;
    data.refraction_index = mat.refraction_index;

    store_g_buffer_data(data, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_geometry_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = load_g_buffer_data(pixel);

    RGBF record = load_RGBF(device.records + pixel);

    if (data.albedo.a > 0.0f && color_any(data.emission)) {
      write_albedo_buffer(add_color(data.emission, opaque_color(data.albedo)), pixel);

      RGBF emission = mul_color(data.emission, record);

      if (device.iteration_type == TYPE_BOUNCE) {
        const float mis_weight = device.ptrs.mis_buffer[pixel];
        emission               = scale_color(emission, mis_weight);
      }
      else if (device.iteration_type == TYPE_LIGHT) {
        emission = scale_color(emission, get_light_transparency_weight(pixel));
      }

      const uint32_t light = device.ptrs.light_sample_history[pixel];

      const uint32_t triangle_light_id = __ldg(&(device.scene.triangles[task.hit_id].light_id));

      if (proper_light_sample(light, triangle_light_id)) {
        store_RGBF(device.ptrs.frame_buffer + pixel, add_color(load_RGBF(device.ptrs.frame_buffer + pixel), emission));
      }
    }

    write_normal_buffer(data.normal, pixel);

    if (data.flags & G_BUFFER_TRANSPARENT_PASS && !device.scene.material.colored_transparency) {
      data.albedo.r = 1.0f;
      data.albedo.g = 1.0f;
      data.albedo.b = 1.0f;
    }

    BRDFInstance brdf = brdf_get_instance(data.albedo, data.V, data.normal, data.roughness, data.metallic);

    if (data.flags & G_BUFFER_TRANSPARENT_PASS) {
      if (device.iteration_type != TYPE_LIGHT) {
        const float refraction_index =
          (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index / 1.0f : 1.0f / data.refraction_index;

        brdf = brdf_sample_ray_refraction(brdf, refraction_index, pixel);
      }
      else {
        brdf.term = mul_color(brdf.term, opaque_color(data.albedo));
        brdf.L    = task.ray;
      }

      record = mul_color(record, brdf.term);

      TraceTask new_task;
      new_task.origin = data.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          if (validate_trace_task(new_task, pixel, record)) {
            device.ptrs.mis_buffer[pixel] = 1.0f;
            store_RGBF(device.ptrs.bounce_records + pixel, record);
            store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          }
          break;
        case TYPE_LIGHT:
          device.ptrs.light_transparency_weight_buffer[pixel] *= 2.0f;
          if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY_ROULETTE, pixel) > 0.5f) {
            if (state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
              store_RGBF(device.ptrs.light_records + pixel, record);
              store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
            }
          }
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      if (!material_is_mirror(data.roughness, data.metallic))
        write_albedo_buffer(opaque_color(data.albedo), pixel);

      bool bounce_is_specular;
      BRDFInstance bounce_brdf = brdf_sample_ray(brdf, pixel, bounce_is_specular);

      float bounce_mis_weight = 1.0f;

      if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
        uint32_t light_history_buffer_entry = LIGHT_ID_ANY;
        LightSample light                   = load_light_sample(device.ptrs.light_samples, pixel);

        if (light.weight > 0.0f) {
          const BRDFInstance brdf_sample = brdf_apply_sample_weight(brdf_apply_sample(brdf, light, data.position, pixel));

          const RGBF light_record = mul_color(record, brdf_sample.term);

          TraceTask light_task;
          light_task.origin = data.position;
          light_task.ray    = brdf_sample.L;
          light_task.index  = task.index;

          if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
            const float light_mis_weight = (bounce_is_specular) ? data.roughness * data.roughness : 1.0f;
            bounce_mis_weight            = 1.0f - light_mis_weight;

            store_RGBF(device.ptrs.light_records + pixel, scale_color(light_record, light_mis_weight));
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }

        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
      }

      RGBF bounce_record = mul_color(record, bounce_brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = data.position;
      bounce_task.ray    = bounce_brdf.L;
      bounce_task.index  = task.index;

      if (validate_trace_task(bounce_task, pixel, bounce_record)) {
        device.ptrs.mis_buffer[pixel] = bounce_mis_weight;
        store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
        store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_debug_geometry_tasks() {
  const int task_count = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(i));
    const int pixel   = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      const GBufferData data = load_g_buffer_data(pixel);

      store_RGBF(device.ptrs.frame_buffer + pixel, add_color(opaque_color(data.albedo), data.emission));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(value, value, value));
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = load_g_buffer_data(pixel);

      const vec3 normal = data.normal;

      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device.shading_mode == SHADING_HEAT) {
      const float cost  = device.ptrs.trace_result_buffer[pixel].depth;
      const float value = 0.1f * cost;
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(red, green, blue));
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

      store_RGBF(device.ptrs.frame_buffer + pixel, color);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      const float4* hit_address = (float4*) (device.scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t5 = __ldg(hit_address + 4);
      const float4 t6 = __ldg(hit_address + 5);
      const float2 t7 = __ldg((float2*) (hit_address + 6));

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

      const float2 coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      UV vertex_texture = get_UV(t5.z, t5.w);
      UV edge1_texture  = get_UV(t6.x, t6.y);
      UV edge2_texture  = get_UV(t6.z, t6.w);

      const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);

      const int material_id = __float_as_int(t7.x);
      const int light_id    = __float_as_int(t7.y);

      RGBF color;

      if (light_id != LIGHT_ID_NONE) {
        color = get_color(100.0f, 100.0f, 100.0f);
      }
      else {
        const Material mat = load_material(device.materials, material_id);

        if (mat.albedo_map != TEXTURE_NONE) {
          const float4 albedo_f = texture_load(device.ptrs.albedo_atlas[mat.albedo_map], tex_coords);
          color                 = get_color(albedo_f.x, albedo_f.y, albedo_f.z);
        }
        else {
          color = get_color(0.9f, 0.9f, 0.9f);
        }

        color = scale_color(color, 0.1f);
      }

      store_RGBF(device.ptrs.frame_buffer + pixel, color);
    }
  }
}

#endif /* CU_GEOMETRY_H */
