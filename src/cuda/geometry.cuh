__global__ __launch_bounds__(THREADS_PER_BLOCK, 8) void process_geometry_tasks() {
  const int task_count   = device.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6];
  int light_trace_count  = device.light_trace_count[threadIdx.x + blockIdx.x * blockDim.x];
  int bounce_trace_count = device.bounce_trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device_trace_tasks + get_task_address(i));
    const int pixel   = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

    const float4 t1 = __ldg(hit_address);
    const float4 t2 = __ldg(hit_address + 1);
    const float4 t3 = __ldg(hit_address + 2);
    const float4 t4 = __ldg(hit_address + 3);
    const float4 t5 = __ldg(hit_address + 4);
    const float4 t6 = __ldg(hit_address + 5);
    const float2 t7 = __ldg((float2*) (hit_address + 6));

    const vec3 vertex = get_vector(t1.x, t1.y, t1.z);
    const vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
    const vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

    vec3 face_normal = normalize_vector(cross_product(edge1, edge2));

    const UV coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

    vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
    vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
    vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

    vec3 normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, coords, face_normal);

    if (dot_product(normal, face_normal) < 0.0f) {
      face_normal = scale_vector(face_normal, -1.0f);
    }

    if (dot_product(face_normal, ray) > 0.0f) {
      normal        = scale_vector(normal, -1.0f);
      face_normal   = scale_vector(face_normal, -1.0f);
      vertex_normal = scale_vector(vertex_normal, -1.0f);
      edge1_normal  = scale_vector(edge1_normal, -1.0f);
      edge2_normal  = scale_vector(edge2_normal, -1.0f);
    }

    const vec3 terminator = terminator_fix(task.position, vertex, edge1, edge2, vertex_normal, edge1_normal, edge2_normal, coords);

    const UV vertex_texture = get_UV(t5.z, t5.w);
    const UV edge1_texture  = get_UV(t6.x, t6.y);
    const UV edge2_texture  = get_UV(t6.z, t6.w);

    const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);

    const int texture_object         = __float_as_int(t7.x);
    const uint32_t triangle_light_id = __float_as_uint(t7.y);

    const ushort4 maps = __ldg((ushort4*) (device_texture_assignments + texture_object));

    float roughness;
    float metallic;
    float intensity;

    if (maps.z) {
      const float4 material_f = tex2D<float4>(device.material_atlas[maps.z], tex_coords.u, 1.0f - tex_coords.v);

      roughness = (1.0f - material_f.x);
      metallic  = material_f.y;
      intensity = material_f.z * 255.0f;
    }
    else {
      roughness = (1.0f - device_scene.material.default_material.r);
      metallic  = device_scene.material.default_material.g;
      intensity = device_scene.material.default_material.b;
    }

    RGBAF albedo;

    if (maps.x) {
      const float4 albedo_f = tex2D<float4>(device.albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
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

    RGBAhalf emission = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);

    if (maps.y && device_scene.material.lights_active) {
      const float4 illuminance_f = tex2D<float4>(device.illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

      emission = get_RGBAhalf(illuminance_f.x, illuminance_f.y, illuminance_f.z, illuminance_f.w);
      emission = scale_RGBAhalf(emission, intensity);
    }

    if (albedo.a < device_scene.material.alpha_cutoff)
      albedo.a = 0.0f;

    RGBAhalf record = load_RGBAhalf(device_records + pixel);

    if (albedo.a > 0.0f && any_RGBAhalf(emission)) {
      write_albedo_buffer(RGBAhalf_to_RGBF(emission), pixel);

      emission = mul_RGBAhalf(emission, record);

      const uint32_t light = device.light_sample_history[pixel];

      if (proper_light_sample(light, triangle_light_id)) {
        store_RGBAhalf(device.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.frame_buffer + pixel), emission));
      }
    }

    if (albedo.a < 1.0f && white_noise() > albedo.a) {
      task.position = add_vector(task.position, scale_vector(ray, 2.0f * eps));

      RGBF record_alpha_factor;

      record_alpha_factor.r = (albedo.r * albedo.a + 1.0f - albedo.a);
      record_alpha_factor.g = (albedo.g * albedo.a + 1.0f - albedo.a);
      record_alpha_factor.b = (albedo.b * albedo.a + 1.0f - albedo.a);

      record = mul_RGBAhalf(record, RGBF_to_RGBAhalf(record_alpha_factor));

      TraceTask new_task;
      new_task.origin = task.position;
      new_task.ray    = ray;
      new_task.index  = task.index;
      new_task.state  = task.state;

      switch (device_iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBAhalf(device.bounce_records + pixel, record);
          store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f)
            break;
          store_RGBAhalf(device.light_records + pixel, scale_RGBAhalf(record, 2.0f));
          store_trace_task(device.light_trace + get_task_address(light_trace_count++), new_task);
          device.state_buffer[pixel] |= STATE_LIGHT_OCCUPIED;
          break;
      }
    }
    else if (device_iteration_type != TYPE_LIGHT) {
      const int is_mirror = material_is_mirror(roughness, metallic);

      if (!is_mirror)
        write_albedo_buffer(get_color(albedo.r, albedo.g, albedo.b), pixel);

      const vec3 V               = scale_vector(ray, -1.0f);
      const int use_light_sample = !is_mirror && !(device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

      task.position = terminator;
      task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));
      task.state    = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance(RGBAF_to_RGBAhalf(albedo), V, normal, roughness, metallic);

      if (use_light_sample) {
        LightSample light = load_light_sample(device.light_samples, pixel);

        light = brdf_finalize_light_sample(light, task.position);

        if (light.weight > 0.0f) {
          brdf.L = brdf_sample_light_ray(light, task.position);

          const RGBAhalf light_record = mul_RGBAhalf(record, scale_RGBAhalf(brdf_evaluate(brdf).term, light.weight));

          TraceTask light_task;
          light_task.origin = task.position;
          light_task.ray    = brdf.L;
          light_task.index  = task.index;
          light_task.state  = task.state;

          if (any_RGBAhalf(light_record)) {
            store_RGBAhalf(device.light_records + pixel, light_record);
            light_history_buffer_entry = light.id;
            store_trace_task(device.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }
      }

      if (!(device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED))
        device.light_sample_history[pixel] = light_history_buffer_entry;

      brdf                   = brdf_sample_ray(brdf, task.index, task.state);
      RGBAhalf bounce_record = mul_RGBAhalf(record, brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = brdf.L;
      bounce_task.index  = task.index;
      bounce_task.state  = task.state;

      if (validate_trace_task(bounce_task, bounce_record)) {
        store_RGBAhalf(device.bounce_records + pixel, bounce_record);
        store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
    }
  }

  device.light_trace_count[threadIdx.x + blockIdx.x * blockDim.x]  = light_trace_count;
  device.bounce_trace_count[threadIdx.x + blockIdx.x * blockDim.x] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_debug_geometry_tasks() {
  const int task_count = device.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device_trace_tasks + get_task_address(i));
    const int pixel   = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      RGBF color = get_color(0.0f, 0.0f, 0.0f);

      const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t4 = __ldg(hit_address + 3);
      const float4 t5 = __ldg(hit_address + 4);
      const float4 t6 = __ldg(hit_address + 5);
      const float2 t7 = __ldg((float2*) (hit_address + 6));

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

      const UV coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      UV vertex_texture = get_UV(t5.z, t5.w);
      UV edge1_texture  = get_UV(t6.x, t6.y);
      UV edge2_texture  = get_UV(t6.z, t6.w);

      const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);

      const int texture_object = __float_as_int(t7.x);

      const ushort4 maps = __ldg((ushort4*) (device_texture_assignments + texture_object));

      if (maps.x) {
        const float4 albedo_f = tex2D<float4>(device.albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
        color                 = add_color(color, get_color(albedo_f.x, albedo_f.y, albedo_f.z));
      }
      else {
        color = add_color(color, get_color(0.9f, 0.9f, 0.9f));
      }

      if (maps.y && device_scene.material.lights_active) {
        const float4 illuminance_f = tex2D<float4>(device.illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

        color = add_color(color, get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z));
      }

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(color);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device_scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      store_RGBAhalf(device.frame_buffer + pixel, get_RGBAhalf(value, value, value, value));
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t4 = __ldg(hit_address + 3);
      const float4 t5 = __ldg(hit_address + 4);

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

      vec3 face_normal = normalize_vector(cross_product(edge1, edge2));

      const UV coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
      vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
      vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

      vec3 normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, coords, face_normal);

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device_shading_mode == SHADING_HEAT) {
      const float cost  = device.trace_result_buffer[pixel].depth;
      const float value = 1.0f - 1.0f / (powf(cost, 0.25f));
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(red, green, blue));
    }
    else if (device_shading_mode == SHADING_WIREFRAME) {
      const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float t3  = __ldg((float*) (hit_address + 2));

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3);

      const UV coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      int a = fabsf(coords.u + coords.v - 1.0f) < 0.001f;
      int b = fabsf(coords.u) < 0.001f;
      int c = fabsf(coords.v) < 0.001f;

      float light = (a || b || c) ? 1.0f : 0.0f;

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(light, 0.5f * light, 0.0f));
    }
  }
}
