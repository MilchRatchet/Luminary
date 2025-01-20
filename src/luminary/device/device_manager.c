#include "device_manager.h"

#include "ceb.h"
#include "device_structs.h"
#include "device_utils.h"
#include "host/internal_host.h"
#include "internal_error.h"
#include "scene.h"

#define DEVICE_MANAGER_RINGBUFFER_SIZE (0x100000ull)
#define DEVICE_MANAGER_QUEUE_SIZE (0x400ull)

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

static LuminaryResult _device_manager_queue_worker(DeviceManager* device_manager) {
  bool success = true;

  while (success) {
    QueueEntry entry;
    __FAILURE_HANDLE(queue_pop_blocking(device_manager->work_queue, &entry, &success));

    if (!success)
      return LUMINARY_SUCCESS;

    __FAILURE_HANDLE(wall_time_set_string(device_manager->queue_wall_time, entry.name));
    __FAILURE_HANDLE(wall_time_start(device_manager->queue_wall_time));

    __FAILURE_HANDLE(entry.function(device_manager, entry.args));

    if (entry.clear_func) {
      __FAILURE_HANDLE(entry.clear_func(device_manager, entry.args));
    }

    __FAILURE_HANDLE(wall_time_stop(device_manager->queue_wall_time));
    __FAILURE_HANDLE(wall_time_set_string(device_manager->queue_wall_time, (const char*) 0));

#ifdef LUMINARY_WORK_QUEUE_STATS_PRINT
    double time;
    __FAILURE_HANDLE(wall_time_get_time(device_manager->queue_wall_time, &time));

    if (time > LUMINARY_WORK_QUEUE_STATS_PRINT_THRESHOLD) {
      warn_message("device_manager queue: %s (%fs)", entry.name, time);
    }
#endif
  }

  return LUMINARY_SUCCESS;
}

static bool _device_manager_queue_entry_equal_operator(QueueEntry* left, QueueEntry* right) {
  return (left->function == right->function);
}

////////////////////////////////////////////////////////////////////
// Internal utility functions
////////////////////////////////////////////////////////////////////

static LuminaryResult _device_manager_update_scene_entity_on_devices(DeviceManager* device_manager, void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(object);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_update_scene_entity(device_manager->devices[device_id], object, entity));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_handle_device_material_updates(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  ARRAY MaterialUpdate* material_updates;
  __FAILURE_HANDLE(scene_get_list_changes(device_manager->scene_device, (void**) &material_updates, SCENE_ENTITY_MATERIALS));

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(material_updates, &num_updates));

  ARRAY DeviceMaterialCompressed* device_material_updates;
  __FAILURE_HANDLE(array_create(&device_material_updates, sizeof(DeviceMaterialCompressed), num_updates));

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    DeviceMaterialCompressed device_material;
    __FAILURE_HANDLE(device_struct_material_convert(&material_updates[update_id].material, &device_material));

    __FAILURE_HANDLE(array_push(&device_material_updates, &device_material));
  }

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_apply_material_updates(device_manager->devices[device_id], material_updates, device_material_updates));
  }

  uint32_t num_material_updates;
  __FAILURE_HANDLE(array_get_num_elements(material_updates, &num_material_updates));

  for (uint32_t material_update_id = 0; material_update_id < num_material_updates; material_update_id++) {
    __FAILURE_HANDLE(light_tree_update_cache_material(device_manager->light_tree, &material_updates[material_update_id].material));
  }

  __FAILURE_HANDLE(array_destroy(&material_updates));
  __FAILURE_HANDLE(array_destroy(&device_material_updates));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_handle_device_instance_updates(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  ARRAY MeshInstanceUpdate* instance_updates;
  __FAILURE_HANDLE(scene_get_list_changes(device_manager->scene_device, (void**) &instance_updates, SCENE_ENTITY_INSTANCES));

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_apply_instance_updates(device_manager->devices[device_id], instance_updates));
  }

  uint32_t num_instance_updates;
  __FAILURE_HANDLE(array_get_num_elements(instance_updates, &num_instance_updates));

  for (uint32_t instance_update_id = 0; instance_update_id < num_instance_updates; instance_update_id++) {
    __FAILURE_HANDLE(light_tree_update_cache_instance(device_manager->light_tree, &instance_updates[instance_update_id].instance));
  }

  __FAILURE_HANDLE(array_destroy(&instance_updates));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_handle_device_render(DeviceManager* device_manager, DeviceRenderCallbackData* data) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(data);

  Device* device = device_manager->devices[data->common.device_index];

  __FAILURE_HANDLE(sample_time_set_time(device_manager->sample_time, data->common.device_index, device->renderer->last_time));
  __FAILURE_HANDLE(device_continue_render(device, &device_manager->sample_count, data));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_clear_handle_device_render(DeviceManager* device_manager, DeviceRenderCallbackData* data) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(data);

  Device* device = device_manager->devices[data->common.device_index];

  __FAILURE_HANDLE(device_renderer_clear_callback_data(device->renderer));

  return LUMINARY_SUCCESS;
}

static void _device_manager_render_callback(DeviceRenderCallbackData* data) {
  QueueEntry entry;

  entry.name              = "Handle Device Render";
  entry.function          = (QueueEntryFunction) _device_manager_handle_device_render;
  entry.clear_func        = (QueueEntryFunction) _device_manager_clear_handle_device_render;
  entry.args              = (void*) data;
  entry.remove_duplicates = false;

  LuminaryResult result = device_manager_queue_work(data->common.device_manager, &entry);

  if (result) {
    // TODO: Do proper handling.
    error_message("Failed to queue _device_manager_handle_device_render.");
  }
}

static LuminaryResult _device_manager_handle_device_output(DeviceManager* device_manager, DeviceOutputCallbackData* data) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(data);

  Device* device = device_manager->devices[data->common.device_index];

  __FAILURE_HANDLE(device_renderer_get_render_time(device->renderer, data->render_event_id, &data->descriptor.meta_data.time));
  __FAILURE_HANDLE(host_queue_output_copy_from_device(device_manager->host, data->descriptor));

  return LUMINARY_SUCCESS;
}

static void _device_manager_output_callback(DeviceOutputCallbackData* data) {
  // Don't output aborted outputs unless it is the first output.
  if (data->common.device_manager->devices[data->common.device_index]->state_abort && !data->descriptor.meta_data.is_first_output)
    return;

  QueueEntry entry;

  entry.name              = "Handle Device Output";
  entry.function          = (QueueEntryFunction) _device_manager_handle_device_output;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) data;
  entry.remove_duplicates = false;

  LuminaryResult result = device_manager_queue_work(data->common.device_manager, &entry);

  if (result) {
    // TODO: Do proper handling.
    error_message("Failed to queue _device_manager_handle_device_output.");
  }
}

////////////////////////////////////////////////////////////////////
// Queue work functions
////////////////////////////////////////////////////////////////////

static LuminaryResult _device_manager_handle_scene_updates_queue_work(DeviceManager* device_manager, void* args) {
  __CHECK_NULL_ARGUMENT(device_manager);

  LUM_UNUSED(args);

  Scene* scene = device_manager->scene_device;

  SceneEntityCover entity_buffer;
  DeviceSceneEntityCover device_entity_buffer;

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock_all(scene));

  uint32_t device_count;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(device_manager->devices, &device_count));

  SceneDirtyFlags flags;
  __FAILURE_HANDLE_CRITICAL(scene_get_dirty_flags(scene, &flags));

  bool update_device_data_asynchronously = true;

  if (flags & SCENE_DIRTY_FLAG_INTEGRATION) {
    // We will override rendering related data, we need to do this synchronously so the stale
    // render kernels don't read crap and crash.
    // We unset the abort later just before we start rendering again to hide latency coming from the abort procedure.
    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];

      __FAILURE_HANDLE_CRITICAL(device_set_abort(device));
    }

    update_device_data_asynchronously = false;
  }

  uint64_t current_entity = SCENE_ENTITY_GLOBAL_START;
  while (flags && current_entity <= SCENE_ENTITY_GLOBAL_END) {
    if (flags & SCENE_ENTITY_TO_DIRTY(current_entity)) {
      __FAILURE_HANDLE_CRITICAL(scene_get(scene, &entity_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(device_struct_scene_entity_convert(&entity_buffer, &device_entity_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(_device_manager_update_scene_entity_on_devices(device_manager, &device_entity_buffer, current_entity));
    }

    current_entity++;
  }

  if (flags & SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_SKY)) {
    Sky sky;
    __FAILURE_HANDLE_CRITICAL(scene_get(scene, &sky, SCENE_ENTITY_SKY));

    __FAILURE_HANDLE_CRITICAL(sky_lut_update(device_manager->sky_lut, &sky));
    __FAILURE_HANDLE_CRITICAL(device_build_sky_lut(device_manager->devices[device_manager->main_device_index], device_manager->sky_lut));

    __FAILURE_HANDLE_CRITICAL(sky_hdri_update(device_manager->sky_hdri, &sky));
    __FAILURE_HANDLE_CRITICAL(device_build_sky_hdri(device_manager->devices[device_manager->main_device_index], device_manager->sky_hdri));

    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];

      __FAILURE_HANDLE_CRITICAL(device_update_sky_lut(device, device_manager->sky_lut));
      __FAILURE_HANDLE_CRITICAL(device_update_sky_hdri(device, device_manager->sky_hdri));
    }
  }

  if (flags & SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_CLOUD)) {
    Cloud cloud;
    __FAILURE_HANDLE_CRITICAL(scene_get(scene, &cloud, SCENE_ENTITY_CLOUD));

    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];

      __FAILURE_HANDLE_CRITICAL(device_update_cloud_noise(device, &cloud));
    }
  }

  if (flags & SCENE_ENTITY_TO_DIRTY(SCENE_ENTITY_PARTICLES)) {
    Particles particles;
    __FAILURE_HANDLE_CRITICAL(scene_get(scene, &particles, SCENE_ENTITY_PARTICLES));

    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];

      __FAILURE_HANDLE_CRITICAL(device_update_particles(device, &particles));
    }
  }

  if (flags & SCENE_DIRTY_FLAG_OUTPUT) {
    // TODO: Signal main device to output current image again.

    // If only the output is dirty, we need to make sure that the changes are actually uploaded.
    if ((flags & SCENE_DIRTY_FLAG_INTEGRATION) == 0) {
      for (uint32_t device_id = 0; device_id < device_count; device_id++) {
        Device* device = device_manager->devices[device_id];
        __FAILURE_HANDLE_CRITICAL(device_sync_constant_memory(device));
      }
    }
  }

  if (flags & SCENE_DIRTY_FLAG_BUFFERS) {
    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];
      __FAILURE_HANDLE_CRITICAL(device_allocate_work_buffers(device));
    }

    // TODO: Reallocate buffers on all devices
  }

  if (flags & SCENE_DIRTY_FLAG_MATERIALS) {
    __FAILURE_HANDLE_CRITICAL(_device_manager_handle_device_material_updates(device_manager));
  }

  if (flags & SCENE_DIRTY_FLAG_INSTANCES) {
    __FAILURE_HANDLE_CRITICAL(_device_manager_handle_device_instance_updates(device_manager));
  }

  if (flags & SCENE_DIRTY_FLAG_INTEGRATION) {
    const uint32_t previous_light_tree_build_id = device_manager->light_tree->build_id;
    __FAILURE_HANDLE_CRITICAL(
      device_build_light_tree(device_manager->devices[device_manager->main_device_index], device_manager->light_tree));

    if (previous_light_tree_build_id != device_manager->light_tree->build_id) {
      for (uint32_t device_id = 0; device_id < device_count; device_id++) {
        Device* device = device_manager->devices[device_id];
        __FAILURE_HANDLE_CRITICAL(device_update_light_tree_data(device, device_manager->light_tree));
      }
    }

    __FAILURE_HANDLE_CRITICAL(sample_count_reset(&device_manager->sample_count, device_manager->scene_device->settings.max_sample_count));

    // Main device always computes the first samples
    __FAILURE_HANDLE_CRITICAL(
      device_setup_undersampling(device_manager->devices[device_manager->main_device_index], scene->settings.undersampling));
    __FAILURE_HANDLE_CRITICAL(
      device_update_sample_count(device_manager->devices[device_manager->main_device_index], &device_manager->sample_count));

    DeviceRendererQueueArgs render_args;
    render_args.max_depth           = scene->settings.max_ray_depth;
    render_args.render_clouds       = scene->cloud.active && scene->sky.mode == LUMINARY_SKY_MODE_DEFAULT;
    render_args.render_inscattering = scene->sky.aerial_perspective && scene->sky.mode != LUMINARY_SKY_MODE_CONSTANT_COLOR;
    render_args.render_particles    = scene->particles.active;
    render_args.render_volumes      = scene->fog.active || scene->ocean.active;
    render_args.render_lights       = true;
    render_args.shading_mode        = scene->settings.shading_mode;

    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];

      __FAILURE_HANDLE_CRITICAL(device_update_sample_count(device, &device_manager->sample_count));
      __FAILURE_HANDLE_CRITICAL(device_unset_abort(device));
      __FAILURE_HANDLE_CRITICAL(device_clear_lighting_buffers(device));
      __FAILURE_HANDLE_CRITICAL(device_start_render(device, &render_args));
    }
  }

  __FAILURE_HANDLE_CRITICAL(scene_apply_changes(device_manager->scene_device));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock_all(device_manager->scene_device));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

struct DeviceManagerSetOutputPropertiesArgs {
  uint32_t width;
  uint32_t height;
} typedef DeviceManagerSetOutputPropertiesArgs;

static LuminaryResult _device_manager_set_output_properties_clear_work(
  DeviceManager* device_manager, DeviceManagerSetOutputPropertiesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerSetOutputPropertiesArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_set_output_properties_queue_work(
  DeviceManager* device_manager, DeviceManagerSetOutputPropertiesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  Device* device = device_manager->devices[device_manager->main_device_index];

  __FAILURE_HANDLE(device_update_output_properties(device, args->width, args->height));

  return LUMINARY_SUCCESS;
}

struct DeviceManagerAddOutputRequestArgs {
  OutputRequestProperties props;
} typedef DeviceManagerAddOutputRequestArgs;

static LuminaryResult _device_manager_add_output_request_clear_work(
  DeviceManager* device_manager, DeviceManagerAddOutputRequestArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddOutputRequestArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_add_output_request_queue_work(
  DeviceManager* device_manager, DeviceManagerAddOutputRequestArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  Device* device = device_manager->devices[device_manager->main_device_index];

  __FAILURE_HANDLE(device_add_output_request(device, args->props));

  return LUMINARY_SUCCESS;
}

struct DeviceManagerAddMeshesArgs {
  const Mesh** meshes;
  uint32_t num_meshes;
} typedef DeviceManagerAddMeshesArgs;

static LuminaryResult _device_manager_add_meshes_clear(DeviceManager* device_manager, DeviceManagerAddMeshesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddMeshesArgs)));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(Mesh*) * args->num_meshes));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_add_meshes(DeviceManager* device_manager, DeviceManagerAddMeshesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    Device* device = device_manager->devices[device_id];

    for (uint32_t mesh_id = 0; mesh_id < args->num_meshes; mesh_id++) {
      __FAILURE_HANDLE(device_update_mesh(device, args->meshes[mesh_id]));
      __FAILURE_HANDLE(light_tree_update_cache_mesh(device_manager->light_tree, args->meshes[mesh_id]));
    }
  }

  return LUMINARY_SUCCESS;
}

struct DeviceManagerAddTexturesArgs {
  const Texture** textures;
  uint32_t num_textures;
} typedef DeviceManagerAddTexturesArgs;

static LuminaryResult _device_manager_add_textures_clear(DeviceManager* device_manager, DeviceManagerAddTexturesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(host_free(&args->textures));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddTexturesArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_add_textures(DeviceManager* device_manager, DeviceManagerAddTexturesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_add_textures(device_manager->devices[device_id], args->textures, args->num_textures));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_generate_bsdf_luts(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  __FAILURE_HANDLE(device_build_bsdf_lut(device_manager->devices[device_manager->main_device_index], device_manager->bsdf_lut));

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_update_bsdf_lut(device_manager->devices[device_id], device_manager->bsdf_lut));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_compile_kernels(DeviceManager* device_manager, void* args) {
  LUM_UNUSED(args);

  __CHECK_NULL_ARGUMENT(device_manager);

  ////////////////////////////////////////////////////////////////////
  // Load CUBIN
  ////////////////////////////////////////////////////////////////////

  uint64_t info = 0;

  void* cuda_kernels_data;
  int64_t cuda_kernels_data_length;
  ceb_access("cuda_kernels.cubin", &cuda_kernels_data, &cuda_kernels_data_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Failed to load cuda_kernels cubin. Luminary was not compiled correctly.");
  }

  // Tells CUDA that we keep the cubin data unchanged which allows CUDA to not create a copy.
  CUlibraryOption library_option = CU_LIBRARY_BINARY_IS_PRESERVED;

  CUDA_FAILURE_HANDLE(cuLibraryLoadData(&device_manager->cuda_library, cuda_kernels_data, 0, 0, 0, &library_option, 0, 1));

  ////////////////////////////////////////////////////////////////////
  // Gather library content
  ////////////////////////////////////////////////////////////////////

  uint32_t kernel_count;
  CUDA_FAILURE_HANDLE(cuLibraryGetKernelCount(&kernel_count, device_manager->cuda_library));

  CUkernel* kernels;
  __FAILURE_HANDLE(host_malloc(&kernels, sizeof(CUkernel) * kernel_count));

  CUDA_FAILURE_HANDLE(cuLibraryEnumerateKernels(kernels, kernel_count, device_manager->cuda_library));

  for (uint32_t kernel_id = 0; kernel_id < kernel_count; kernel_id++) {
    const char* kernel_name;
    CUDA_FAILURE_HANDLE(cuKernelGetName(&kernel_name, kernels[kernel_id]));

    log_message("CUDA Kernel: %s", kernel_name);
  }

  __FAILURE_HANDLE(host_free(&kernels));

  ////////////////////////////////////////////////////////////////////
  // Compile kernels
  ////////////////////////////////////////////////////////////////////

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_compile_kernels(device_manager->devices[device_id], device_manager->cuda_library));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_initialize_devices(DeviceManager* device_manager, void* args) {
  LUM_UNUSED(args);

  __CHECK_NULL_ARGUMENT(device_manager);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    Device* device = device_manager->devices[device_id];

    __FAILURE_HANDLE(device_load_embedded_data(device));

    DeviceCommonCallbackData callback_data;

    callback_data.device_manager = device_manager;
    callback_data.device_index   = device_id;

    __FAILURE_HANDLE(device_register_callbacks(
      device, (CUhostFn) _device_manager_render_callback, (CUhostFn) _device_manager_output_callback, callback_data));
  }

  __FAILURE_HANDLE(_device_manager_generate_bsdf_luts(device_manager));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_select_main_device(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  uint32_t num_devices;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &num_devices));

  DeviceArch max_arch      = DEVICE_ARCH_UNKNOWN;
  size_t max_memory        = 0;
  uint32_t selected_device = 0;

  for (uint32_t device_id = 0; device_id < num_devices; device_id++) {
    const Device* device = device_manager->devices[device_id];

    if ((device->properties.arch > max_arch) || (device->properties.arch == max_arch && device->properties.memory_size > max_memory)) {
      max_arch   = device->properties.arch;
      max_memory = device->properties.memory_size;

      selected_device = device_id;
    }
  }

  device_manager->main_device_index = selected_device;

  __FAILURE_HANDLE(device_register_as_main(device_manager->devices[device_manager->main_device_index]));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

LuminaryResult device_manager_create(DeviceManager** _device_manager, Host* host) {
  __CHECK_NULL_ARGUMENT(_device_manager);
  __CHECK_NULL_ARGUMENT(host);

  DeviceManager* device_manager;
  __FAILURE_HANDLE(host_malloc(&device_manager, sizeof(DeviceManager)));
  memset(device_manager, 0, sizeof(DeviceManager));

  device_manager->host = host;

  __FAILURE_HANDLE(scene_create(&device_manager->scene_device));

  int32_t device_count;
  CUDA_FAILURE_HANDLE(cuDeviceGetCount(&device_count));

  __FAILURE_HANDLE(array_create(&device_manager->devices, sizeof(Device*), device_count));

  for (int32_t device_id = 0; device_id < device_count; device_id++) {
    Device* device;
    __FAILURE_HANDLE(device_create(&device, device_id));

    __FAILURE_HANDLE(array_push(&device_manager->devices, &device));
  }

  __FAILURE_HANDLE(light_tree_create(&device_manager->light_tree));
  __FAILURE_HANDLE(sky_lut_create(&device_manager->sky_lut));
  __FAILURE_HANDLE(sky_hdri_create(&device_manager->sky_hdri));
  __FAILURE_HANDLE(bsdf_lut_create(&device_manager->bsdf_lut));
  __FAILURE_HANDLE(sample_time_create(&device_manager->sample_time));

  ////////////////////////////////////////////////////////////////////
  // Select main device
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(_device_manager_select_main_device(device_manager));

  ////////////////////////////////////////////////////////////////////
  // Create work queue
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(thread_create(&device_manager->work_thread));
  __FAILURE_HANDLE(queue_create(&device_manager->work_queue, sizeof(QueueEntry), DEVICE_MANAGER_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&device_manager->ringbuffer, DEVICE_MANAGER_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&device_manager->queue_wall_time));

  __FAILURE_HANDLE(device_manager_start_queue(device_manager));

  ////////////////////////////////////////////////////////////////////
  // Queue setup functions
  ////////////////////////////////////////////////////////////////////

  QueueEntry entry;

  entry.name              = "Kernel compilation";
  entry.function          = (QueueEntryFunction) _device_manager_compile_kernels;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) 0;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  entry.name              = "Device initialization";
  entry.function          = (QueueEntryFunction) _device_manager_initialize_devices;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) 0;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  ////////////////////////////////////////////////////////////////////
  // Finalize
  ////////////////////////////////////////////////////////////////////

  *_device_manager = device_manager;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_start_queue(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  bool device_thread_is_running;
  __FAILURE_HANDLE(thread_is_running(device_manager->work_thread, &device_thread_is_running));

  if (device_thread_is_running)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(queue_set_is_blocking(device_manager->work_queue, true));
  __FAILURE_HANDLE(thread_start(device_manager->work_thread, (ThreadMainFunc) _device_manager_queue_worker, device_manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_queue_work(DeviceManager* device_manager, QueueEntry* entry) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(entry);

  if (device_manager->is_shutdown) {
    if (entry->clear_func) {
      __FAILURE_HANDLE(entry->clear_func(device_manager, entry->args));
    }

    return LUMINARY_SUCCESS;
  }

  if (entry->remove_duplicates) {
    bool entry_already_queued = false;
    __FAILURE_HANDLE(queue_push_unique(
      device_manager->work_queue, entry, (LuminaryEqOp) _device_manager_queue_entry_equal_operator, &entry_already_queued));

    if (entry_already_queued) {
      if (entry->clear_func) {
        __FAILURE_HANDLE(entry->clear_func(device_manager, entry->args));
      }

      return LUMINARY_SUCCESS;
    }
  }
  else {
    __FAILURE_HANDLE(queue_push(device_manager->work_queue, entry));
  }

  bool device_thread_is_running;
  __FAILURE_HANDLE(thread_is_running(device_manager->work_thread, &device_thread_is_running));

  // If the device thread is not running, execute on current thread.
  if (!device_thread_is_running) {
    __FAILURE_HANDLE(_device_manager_queue_worker(device_manager));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_shutdown_queue(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  bool device_thread_is_running;
  __FAILURE_HANDLE(thread_is_running(device_manager->work_thread, &device_thread_is_running));

  if (!device_thread_is_running)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(queue_set_is_blocking(device_manager->work_queue, false));
  __FAILURE_HANDLE(thread_join(device_manager->work_thread));
  __FAILURE_HANDLE(thread_get_last_result(device_manager->work_thread));

  // There could still be some unfinished work that was queued during shutdown, so execute that now.
  __FAILURE_HANDLE(_device_manager_queue_worker(device_manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_update_scene(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  __FAILURE_HANDLE(scene_propagate_changes(device_manager->scene_device, device_manager->host->scene_host));

  QueueEntry entry;

  entry.name              = "Update device scene";
  entry.function          = (QueueEntryFunction) _device_manager_handle_scene_updates_queue_work;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) 0;
  entry.remove_duplicates = true;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_set_output_properties(DeviceManager* device_manager, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(device_manager);

  DeviceManagerSetOutputPropertiesArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerSetOutputPropertiesArgs), (void**) &args));

  args->width  = width;
  args->height = height;

  QueueEntry entry;

  entry.name              = "Set Output Properties";
  entry.function          = (QueueEntryFunction) _device_manager_set_output_properties_queue_work;
  entry.clear_func        = (QueueEntryFunction) _device_manager_set_output_properties_clear_work;
  entry.args              = args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_add_output_request(DeviceManager* device_manager, OutputRequestProperties properties) {
  __CHECK_NULL_ARGUMENT(device_manager);

  DeviceManagerAddOutputRequestArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddOutputRequestArgs), (void**) &args));

  args->props = properties;

  QueueEntry entry;

  entry.name              = "Add output request";
  entry.function          = (QueueEntryFunction) _device_manager_add_output_request_clear_work;
  entry.clear_func        = (QueueEntryFunction) _device_manager_add_output_request_queue_work;
  entry.args              = args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_add_meshes(DeviceManager* device_manager, const Mesh** meshes, uint32_t num_meshes) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(meshes);

  DeviceManagerAddMeshesArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddMeshesArgs), (void**) &args));

  args->num_meshes = num_meshes;

  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(Mesh*) * num_meshes, (void**) &args->meshes));

  memcpy(args->meshes, meshes, sizeof(Mesh*) * num_meshes);

  QueueEntry entry;

  entry.name              = "Add meshes";
  entry.function          = (QueueEntryFunction) _device_manager_add_meshes;
  entry.clear_func        = (QueueEntryFunction) _device_manager_add_meshes_clear;
  entry.args              = args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_add_textures(DeviceManager* device_manager, const Texture** textures, uint32_t num_textures) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(textures);

  // We need to make a copy because the caller could add textures which could cause a reallocation of the textures array in the host which
  // would invalidate our pointer.
  Texture** textures_copy;
  __FAILURE_HANDLE(host_malloc(&textures_copy, sizeof(Texture*) * num_textures));

  memcpy(textures_copy, textures, sizeof(Texture*) * num_textures);

  DeviceManagerAddTexturesArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddTexturesArgs), (void**) &args));

  args->textures     = (const Texture**) textures_copy;
  args->num_textures = num_textures;

  QueueEntry entry;

  entry.name              = "Add textures";
  entry.function          = (QueueEntryFunction) _device_manager_add_textures;
  entry.clear_func        = (QueueEntryFunction) _device_manager_add_textures_clear;
  entry.args              = (void*) args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_destroy(DeviceManager** device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(*device_manager);

  (*device_manager)->is_shutdown = true;

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements((*device_manager)->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_set_abort((*device_manager)->devices[device_id]));
  }

  __FAILURE_HANDLE(device_manager_shutdown_queue(*device_manager));

  __FAILURE_HANDLE(wall_time_destroy(&(*device_manager)->queue_wall_time));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*device_manager)->ringbuffer));
  __FAILURE_HANDLE(queue_destroy(&(*device_manager)->work_queue));

  __FAILURE_HANDLE(thread_destroy(&(*device_manager)->work_thread));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_destroy(&((*device_manager)->devices[device_id])));
  }

  __FAILURE_HANDLE(array_destroy(&(*device_manager)->devices));

  __FAILURE_HANDLE(sky_lut_destroy(&(*device_manager)->sky_lut));
  __FAILURE_HANDLE(sky_hdri_destroy(&(*device_manager)->sky_hdri));
  __FAILURE_HANDLE(light_tree_destroy(&(*device_manager)->light_tree));
  __FAILURE_HANDLE(bsdf_lut_destroy(&(*device_manager)->bsdf_lut));
  __FAILURE_HANDLE(sample_time_destroy(&(*device_manager)->sample_time));

  __FAILURE_HANDLE(scene_destroy(&(*device_manager)->scene_device));

  __FAILURE_HANDLE(host_free(device_manager));

  return LUMINARY_SUCCESS;
}
