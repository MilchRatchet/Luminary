#include "device_manager.h"

#include "ceb.h"
#include "device_structs.h"
#include "device_utils.h"
#include "host/internal_host.h"
#include "internal_error.h"
#include "scene.h"

#define DEVICE_MANAGER_RINGBUFFER_SIZE (0x100000ull)
#define DEVICE_MANAGER_QUEUE_SIZE (0x100ull)

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

    __FAILURE_HANDLE(wall_time_stop(device_manager->queue_wall_time));
    __FAILURE_HANDLE(wall_time_set_string(device_manager->queue_wall_time, (const char*) 0));
  }

  return LUMINARY_SUCCESS;
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

  ARRAY MaterialUpdate* list_updates;
  __FAILURE_HANDLE(scene_get_list_changes(device_manager->scene_device, (void**) &list_updates, SCENE_ENTITY_MATERIALS));

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(list_updates, &num_updates));

  ARRAY DeviceMaterialCompressed* device_list_updates;
  __FAILURE_HANDLE(array_create(&device_list_updates, sizeof(DeviceMaterialCompressed), num_updates));

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    DeviceMaterialCompressed device_material;
    __FAILURE_HANDLE(device_struct_material_convert(&list_updates[update_id].material, &device_material));

    __FAILURE_HANDLE(array_push(&device_list_updates, &device_material));
  }

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_apply_material_updates(device_manager->devices[device_id], list_updates, device_list_updates));
  }

  __FAILURE_HANDLE(array_destroy(&list_updates));
  __FAILURE_HANDLE(array_destroy(&device_list_updates));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Queue work functions
////////////////////////////////////////////////////////////////////

#define DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE (4096)

struct DeviceManagerHandleSceneUpdatesArgs {
  void* entity_buffer;
  void* device_entity_buffer;
} typedef DeviceManagerHandleSceneUpdatesArgs;

static LuminaryResult _device_manager_handle_scene_updates_queue_work(
  DeviceManager* device_manager, DeviceManagerHandleSceneUpdatesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock_all(device_manager->scene_device));

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  SceneDirtyFlags flags;
  __FAILURE_HANDLE_CRITICAL(scene_get_dirty_flags(device_manager->scene_device, &flags));

  bool update_device_data_asynchronously = true;

  if (flags & SCENE_DIRTY_FLAG_INTEGRATION) {
    // We will override rendering related data, we need to do this synchronously so the stale
    // render kernels don't read crap and crash,
    update_device_data_asynchronously = false;

    __FAILURE_HANDLE(sample_count_reset(&device_manager->sample_count, device_manager->scene_device->settings.max_sample_count));

    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];
      __FAILURE_HANDLE(sample_count_get_slice(&device_manager->sample_count, 32, &device->sample_count));
      // TODO: Signal all devices to restart integration
    }
  }

  uint64_t current_entity = SCENE_ENTITY_GLOBAL_START;
  while (flags && current_entity <= SCENE_ENTITY_GLOBAL_END) {
    if (flags & SCENE_ENTITY_TO_DIRTY(current_entity)) {
      __FAILURE_HANDLE_CRITICAL(scene_get(device_manager->scene_device, args->entity_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(device_struct_scene_entity_convert(args->entity_buffer, args->device_entity_buffer, current_entity));
      __FAILURE_HANDLE_CRITICAL(_device_manager_update_scene_entity_on_devices(device_manager, args->device_entity_buffer, current_entity));
    }

    current_entity++;
  }

  if (flags & SCENE_DIRTY_FLAG_MATERIALS) {
    __FAILURE_HANDLE_CRITICAL(_device_manager_handle_device_material_updates(device_manager));
  }

  current_entity = SCENE_ENTITY_LIST_START;
  while (flags && current_entity <= SCENE_ENTITY_LIST_END) {
    if (flags & SCENE_ENTITY_TO_DIRTY(current_entity)) {
      // TODO: Update entity on devices
    }

    current_entity++;
  }

  if (flags & SCENE_DIRTY_FLAG_OUTPUT) {
    // TODO: Signal main device to output current image again.
  }

  if (flags & SCENE_DIRTY_FLAG_BUFFERS) {
    for (uint32_t device_id = 0; device_id < device_count; device_id++) {
      Device* device = device_manager->devices[device_id];
      __FAILURE_HANDLE(device_allocate_work_buffers(device));
    }

    // TODO: Reallocate buffers on all devices
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock_all(device_manager->scene_device));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  // Cleanup
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerHandleSceneUpdatesArgs)));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_upload_meshes(DeviceManager* device_manager, void* args) {
  LUM_UNUSED(args);

  __CHECK_NULL_ARGUMENT(device_manager);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_upload_meshes(device_manager->devices[device_id], (const ARRAY DeviceMesh**) device_manager->meshes));
  }

  return LUMINARY_SUCCESS;
}

struct DeviceManagerAddTexturesArgs {
  const Texture** textures;
  uint32_t num_textures;
} typedef DeviceManagerAddTexturesArgs;

static LuminaryResult _device_manager_add_textures(DeviceManager* device_manager, DeviceManagerAddTexturesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_add_textures(device_manager->devices[device_id], args->textures, args->num_textures));
  }

  // Cleanup
  __FAILURE_HANDLE(host_free(&args->textures));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerAddTexturesArgs)));

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

    info_message("CUDA Kernel: %s", kernel_name);
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
    __FAILURE_HANDLE(device_load_embedded_data(device_manager->devices[device_id]));
  }

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
  __FAILURE_HANDLE(array_create(&device_manager->meshes, sizeof(DeviceMesh*), 4));

  int32_t device_count;
  CUDA_FAILURE_HANDLE(cuDeviceGetCount(&device_count));

  __FAILURE_HANDLE(array_create(&device_manager->devices, sizeof(Device*), device_count));

  for (int32_t device_id = 0; device_id < device_count; device_id++) {
    Device* device;
    __FAILURE_HANDLE(device_create(&device, device_id));

    __FAILURE_HANDLE(array_push(&device_manager->devices, &device));
  }

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

  entry.name     = "Kernel compilation";
  entry.function = (QueueEntryFunction) _device_manager_compile_kernels;
  entry.args     = (void*) 0;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  entry.name     = "Device initialization";
  entry.function = (QueueEntryFunction) _device_manager_initialize_devices;
  entry.args     = (void*) 0;

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

  __FAILURE_HANDLE(queue_push(device_manager->work_queue, entry));

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

LuminaryResult device_manager_destroy(DeviceManager** device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(*device_manager);

  __FAILURE_HANDLE(device_manager_shutdown_queue(*device_manager));

  __FAILURE_HANDLE(wall_time_destroy(&(*device_manager)->queue_wall_time));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*device_manager)->ringbuffer));
  __FAILURE_HANDLE(queue_destroy(&(*device_manager)->work_queue));

  __FAILURE_HANDLE(thread_destroy(&(*device_manager)->work_thread));

  uint32_t mesh_count;
  __FAILURE_HANDLE(array_get_num_elements((*device_manager)->meshes, &mesh_count));

  for (uint32_t mesh_id = 0; mesh_id < mesh_count; mesh_id++) {
    __FAILURE_HANDLE(device_mesh_destroy(&(*device_manager)->meshes[mesh_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*device_manager)->meshes));

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements((*device_manager)->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_destroy(&((*device_manager)->devices[device_id])));
  }

  __FAILURE_HANDLE(array_destroy(&(*device_manager)->devices));

  __FAILURE_HANDLE(scene_destroy(&(*device_manager)->scene_device));

  __FAILURE_HANDLE(host_free(device_manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_update_scene(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  __FAILURE_HANDLE(scene_propagate_changes(device_manager->scene_device, device_manager->host->scene_host));

  DeviceManagerHandleSceneUpdatesArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerHandleSceneUpdatesArgs), (void**) &args));

  __FAILURE_HANDLE(
    ringbuffer_allocate_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE, &args->entity_buffer));
  __FAILURE_HANDLE(ringbuffer_allocate_entry(
    device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE, &args->device_entity_buffer));

  QueueEntry entry;

  entry.name     = "Update device scene";
  entry.function = (QueueEntryFunction) _device_manager_handle_scene_updates_queue_work;
  entry.args     = args;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_add_meshes(DeviceManager* device_manager, const Mesh** meshes, uint32_t num_meshes) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(meshes);

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    DeviceMesh* device_mesh;
    __FAILURE_HANDLE(device_mesh_create(&device_mesh, meshes[mesh_id]));

    __FAILURE_HANDLE(array_push(&device_manager->meshes, &device_mesh));
  }

  QueueEntry entry;

  entry.name     = "Upload meshes";
  entry.function = (QueueEntryFunction) _device_manager_upload_meshes;
  entry.args     = (void*) 0;

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

  entry.name     = "Add textures";
  entry.function = (QueueEntryFunction) _device_manager_add_textures;
  entry.args     = (void*) args;

  __FAILURE_HANDLE(device_manager_queue_work(device_manager, &entry));

  return LUMINARY_SUCCESS;
}
