#include "device_manager.h"

#include "device_structs.h"
#include "device_utils.h"
#include "host/internal_host.h"
#include "internal_error.h"
#include "scene.h"

#define DEVICE_MANAGER_RINGBUFFER_SIZE (0x10000ull)
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
// Queue work functions
////////////////////////////////////////////////////////////////////

#define DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE (1024)

struct DeviceManagerHandleSceneUpdatesArgs {
  void* entity_buffer;
  void* device_entity_buffer;
} typedef DeviceManagerHandleSceneUpdatesArgs;

static LuminaryResult _device_manager_handle_scene_updates_queue_work(
  DeviceManager* device_manager, DeviceManagerHandleSceneUpdatesArgs* args) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(scene_lock(device_manager->host->scene_internal));

  SceneDirtyFlags flags;
  __FAILURE_HANDLE(scene_get_dirty_flags(device_manager->host->scene_internal, &flags));

  // SCENE_ENTITY_SAMPLE_COUNT is handled differently
  uint64_t current_entity = SCENE_ENTITY_SETTINGS;
  while (flags && current_entity < SCENE_ENTITY_COUNT) {
    if (flags & SCENE_ENTITY_TO_DIRTY(current_entity)) {
      __FAILURE_HANDLE(scene_get(device_manager->host->scene_internal, args->entity_buffer, current_entity));
      __FAILURE_HANDLE(device_struct_scene_entity_convert(args->entity_buffer, args->device_entity_buffer, current_entity));
      // TODO: Update entity on devices
    }

    current_entity = current_entity << 1;
  }

  if (flags & SCENE_DIRTY_FLAG_OUTPUT) {
    // TODO: Signal main device to output current image again.
  }

  if (flags & SCENE_DIRTY_FLAG_INTEGRATION) {
    // TODO: Signal all devices to restart integration
  }

  if (flags & SCENE_DIRTY_FLAG_BUFFERS) {
    // TODO: Reallocate buffers on all devices
  }

  __FAILURE_HANDLE(scene_unlock(device_manager->host->scene_internal));

  // Cleanup
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, sizeof(DeviceManagerHandleSceneUpdatesArgs)));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE));
  __FAILURE_HANDLE(ringbuffer_release_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_manager_compile_kernels(DeviceManager* device_manager, void* args) {
  LUM_UNUSED(args);

  __CHECK_NULL_ARGUMENT(device_manager);

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements(device_manager->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_compile_kernels(device_manager->devices[device_id]));
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

LuminaryResult device_manager_create(DeviceManager** _device_manager) {
  __CHECK_NULL_ARGUMENT(_device_manager);

  DeviceManager* device_manager;
  __FAILURE_HANDLE(host_malloc(&device_manager, sizeof(DeviceManager)));
  memset(device_manager, 0, sizeof(DeviceManager));

  int device_count;
  CUDA_FAILURE_HANDLE(cudaGetDeviceCount(&device_count));

  __FAILURE_HANDLE(array_create(&device_manager->devices, sizeof(Device*), device_count));

  for (int device_id = 0; device_id < device_count; device_id++) {
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

  __FAILURE_HANDLE(thread_start(device_manager->work_thread, (ThreadMainFunc) _device_manager_queue_worker, device_manager));

  ////////////////////////////////////////////////////////////////////
  // Queue setup functions
  ////////////////////////////////////////////////////////////////////

  QueueEntry entry;

  entry.name     = "Kernel compilation";
  entry.function = (QueueEntryFunction) _device_manager_compile_kernels;
  entry.args     = (void*) 0;

  __FAILURE_HANDLE(queue_push(device_manager->work_queue, &entry));

  ////////////////////////////////////////////////////////////////////
  // Finalize
  ////////////////////////////////////////////////////////////////////

  *_device_manager = device_manager;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_destroy(DeviceManager** device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);
  __CHECK_NULL_ARGUMENT(*device_manager);

  __FAILURE_HANDLE(queue_flush_blocking((*device_manager)->work_queue));

  __FAILURE_HANDLE(thread_join((*device_manager)->work_thread));

  __FAILURE_HANDLE(thread_get_last_result((*device_manager)->work_thread));

  __FAILURE_HANDLE(wall_time_destroy(&(*device_manager)->queue_wall_time));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*device_manager)->ringbuffer));
  __FAILURE_HANDLE(queue_destroy(&(*device_manager)->work_queue));

  __FAILURE_HANDLE(thread_destroy(&(*device_manager)->work_thread));

  uint32_t device_count;
  __FAILURE_HANDLE(array_get_num_elements((*device_manager)->devices, &device_count));

  for (uint32_t device_id = 0; device_id < device_count; device_id++) {
    __FAILURE_HANDLE(device_destroy(&((*device_manager)->devices[device_id])));
  }

  __FAILURE_HANDLE(array_destroy(&(*device_manager)->devices));

  __FAILURE_HANDLE(host_free(device_manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_manager_handle_scene_updates(DeviceManager* device_manager) {
  __CHECK_NULL_ARGUMENT(device_manager);

  DeviceManagerHandleSceneUpdatesArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(device_manager->ringbuffer, sizeof(DeviceManagerHandleSceneUpdatesArgs), (void**) &args));

  __FAILURE_HANDLE(
    ringbuffer_allocate_entry(device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE, &args->entity_buffer));
  __FAILURE_HANDLE(ringbuffer_allocate_entry(
    device_manager->ringbuffer, DEVICE_MANAGER_HANDLE_SCENE_UPDATES_WORK_BUFFER_SIZE, &args->device_entity_buffer));

  QueueEntry entry;

  entry.name     = "Device handle scene updates";
  entry.function = (QueueEntryFunction) _device_manager_handle_scene_updates_queue_work;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(device_manager->work_queue, &entry));

  return LUMINARY_SUCCESS;
}
