#ifndef LUMINARY_DEVICE_MANAGER_H
#define LUMINARY_DEVICE_MANAGER_H

// For multi GPU support we will use cudaSetDevice and cudaGetDevice.
// These are not multithreading compatible. Hence all device work
// needs to be handled within one queue instead of using per device queues.
// Use cudaLaunchHostFunc to queue more work on demand. Since this func can't do CUDA calls
// it must be used to queue the next bounce.
// So the queue thread launches the kernels, then waits in the queue in case of additional work coming in.
// Once the kernels have reached a certain point, the kernels are queued again. This gives a consistent time
// window for other work to be queued.

// For shutdown of the queues I need to have a specific order:
// The API thread calls shutdown, this creates a device shutdown task on the host queue
// This means all currently queued host work will be done, then the device shutdown will be initialized
// Then all the devices get to finish their queue work. Then the host finishes whatever work
// the devices had queued for the host. Then the host finally shuts down.
// There is one tricky part:
// If a device task queues a host task that in return queues a device task then
// this will not work. So when finishing the last host tasks after shutting down the
// devices, I need to mark the devices as offline so that these host tasks know to not
// queue a device task.

#include "device.h"
#include "thread.h"

struct DeviceManager {
  Host* host;
  Scene* scene_device;
  SampleCountSlice sample_count;
  ARRAY Device** devices;
  uint32_t main_device_index;
  CUlibrary cuda_library;
  Queue* work_queue;
  RingBuffer* ringbuffer;
  WallTime* queue_wall_time;
  Thread* work_thread;
} typedef DeviceManager;

LuminaryResult device_manager_create(DeviceManager** device_manager, Host* host);
LuminaryResult device_manager_update_scene(DeviceManager* device_manager);
LuminaryResult device_manager_add_meshes(DeviceManager* device_manager, const Mesh** meshes, uint32_t num_meshes);
LuminaryResult device_manager_add_textures(DeviceManager* device_manager, const Texture** textures, uint32_t num_textures);
LuminaryResult device_manager_start_queue(DeviceManager* device_manager);
LuminaryResult device_manager_queue_work(DeviceManager* device_manager, QueueEntry* entry);
LuminaryResult device_manager_shutdown_queue(DeviceManager* device_manager);
LuminaryResult device_manager_destroy(DeviceManager** device_manager);

#endif /* LUMINARY_DEVICE_MANAGER */
