#ifndef LUMINARY_DEVICE_MANAGER_H
#define LUMINARY_DEVICE_MANAGER_H

#include "device.h"
#include "device_bsdf.h"
#include "device_library.h"
#include "device_light.h"
#include "device_result_interface.h"
#include "device_sampletime.h"
#include "device_sky.h"
#include "queue_worker.h"

struct DeviceManagerCreateInfo {
  uint32_t device_mask;
} typedef DeviceManagerCreateInfo;

struct DeviceManager {
  Host* host;
  bool is_shutdown;
  Scene* scene_device;
  SampleCountSlice sample_count;
  ARRAY Device** devices;
  uint32_t main_device_index;
  DeviceLibrary* library;
  Queue* work_queue;
  QueueWorker* queue_worker_main;
  RingBuffer* ringbuffer;
  LightTree* light_tree;
  SkyLUT* sky_lut;
  SkyHDRI* sky_hdri;
  SkyStars* sky_stars;
  BSDFLUT* bsdf_lut;
  SampleTime* sample_time;
  DeviceResultInterface* result_interface;
} typedef DeviceManager;

LuminaryResult device_manager_create(DeviceManager** device_manager, Host* host, DeviceManagerCreateInfo info);
LuminaryResult device_manager_enable_device(DeviceManager* device_manager, uint32_t device_id, bool enable);
LuminaryResult device_manager_update_scene(DeviceManager* device_manager);
LuminaryResult device_manager_set_output_properties(DeviceManager* device_manager, LuminaryOutputProperties properties);
LuminaryResult device_manager_add_output_request(DeviceManager* device_manager, OutputRequestProperties properties);
LuminaryResult device_manager_add_meshes(DeviceManager* device_manager, const Mesh** meshes, uint32_t num_meshes);
LuminaryResult device_manager_add_textures(DeviceManager* device_manager, const Texture** textures, uint32_t num_textures);
LuminaryResult device_manager_start_queue(DeviceManager* device_manager);
LuminaryResult device_manager_queue_work(DeviceManager* device_manager, QueueEntry* entry);
LuminaryResult device_manager_shutdown_queue(DeviceManager* device_manager);
LuminaryResult device_manager_destroy(DeviceManager** device_manager);

#endif /* LUMINARY_DEVICE_MANAGER */
