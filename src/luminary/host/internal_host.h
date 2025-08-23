#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device/device_manager.h"
#include "host_output_handler.h"
#include "mesh.h"
#include "mutex.h"
#include "output_descriptor.h"
#include "queue_worker.h"
#include "scene.h"
#include "texture.h"
#include "thread.h"
#include "utils.h"

struct LuminaryHost {
  DeviceManager* device_manager;
  QueueWorker* queue_worker_main;
  Queue* work_queue;
  RingBuffer* ringbuffer;
  ARRAY Mesh** meshes;
  ARRAY Texture** textures;
  bool enable_output;
  Scene* scene_host;
  Scene* scene_caller;
  OutputHandler* output_handler;
} typedef LuminaryHost;

LuminaryResult host_queue_output_copy_from_device(Host* host, OutputDescriptor descriptor);

#endif /* LUMINARY_INTERNAL_HOST_H */
