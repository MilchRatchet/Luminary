#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device/device_manager.h"
#include "dictionary.h"
#include "host_output_handler.h"
#include "mesh.h"
#include "output_descriptor.h"
#include "queue_worker.h"
#include "scene.h"
#include "texture.h"
#include "thread.h"
#include "utils.h"
#include "wavefront.h"

struct HostLoadObjArgs {
  Path* path;
  WavefrontArguments* wavefront_args;
} typedef HostLoadObjArgs;

struct LuminaryHost {
  DeviceManager* device_manager;
  QueueWorker* queue_worker_main;
  Queue* work_queue;
  ARRAY QueueWorker** queue_worker_secondary;
  Queue* secondary_work_queue;
  RingBuffer* ringbuffer;
  ARRAY Mesh** meshes;
  ARRAY Texture** textures;
  bool enable_output;
  Scene* scene_host;
  Scene* scene_caller;
  OutputHandler* output_handler;
  Dictionary* mesh_instance_name_dict;
  Dictionary* material_name_dict;
  Dictionary* mesh_name_dict;
  Dictionary* texture_name_dict;
  bool scene_locked_by_caller;
  ARRAY HostLoadObjArgs* loaded_obj_files;
} typedef LuminaryHost;

LuminaryResult host_queue_output_copy_from_device(Host* host, OutputDescriptor descriptor);
LuminaryResult host_load_obj_file(Host* host, Path* path, WavefrontArguments* wavefront_args);
LuminaryResult host_update_scene(Host* host);
LuminaryResult host_add_meshes(Host* host, ARRAY Mesh** meshes, Dictionary* mesh_name_dict);
LuminaryResult host_add_textures(Host* host, ARRAY Texture** textures, Dictionary* texture_name_dict);

#endif /* LUMINARY_INTERNAL_HOST_H */
