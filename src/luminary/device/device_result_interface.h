#ifndef LUMINARY_DEVICE_RESULT_INTERFACE_H
#define LUMINARY_DEVICE_RESULT_INTERFACE_H

#include "device_memory.h"
#include "device_utils.h"
#include "mutex.h"

struct Device typedef Device;

struct DeviceResultEntry {
  uint32_t sample_count;
  STAGING void* direct_lighting_results;
  STAGING void* indirect_lighting_results_red[MAX_NUM_INDIRECT_BUCKETS];
  STAGING void* indirect_lighting_results_green[MAX_NUM_INDIRECT_BUCKETS];
  STAGING void* indirect_lighting_results_blue[MAX_NUM_INDIRECT_BUCKETS];
  CUevent available_event;
  uint32_t consumer_event_id;
  bool queued;
} typedef DeviceResultEntry;

struct DeviceResultMap {
  uint32_t device_id;
  uint32_t allocation_id;
} typedef DeviceResultMap;

struct DeviceResultEvent {
  CUevent event;
  bool assigned;
} typedef DeviceResultEvent;

struct DeviceResultInterface {
  Mutex* mutex;
  uint32_t pixel_count;
  ARRAY DeviceResultMap* queued_results;
  ARRAY DeviceResultEntry* allocated_results[LUMINARY_MAX_NUM_DEVICES];
  ARRAY DeviceResultEvent* allocated_events;
} typedef DeviceResultInterface;

LuminaryResult device_result_interface_create(DeviceResultInterface** interface);
DEVICE_CTX_FUNC LuminaryResult device_result_interface_set_pixel_count(DeviceResultInterface* interface, uint32_t width, uint32_t height);
DEVICE_CTX_FUNC LuminaryResult device_result_interface_queue_result(DeviceResultInterface* interface, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_result_interface_gather_results(DeviceResultInterface* interface, Device* device);
LuminaryResult device_result_interface_destroy(DeviceResultInterface** interface);

#endif /* LUMINARY_DEVICE_RESULT_INTERFACE_H */
