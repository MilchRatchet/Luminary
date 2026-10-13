#ifndef LUMINARY_OUTPUT_DESCRIPTOR_H
#define LUMINARY_OUTPUT_DESCRIPTOR_H

#include "utils.h"
#include "vault_object.h"

struct OutputDescriptor {
  bool is_recurring_output;
  uint32_t device_index;
  uint64_t output_request_id;
  struct {
    uint32_t width;
    uint32_t height;
    uint32_t sample_count;
    bool is_first_output;
    float time;
  } meta_data;
  VaultHandle* data_handle;
} typedef OutputDescriptor;

#endif /* LUMINARY_OUTPUT_DESCRIPTOR_H */
