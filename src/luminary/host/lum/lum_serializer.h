#ifndef LUMINARY_LUM_SERIALIZER_H
#define LUMINARY_LUM_SERIALIZER_H

#include "utils.h"

struct LumSerializer {
  char* serialized_data;
  size_t serialized_data_allocated_size;
  size_t serialized_data_length;
} typedef LumSerializer;

LuminaryResult lum_serializer_create(LumSerializer** serializer);
LuminaryResult lum_serializer_serialize(LumSerializer* serializer, Host* host);
LuminaryResult lum_serializer_store(LumSerializer* serializer, Path* path);
LuminaryResult lum_serializer_destroy(LumSerializer** serializer);

#endif /* LUMINARY_LUM_SERIALIZER_H */
