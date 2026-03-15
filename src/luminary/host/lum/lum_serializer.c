#include "lum_serializer.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "host/internal_host.h"
#include "internal_error.h"
#include "internal_path.h"
#include "lum_builtins.h"

static LuminaryResult _lum_serializer_serialize_literal(LumSerializer* serializer, LumBuiltinType type, const void* data);
static LuminaryResult _lum_serializer_serialize_struct(LumSerializer* serializer, LumBuiltinType type, const void* data);

LuminaryResult lum_serializer_create(LumSerializer** serializer) {
  __CHECK_NULL_ARGUMENT(serializer);

  __FAILURE_HANDLE(host_malloc(serializer, sizeof(LumSerializer)));
  memset(*serializer, 0, sizeof(LumSerializer));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_write(LumSerializer* serializer, const char* format, ...) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(format);

  va_list args;
  va_start(args, format);

  const size_t offset = serializer->serialized_data_length;

  int size = vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args);

  if (size + offset > serializer->serialized_data_allocated_size) {
    serializer->serialized_data_allocated_size = (size + offset) * 2;
    __FAILURE_HANDLE(host_realloc(&serializer->serialized_data, serializer->serialized_data_allocated_size));

    vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args);
  }

  serializer->serialized_data_length += size;

  va_end(args);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_literal(LumSerializer* serializer, LumBuiltinType type, const void* data) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);

  switch (type) {
    case LUM_BUILTIN_TYPE_UINT: {
      uint32_t value = *(uint32_t*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%u", value));
    } break;
    case LUM_BUILTIN_TYPE_BOOL: {
      bool value = *(bool*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%s", value ? "true" : "false"));
    } break;
    case LUM_BUILTIN_TYPE_FLOAT: {
      float value = *(float*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%f", value));
    } break;
    case LUM_BUILTIN_TYPE_ENUM: {
      // TODO: ENUMS
    } break;
    default: {
      const uint32_t member_count = lum_builtin_types_member_counts[type];

      if (member_count == 0)
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' cannot be written as a literal.");

      // Type is struct
      __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, type, data));
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_struct(LumSerializer* serializer, LumBuiltinType type, const void* data) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);

  // TODO: Addressables, probably a separate function
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s]\n", lum_builtin_types_strings[type]));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "{\n"));

  uint32_t member_count               = lum_builtin_types_member_counts[type];
  const LumBuiltinTypeMember* members = lum_builtin_types_member[type];

  for (uint32_t member_id = 0; member_id < member_count; member_id++) {
    const LumBuiltinTypeMember* member = members + member_id;

    if (member->max_version < LUM_VERSION_CURRENT)
      continue;

    __FAILURE_HANDLE(_lum_serializer_write(serializer, "\t.%s = \n", member->name));
    __FAILURE_HANDLE(_lum_serializer_serialize_literal(serializer, member->type, ((const char*) data) + member->offset));
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  }

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "}\n"));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_serializer_serialize(LumSerializer* serializer, Host* host) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(host);

  if (serializer->serialized_data)
    __FAILURE_HANDLE(host_free(&serializer->serialized_data));

  serializer->serialized_data_allocated_size = 4096;
  serializer->serialized_data_length         = 0;

  __FAILURE_HANDLE(host_malloc(&serializer->serialized_data, serializer->serialized_data_allocated_size));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "Luminary\nVersion 5\n"));

  RendererSettings settings;
  __FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_SETTINGS, (const void*) &settings));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_serializer_store(LumSerializer* serializer, Path* path) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(path);

  const char* file_path_string;
  __FAILURE_HANDLE(luminary_path_apply(path, (const char*) 0, &file_path_string));

  FILE* file = fopen(file_path_string, "wb");

  if (file == (FILE*) 0)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to open file '%s'.", file_path_string);

  size_t write_length = fwrite(serializer->serialized_data, 1, serializer->serialized_data_length, file);

  if (write_length < serializer->serialized_data_length)
    warn_message("Write to file terminated prematurely. The file might be corrupted.");

  fclose(file);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_serializer_destroy(LumSerializer** serializer) {
  __CHECK_NULL_ARGUMENT(serializer);

  if ((*serializer)->serialized_data)
    __FAILURE_HANDLE(host_free(&(*serializer)->serialized_data));

  __FAILURE_HANDLE(host_free(serializer));

  return LUMINARY_SUCCESS;
}
