#include "lum_serializer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host/internal_host.h"
#include "internal_error.h"
#include "internal_path.h"
#include "lum_builtins.h"

#define LUM_SERIALIZER_PATH_BUFFER_SIZE (1u << 14)

LuminaryResult lum_serializer_create(LumSerializer** serializer) {
  __CHECK_NULL_ARGUMENT(serializer);

  __FAILURE_HANDLE(host_malloc(serializer, sizeof(LumSerializer)));
  memset(*serializer, 0, sizeof(LumSerializer));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_addressable_literal_is_valid(
  LumSerializer* serializer, LumBuiltinType type, const void* data, bool* is_valid) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(is_valid);

  switch (type) {
    case LUM_BUILTIN_TYPE_MESH: {
      LumBuiltinMesh mesh = *(LumBuiltinMesh*) data;
      *is_valid           = mesh.id != MESH_ID_INVALID;
    } break;
    case LUM_BUILTIN_TYPE_TEXTURE: {
      LumBuiltinTexture tex = *(LumBuiltinTexture*) data;
      *is_valid             = tex.id != TEXTURE_ID_INVALID;
    } break;
    default: {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' is not a addressable literal.", lum_builtin_types_strings[type]);
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_write(LumSerializer* serializer, const char* format, ...) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(format);

  va_list args;
  va_start(args, format);

  va_list args_copy;
  va_copy(args_copy, args);

  const size_t offset = serializer->serialized_data_length;

  int string_length = vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args);

  if (string_length < 0)
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "vsnprintf returned error code %d", string_length);

  // vsnprintf returns size excluding the NULL terminator but for allocation reasons we care about the size including the NULL terminator.
  int required_size = string_length + 1;

  if (required_size + offset > serializer->serialized_data_allocated_size) {
    serializer->serialized_data_allocated_size = (required_size + offset) * 2;
    __FAILURE_HANDLE(host_realloc(&serializer->serialized_data, serializer->serialized_data_allocated_size));

    vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args_copy);
  }

  serializer->serialized_data_length += string_length;

  va_end(args_copy);
  va_end(args);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_write_quoted_string(LumSerializer* serializer, const char* value) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(value);

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\""));

  for (size_t char_id = 0; value[char_id] != '\0'; char_id++) {
    switch (value[char_id]) {
      case '\\':
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "\\\\"));
        break;
      case '"':
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "\\\""));
        break;
      case '\n':
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "\\n"));
        break;
      case '\r':
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "\\r"));
        break;
      case '\t':
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "\\t"));
        break;
      default:
        __FAILURE_HANDLE(_lum_serializer_write(serializer, "%c", value[char_id]));
        break;
    }
  }

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\""));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_indent(LumSerializer* serializer) {
  __CHECK_NULL_ARGUMENT(serializer);

  for (uint32_t scope_id = 0; scope_id < serializer->current_scope_depth; scope_id++) {
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "\t"));
  }

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
      if (!isfinite(value)) {
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Cannot serialize a non-finite float.");
      }

      char formatted[64];
      const int formatted_length = snprintf(formatted, sizeof(formatted), "%.9g", value);
      if (formatted_length < 0 || (size_t) formatted_length >= sizeof(formatted)) {
        __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to format Luminary float literal.");
      }

      if (strchr(formatted, '.') == (char*) 0 && strchr(formatted, 'e') == (char*) 0 && strchr(formatted, 'E') == (char*) 0) {
        formatted[formatted_length]     = '.';
        formatted[formatted_length + 1] = '0';
        formatted[formatted_length + 2] = '\0';
      }

      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%s", formatted));
    } break;
    case LUM_BUILTIN_TYPE_ENUM: {
      uint32_t value = *(uint32_t*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%u", value));
    } break;
    case LUM_BUILTIN_TYPE_STRING: {
      LumBuiltinString string = *(LumBuiltinString*) data;
      __FAILURE_HANDLE(_lum_serializer_write_quoted_string(serializer, (string.string_ptr != (const char*) 0) ? string.string_ptr : ""));
    } break;
    default: {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' cannot be written as a literal.", lum_builtin_types_strings[type]);
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_addressable_literal(
  LumSerializer* serializer, Host* host, LumBuiltinType type, const void* data) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);

  switch (type) {
    case LUM_BUILTIN_TYPE_MESH: {
      LumBuiltinMesh mesh = *(LumBuiltinMesh*) data;

      const char* name;
      bool found = false;
      __FAILURE_HANDLE(dictionary_find_by_id(host->mesh_name_dict, mesh.id, &name, &found));

      if (found == false)
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Mesh '%u' has no name.", mesh.id);

      __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s ", lum_builtin_types_strings[type]));
      __FAILURE_HANDLE(_lum_serializer_write_quoted_string(serializer, name));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "]"));
    } break;
    case LUM_BUILTIN_TYPE_TEXTURE: {
      LumBuiltinTexture tex = *(LumBuiltinTexture*) data;

      const char* name;
      bool found = false;
      __FAILURE_HANDLE(dictionary_find_by_id(host->texture_name_dict, tex.id, &name, &found));

      if (found == false)
        __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture '%u' has no name.", tex.id);

      __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s ", lum_builtin_types_strings[type]));
      __FAILURE_HANDLE(_lum_serializer_write_quoted_string(serializer, name));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "]"));
    } break;
    default: {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' is not a addressable literal.", lum_builtin_types_strings[type]);
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_struct(
  LumSerializer* serializer, Host* host, LumBuiltinType type, const void* data, const char* name) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);

  if (name == (const char*) 0) {
    __DEBUG_ASSERT(lum_builtin_types_addressable[type] == false);
    if (serializer->current_scope_depth > 0) {
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
    }
    else {
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s]\n", lum_builtin_types_strings[type]));
    }
  }
  else {
    __DEBUG_ASSERT(lum_builtin_types_addressable[type] == true);
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s ", lum_builtin_types_strings[type]));
    __FAILURE_HANDLE(_lum_serializer_write_quoted_string(serializer, name));
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "]\n"));
  }

  __FAILURE_HANDLE(_lum_serializer_indent(serializer));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "{\n"));

  uint32_t member_count               = lum_builtin_types_member_counts[type];
  const LumBuiltinTypeMember* members = lum_builtin_types_member[type];

  serializer->current_scope_depth++;

  for (uint32_t member_id = 0; member_id < member_count; member_id++) {
    const LumBuiltinTypeMember* member = members + member_id;

    if (member->max_version < LUM_VERSION_CURRENT)
      continue;

    const bool member_is_struct      = lum_builtin_types_member_counts[member->type] > 0;
    const bool member_is_addressable = lum_builtin_types_addressable[member->type];

    const char* member_data = ((const char*) data) + member->offset;

    if (member_is_struct) {
      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, host, member->type, member_data, (const char*) 0));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
    else if (member_is_addressable) {
      bool is_valid;
      __FAILURE_HANDLE(_lum_serializer_addressable_literal_is_valid(serializer, member->type, member_data, &is_valid));

      if (is_valid == false)
        continue;

      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s = ", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_addressable_literal(serializer, host, member->type, member_data));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
    else {
      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s = ", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_literal(serializer, member->type, member_data));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
  }

  serializer->current_scope_depth--;

  __FAILURE_HANDLE(_lum_serializer_indent(serializer));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, (serializer->current_scope_depth > 0) ? "}" : "};\n"));

  return LUMINARY_SUCCESS;
}

static bool _lum_serializer_is_path_separator(char value) {
  return value == '/' || value == '\\';
}

static bool _lum_serializer_path_chars_equal(char lhs, char rhs) {
#if defined(WIN32)
  if (lhs == '/' || lhs == '\\')
    lhs = '\\';
  if (rhs == '/' || rhs == '\\')
    rhs = '\\';
  if (lhs >= 'A' && lhs <= 'Z')
    lhs = (char) (lhs - 'A' + 'a');
  if (rhs >= 'A' && rhs <= 'Z')
    rhs = (char) (rhs - 'A' + 'a');
#endif
  return lhs == rhs;
}

static LuminaryResult _lum_serializer_get_absolute_path(const char* path, char* absolute_path, size_t absolute_path_size) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(absolute_path);

#if defined(WIN32)
  if (_fullpath(absolute_path, path, absolute_path_size) == (char*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to resolve path '%s'.", path);
  }
#else
  char* resolved_path = realpath(path, (char*) 0);
  if (resolved_path == (char*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to resolve path '%s'.", path);
  }

  const size_t resolved_path_length = strlen(resolved_path);
  if (resolved_path_length >= absolute_path_size) {
    free(resolved_path);
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Resolved path exceeds serializer path buffer.");
  }

  memcpy(absolute_path, resolved_path, resolved_path_length + 1);
  free(resolved_path);
#endif

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_create_obj_reference(
  const Path* destination_path, const Path* obj_path, char* reference, size_t reference_size) {
  __CHECK_NULL_ARGUMENT(destination_path);
  __CHECK_NULL_ARGUMENT(obj_path);
  __CHECK_NULL_ARGUMENT(reference);

  const char* obj_path_string;
  __FAILURE_HANDLE(luminary_path_apply((Path*) obj_path, (const char*) 0, &obj_path_string));

  char absolute_destination_dir[LUM_SERIALIZER_PATH_BUFFER_SIZE];
  char absolute_obj_path[LUM_SERIALIZER_PATH_BUFFER_SIZE];
  const char* destination_dir =
    (destination_path->working_dir && destination_path->working_dir[0] != '\0') ? destination_path->working_dir : ".";

  __FAILURE_HANDLE(_lum_serializer_get_absolute_path(destination_dir, absolute_destination_dir, sizeof(absolute_destination_dir)));
  __FAILURE_HANDLE(_lum_serializer_get_absolute_path(obj_path_string, absolute_obj_path, sizeof(absolute_obj_path)));

  size_t common_length = 0;
  while (absolute_destination_dir[common_length] != '\0' && absolute_obj_path[common_length] != '\0'
         && _lum_serializer_path_chars_equal(absolute_destination_dir[common_length], absolute_obj_path[common_length])) {
    common_length++;
  }

  if (common_length == 0) {
    const size_t obj_path_length = strlen(absolute_obj_path);
    if (obj_path_length >= reference_size) {
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Absolute OBJ path exceeds serializer path buffer.");
    }
    memcpy(reference, absolute_obj_path, obj_path_length + 1);
    return LUMINARY_SUCCESS;
  }

  const bool common_path_boundary =
    (absolute_destination_dir[common_length] == '\0' && absolute_obj_path[common_length] == '\0')
    || (absolute_destination_dir[common_length] == '\0' && _lum_serializer_is_path_separator(absolute_obj_path[common_length]))
    || (absolute_obj_path[common_length] == '\0' && _lum_serializer_is_path_separator(absolute_destination_dir[common_length]));

  if (common_path_boundary == false) {
    while (common_length > 0 && !_lum_serializer_is_path_separator(absolute_destination_dir[common_length - 1])) {
      common_length--;
    }
  }

  const char* destination_remaining = absolute_destination_dir + common_length;
  const char* obj_remaining         = absolute_obj_path + common_length;
  while (_lum_serializer_is_path_separator(*destination_remaining))
    destination_remaining++;
  while (_lum_serializer_is_path_separator(*obj_remaining))
    obj_remaining++;

  size_t destination_components = 0;
  const char* cursor            = destination_remaining;
  while (*cursor != '\0') {
    while (_lum_serializer_is_path_separator(*cursor))
      cursor++;
    if (*cursor == '\0')
      break;
    destination_components++;
    while (*cursor != '\0' && !_lum_serializer_is_path_separator(*cursor))
      cursor++;
  }

  size_t offset = 0;
#if defined(WIN32)
  const char separator = '\\';
#else
  const char separator = '/';
#endif

  for (size_t component_id = 0; component_id < destination_components; component_id++) {
    if (offset + 3 >= reference_size) {
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Relative OBJ path exceeds serializer path buffer.");
    }
    if (offset > 0)
      reference[offset++] = separator;
    reference[offset++] = '.';
    reference[offset++] = '.';
  }

  cursor = obj_remaining;
  while (*cursor != '\0') {
    while (_lum_serializer_is_path_separator(*cursor))
      cursor++;
    if (*cursor == '\0')
      break;

    const char* component_start = cursor;
    while (*cursor != '\0' && !_lum_serializer_is_path_separator(*cursor))
      cursor++;
    const size_t component_length = (size_t) (cursor - component_start);

    if (offset + (offset > 0) + component_length >= reference_size) {
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Relative OBJ path exceeds serializer path buffer.");
    }
    if (offset > 0)
      reference[offset++] = separator;
    memcpy(reference + offset, component_start, component_length);
    offset += component_length;
  }

  if (offset == 0) {
    const size_t obj_path_length = strlen(absolute_obj_path);
    if (obj_path_length >= reference_size) {
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Absolute OBJ path exceeds serializer path buffer.");
    }
    memcpy(reference, absolute_obj_path, obj_path_length + 1);
  }
  else {
    reference[offset] = '\0';
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_obj_files(LumSerializer* serializer, Host* host, const Path* destination_path) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(destination_path);

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# OBJ Files\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  uint32_t num_obj_files;
  __FAILURE_HANDLE(array_get_num_elements(host->loaded_obj_files, &num_obj_files));

  for (uint32_t obj_id = 0; obj_id < num_obj_files; obj_id++) {
    LumBuiltinWavefrontObjFile obj_file;
    obj_file.name_prefix.string_ptr = host->loaded_obj_files[obj_id].wavefront_args->name_prefix;

    Path* src_obj_path = host->loaded_obj_files[obj_id].path;

    char obj_path[LUM_SERIALIZER_PATH_BUFFER_SIZE];
    __FAILURE_HANDLE(_lum_serializer_create_obj_reference(destination_path, src_obj_path, obj_path, sizeof(obj_path)));

    __FAILURE_HANDLE(
      _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE, (const void*) &obj_file, obj_path));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_globals(LumSerializer* serializer, Host* host) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock(host->scene_host, SCENE_ENTITY_TYPE_GLOBAL));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Renderer Settings\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  RendererSettings settings;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &settings, SCENE_ENTITY_SETTINGS));

  LumBuiltinSettings builtin_settings;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_settings_serialize(&settings, &builtin_settings));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_SETTINGS, (const void*) &builtin_settings, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Camera\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Camera camera;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &camera, SCENE_ENTITY_CAMERA));

  LumBuiltinCamera builtin_camera;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_camera_serialize(&camera, &builtin_camera));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_CAMERA, (const void*) &builtin_camera, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Ocean\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Ocean ocean;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &ocean, SCENE_ENTITY_OCEAN));

  LumBuiltinOcean builtin_ocean;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_ocean_serialize(&ocean, &builtin_ocean));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_OCEAN, (const void*) &builtin_ocean, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Sky\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Sky sky;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &sky, SCENE_ENTITY_SKY));

  LumBuiltinSky builtin_sky;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_sky_serialize(&sky, &builtin_sky));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_SKY, (const void*) &builtin_sky, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Cloud\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Cloud cloud;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &cloud, SCENE_ENTITY_CLOUD));

  LumBuiltinCloud builtin_cloud;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_cloud_serialize(&cloud, &builtin_cloud));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_CLOUD, (const void*) &builtin_cloud, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Fog\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Fog fog;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &fog, SCENE_ENTITY_FOG));

  LumBuiltinFog builtin_fog;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_fog_serialize(&fog, &builtin_fog));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_FOG, (const void*) &builtin_fog, (const char*) 0));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Particles\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  Particles particles;
  __FAILURE_HANDLE_CRITICAL(scene_get(host->scene_host, &particles, SCENE_ENTITY_PARTICLES));

  LumBuiltinParticles builtin_particles;
  __FAILURE_HANDLE_CRITICAL(lum_builtin_particles_serialize(&particles, &builtin_particles));

  __FAILURE_HANDLE_CRITICAL(
    _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_PARTICLES, (const void*) &builtin_particles, (const char*) 0));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(host->scene_host, SCENE_ENTITY_TYPE_GLOBAL));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_lists(LumSerializer* serializer, Host* host) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock(host->scene_host, SCENE_ENTITY_TYPE_LIST));

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Instances\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  uint32_t num_instances;
  __FAILURE_HANDLE_CRITICAL(scene_get_entry_count(host->scene_host, SCENE_ENTITY_INSTANCES, &num_instances));

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    MeshInstance instance;
    __FAILURE_HANDLE_CRITICAL(scene_get_entry(host->scene_host, &instance, SCENE_ENTITY_INSTANCES, instance_id));

    const char* name;
    bool found = false;
    __FAILURE_HANDLE_CRITICAL(dictionary_find_by_id(host->mesh_instance_name_dict, instance_id, &name, &found));
    if (found == false)
      __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Instance '%u' has no name.", instance_id);

    LuminaryInstance lum_instance;
    __FAILURE_HANDLE_CRITICAL(mesh_instance_to_public_api_instance(&lum_instance, &instance));

    LumBuiltinInstance builtin_instance;
    __FAILURE_HANDLE_CRITICAL(lum_builtin_instance_serialize(&lum_instance, &builtin_instance));

    __FAILURE_HANDLE_CRITICAL(
      _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_INSTANCE, (const void*) &builtin_instance, name));
    __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  }

  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "# Materials\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));

  uint32_t num_materials;
  __FAILURE_HANDLE_CRITICAL(scene_get_entry_count(host->scene_host, SCENE_ENTITY_MATERIALS, &num_materials));

  for (uint32_t material_id = 0; material_id < num_materials; material_id++) {
    Material material;
    __FAILURE_HANDLE_CRITICAL(scene_get_entry(host->scene_host, &material, SCENE_ENTITY_MATERIALS, material_id));

    const char* name;
    bool found = false;
    __FAILURE_HANDLE_CRITICAL(dictionary_find_by_id(host->material_name_dict, material_id, &name, &found));
    if (found == false)
      __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Material '%u' has no name.", material_id);

    LumBuiltinMaterial builtin_material;
    __FAILURE_HANDLE_CRITICAL(lum_builtin_material_serialize(&material, &builtin_material));

    __FAILURE_HANDLE_CRITICAL(
      _lum_serializer_serialize_struct(serializer, host, LUM_BUILTIN_TYPE_MATERIAL, (const void*) &builtin_material, name));
    __FAILURE_HANDLE_CRITICAL(_lum_serializer_write(serializer, "\n"));
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(host->scene_host, SCENE_ENTITY_TYPE_LIST));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_serializer_serialize(LumSerializer* serializer, Host* host, Path* destination_path) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(destination_path);

  if (serializer->serialized_data)
    __FAILURE_HANDLE(host_free(&serializer->serialized_data));

  serializer->serialized_data_allocated_size = 4096;
  serializer->serialized_data_length         = 0;

  __FAILURE_HANDLE(host_malloc(&serializer->serialized_data, serializer->serialized_data_allocated_size));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "Luminary\nVersion 5\n\n"));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# This file was automatically created by Luminary.\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Please read the documentation before making changes.\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));

  __FAILURE_HANDLE(_lum_serializer_serialize_obj_files(serializer, host, destination_path));
  __FAILURE_HANDLE(_lum_serializer_serialize_globals(serializer, host));
  __FAILURE_HANDLE(_lum_serializer_serialize_lists(serializer, host));

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

  size_t write_length     = fwrite(serializer->serialized_data, 1, serializer->serialized_data_length, file);
  const bool write_failed = write_length != serializer->serialized_data_length || ferror(file) != 0;
  const int close_result  = fclose(file);

  if (write_failed) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to write complete Luminary V5 file.");
  }

  if (close_result != 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to close Luminary V5 file after writing.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_serializer_destroy(LumSerializer** serializer) {
  __CHECK_NULL_ARGUMENT(serializer);

  if ((*serializer)->serialized_data)
    __FAILURE_HANDLE(host_free(&(*serializer)->serialized_data));

  __FAILURE_HANDLE(host_free(serializer));

  return LUMINARY_SUCCESS;
}
