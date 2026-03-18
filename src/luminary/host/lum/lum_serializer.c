#include "lum_serializer.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "host/internal_host.h"
#include "internal_error.h"
#include "internal_path.h"
#include "lum_builtins.h"

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

  const size_t offset = serializer->serialized_data_length;

  int string_length = vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args);

  if (string_length < 0)
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "vsnprintf returned error code %d", string_length);

  // vsnprintf returns size excluding the NULL terminator but for allocation reasons we care about the size including the NULL terminator.
  int required_size = string_length + 1;

  if (required_size + offset > serializer->serialized_data_allocated_size) {
    serializer->serialized_data_allocated_size = (required_size + offset) * 2;
    __FAILURE_HANDLE(host_realloc(&serializer->serialized_data, serializer->serialized_data_allocated_size));

    vsnprintf(serializer->serialized_data + offset, serializer->serialized_data_allocated_size - offset, format, args);
  }

  serializer->serialized_data_length += string_length;

  va_end(args);

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
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%f", value));
    } break;
    case LUM_BUILTIN_TYPE_ENUM: {
      uint32_t value = *(uint32_t*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "%u", value));
    } break;
    case LUM_BUILTIN_TYPE_STRING: {
      LumBuiltinString string = *(LumBuiltinString*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "\"%s\"", (string.string_ptr != (const char*) 0) ? string.string_ptr : ""));
    } break;
    default: {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' cannot be written as a literal.", lum_builtin_types_strings[type]);
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_addressable_literal(LumSerializer* serializer, LumBuiltinType type, const void* data) {
  __CHECK_NULL_ARGUMENT(serializer);
  __CHECK_NULL_ARGUMENT(data);

  switch (type) {
    case LUM_BUILTIN_TYPE_MESH: {
      // TODO
      LumBuiltinMesh mesh = *(LumBuiltinMesh*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s \"%s\"]", lum_builtin_types_strings[type], "TODO"));
    } break;
    case LUM_BUILTIN_TYPE_TEXTURE: {
      // TODO
      LumBuiltinTexture tex = *(LumBuiltinTexture*) data;
      __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s \"%s\"]", lum_builtin_types_strings[type], "TODO"));
    } break;
    default: {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Builtin type '%s' is not a addressable literal.", lum_builtin_types_strings[type]);
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_serializer_serialize_struct(LumSerializer* serializer, LumBuiltinType type, const void* data, const char* name) {
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
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "[%s \"%s\"]\n", lum_builtin_types_strings[type], name));
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

    if (member_is_struct) {
      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, member->type, ((const char*) data) + member->offset, (const char*) 0));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
    else if (member_is_addressable) {
      bool is_valid;
      __FAILURE_HANDLE(
        _lum_serializer_addressable_literal_is_valid(serializer, member->type, ((const char*) data) + member->offset, &is_valid));

      if (is_valid == false)
        continue;

      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s = ", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_addressable_literal(serializer, member->type, ((const char*) data) + member->offset));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
    else {
      __FAILURE_HANDLE(_lum_serializer_indent(serializer));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ".%s = ", member->name));
      __FAILURE_HANDLE(_lum_serializer_serialize_literal(serializer, member->type, ((const char*) data) + member->offset));
      __FAILURE_HANDLE(_lum_serializer_write(serializer, ",\n"));
    }
  }

  serializer->current_scope_depth--;

  __FAILURE_HANDLE(_lum_serializer_indent(serializer));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, (serializer->current_scope_depth > 0) ? "}" : "};\n"));

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

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "Luminary\nVersion 5\n\n"));

  // TODO: This is bust, this is getting is data from the caller scene when it should be using the host scene
  // TODO: Use critical section failure handles
  // TODO: Im dumb, I need to pass in the builtin types and not the Luminary types

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# This file was automatically created by Luminary.\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Please read the documentation before making changes.\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));

  __FAILURE_HANDLE(scene_lock(host->scene_host, SCENE_ENTITY_TYPE_GLOBAL));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Renderer Settings\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  RendererSettings settings;
  __FAILURE_HANDLE(luminary_host_get_settings(host, &settings));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_SETTINGS, (const void*) &settings, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Camera\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Camera camera;
  __FAILURE_HANDLE(luminary_host_get_camera(host, &camera));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_CAMERA, (const void*) &camera, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Ocean\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Ocean ocean;
  __FAILURE_HANDLE(luminary_host_get_ocean(host, &ocean));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_OCEAN, (const void*) &ocean, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Sky\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Sky sky;
  __FAILURE_HANDLE(luminary_host_get_sky(host, &sky));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_SKY, (const void*) &sky, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Cloud\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Cloud cloud;
  __FAILURE_HANDLE(luminary_host_get_cloud(host, &cloud));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_CLOUD, (const void*) &cloud, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Fog\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Fog fog;
  __FAILURE_HANDLE(luminary_host_get_fog(host, &fog));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_FOG, (const void*) &fog, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Particles\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  Particles particles;
  __FAILURE_HANDLE(luminary_host_get_particles(host, &particles));
  __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_PARTICLES, (const void*) &particles, (const char*) 0));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# OBJ Files\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  uint32_t num_obj_files;
  __FAILURE_HANDLE(array_get_num_elements(host->loaded_obj_files, &num_obj_files));

  for (uint32_t obj_id = 0; obj_id < num_obj_files; obj_id++) {
    LumBuiltinWavefrontObjFile obj_file;

    obj_file.name_prefix.string_ptr = host->loaded_obj_files[obj_id].wavefront_args.name_prefix;

    const char* obj_path;
    __FAILURE_HANDLE(luminary_path_apply(host->loaded_obj_files[obj_id].path, (const char*) 0, &obj_path));

    __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE, (const void*) &obj_file, obj_path));
  }

  __FAILURE_HANDLE(scene_unlock(host->scene_host, SCENE_ENTITY_TYPE_GLOBAL));

  __FAILURE_HANDLE(scene_lock(host->scene_host, SCENE_ENTITY_TYPE_LIST));

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Instances\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  uint32_t num_instances;
  __FAILURE_HANDLE(scene_get_entry_count(host->scene_host, SCENE_ENTITY_INSTANCES, &num_instances));

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    MeshInstance instance;
    __FAILURE_HANDLE(scene_get_entry(host->scene_host, &instance, SCENE_ENTITY_INSTANCES, instance_id));

    const char* name;
    bool found = false;
    __FAILURE_HANDLE(dictionary_find_by_id(host->mesh_instance_name_dict, instance_id, &name, &found));

    __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_INSTANCE, (const void*) &instance, name));
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  }

  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "# Materials\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "#==============================================================\n"));
  __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));

  uint32_t num_materials;
  __FAILURE_HANDLE(scene_get_entry_count(host->scene_host, SCENE_ENTITY_MATERIALS, &num_materials));

  for (uint32_t material_id = 0; material_id < num_materials; material_id++) {
    Material material;
    __FAILURE_HANDLE(scene_get_entry(host->scene_host, &material, SCENE_ENTITY_MATERIALS, material_id));

    const char* name;
    bool found = false;
    __FAILURE_HANDLE(dictionary_find_by_id(host->material_name_dict, material_id, &name, &found));

    __FAILURE_HANDLE(_lum_serializer_serialize_struct(serializer, LUM_BUILTIN_TYPE_MATERIAL, (const void*) &material, name));
    __FAILURE_HANDLE(_lum_serializer_write(serializer, "\n"));
  }

  __FAILURE_HANDLE(scene_unlock(host->scene_host, SCENE_ENTITY_TYPE_LIST));

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
