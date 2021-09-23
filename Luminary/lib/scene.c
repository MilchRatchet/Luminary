#include "scene.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "light.h"
#include "png.h"
#include "raytrace.h"
#include "wavefront.h"

static const int LINE_SIZE       = 4096;
static const int CURRENT_VERSION = 4;

static int validate_filetype(const char* line) {
  int result = 0;

  result += line[0] ^ 'L';
  result += line[1] ^ 'u';
  result += line[2] ^ 'm';
  result += line[3] ^ 'i';
  result += line[4] ^ 'n';
  result += line[5] ^ 'a';
  result += line[6] ^ 'r';
  result += line[7] ^ 'y';

  return result;
}

static void parse_general_settings(General* general, Wavefront_Content* content, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* MESHFILE */
    case 4993446653056992589u:
      char* source = (char*) malloc(LINE_SIZE);
      sscanf_s(value, "%s\n", source, LINE_SIZE);
      if (read_wavefront_file(source, content)) {
        print_error("Mesh file could not be loaded!");
      }
      if (general->mesh_files_count == general->mesh_files_length) {
        general->mesh_files_length *= 2;
        general->mesh_files = safe_realloc(general->mesh_files, sizeof(char*) * general->mesh_files_length);
      }
      general->mesh_files[general->mesh_files_count++] = source;
      break;
    /* WIDTH___ */
    case 6872316320646711639u:
      sscanf_s(value, "%d\n", &general->width);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf_s(value, "%d\n", &general->height);
      break;
    /* BOUNCES_ */
    case 6868910012049477442u:
      sscanf_s(value, "%d\n", &general->max_ray_depth);
      break;
    /* SAMPLES_ */
    case 6868910050737209683u:
      sscanf_s(value, "%d\n", &general->samples);
      break;
    /* DENOISER */
    case 5928236058831373636u:
      sscanf_s(value, "%d\n", &general->denoiser);
      break;
    /* OUTPUTFN */
    case 5640288308724782415u:
      sscanf_s(value, "%s\n", general->output_path, LINE_SIZE);
      break;
    default:
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "%8.8s (%zu) is not a valid GENERAL setting.", line, key);
      print_error(error_msg);
      free(error_msg);
      break;
  }
}

static void parse_camera_settings(Camera* camera, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* POSITION */
    case 5642809484474797904u:
      sscanf_s(value, "%f %f %f\n", &camera->pos.x, &camera->pos.y, &camera->pos.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf_s(value, "%f %f %f\n", &camera->rotation.x, &camera->rotation.y, &camera->rotation.z);
      break;
    /* FOV_____ */
    case 6872316419616689990u:
      sscanf_s(value, "%f\n", &camera->fov);
      break;
    /* FOCALLEN */
    case 5639997998747569990u:
      sscanf_s(value, "%f\n", &camera->focal_length);
      break;
    /* APERTURE */
    case 4995148757353189441u:
      sscanf_s(value, "%f\n", &camera->aperture_size);
      break;
    /* AUTOEXP_ */
    case 6868086486446921025u:
      sscanf_s(value, "%d\n", &camera->auto_exposure);
      break;
    /* EXPOSURE */
    case 4995148753008613445u:
      sscanf_s(value, "%f\n", &camera->exposure);
      break;
    /* BLOOM___ */
    case 6872316342038383682u:
      sscanf_s(value, "%d\n", &camera->bloom);
      break;
    /* BLOOMSTR */
    case 5932458200661969986u:
      sscanf_s(value, "%f\n", &camera->bloom_strength);
      break;
    /* DITHER__ */
    case 6872302013910370628u:
      sscanf_s(value, "%d\n", &camera->dithering);
      break;
    /* FARCLIPD */
    case 4922514984611758406u:
      sscanf_s(value, "%f\n", &camera->far_clip_distance);
      break;
    /* TONEMAP_ */
    case 6868061231871053652u:
      sscanf_s(value, "%d\n", &camera->tonemap);
      break;
    /* ALPHACUT */
    case 6076837219871509569u:
      sscanf_s(value, "%f\n", &camera->alpha_cutoff);
      break;
    /* FILTER__ */
    case 6872302014111172934u:
      sscanf_s(value, "%d\n", &camera->filter);
      break;
    default:
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "%8.8s (%zu) is not a valid CAMERA setting.", line, key);
      print_error(error_msg);
      free(error_msg);
      break;
  }
}

static void parse_sky_settings(Sky* sky, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* SUNCOLOR */
    case 5931043137585567059u:
      sscanf_s(value, "%f %f %f\n", &sky->sun_color.r, &sky->sun_color.g, &sky->sun_color.b);
      break;
    /* STRENGTH */
    case 5211869070270551123u:
      sscanf_s(value, "%f\n", &sky->sun_strength);
      break;
    /* AZIMUTH_ */
    case 6865830357271927361u:
      sscanf_s(value, "%f\n", &sky->azimuth);
      break;
    /* ALTITUDE */
    case 4991208107529227329u:
      sscanf_s(value, "%f\n", &sky->altitude);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf_s(value, "%f\n", &sky->base_density);
      break;
    /* RAYLEIGH */
    case 5208212056059756882u:
      sscanf_s(value, "%f\n", &sky->rayleigh_falloff);
      break;
    /* MIE_____ */
    case 6872316419615574349u:
      sscanf_s(value, "%f\n", &sky->mie_falloff);
      break;
    default:
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "%8.8s (%zu) is not a valid SKY setting.", line, key);
      print_error(error_msg);
      free(error_msg);
      break;
  }
}

static void parse_ocean_settings(Ocean* ocean, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(value, "%d\n", &ocean->active);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf_s(value, "%f\n", &ocean->height);
      break;
    /* AMPLITUD */
    case 4923934441389182273u:
      sscanf_s(value, "%f\n", &ocean->amplitude);
      break;
    /* FREQUENC */
    case 4849890081462637126u:
      sscanf_s(value, "%f\n", &ocean->frequency);
      break;
    /* CHOPPY__ */
    case 6872309757870295107u:
      sscanf_s(value, "%f\n", &ocean->choppyness);
      break;
    /* SPEED___ */
    case 6872316303215251539u:
      sscanf_s(value, "%f\n", &ocean->speed);
      break;
    /* ANIMATED */
    case 4919430807418392129u:
      sscanf_s(value, "%d\n", &ocean->update);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf_s(value, "%f %f %f %f\n", &ocean->albedo.r, &ocean->albedo.g, &ocean->albedo.b, &ocean->albedo.a);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf_s(value, "%d\n", &ocean->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf_s(value, "%f\n", &ocean->refractive_index);
      break;
    default:
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "%8.8s (%zu) is not a valid OCEAN setting.", line, key);
      print_error(error_msg);
      free(error_msg);
      break;
  }
}

static void parse_toy_settings(Toy* toy, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(value, "%d\n", &toy->active);
      break;
    /* POSITION */
    case 5642809484474797904u:
      sscanf_s(value, "%f %f %f\n", &toy->position.x, &toy->position.y, &toy->position.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf_s(value, "%f %f %f\n", &toy->rotation.x, &toy->rotation.y, &toy->rotation.z);
      break;
    /* SHAPE___ */
    case 6872316307694504019u:
      sscanf_s(value, "%d\n", &toy->shape);
      break;
    /* SCALE__ */
    case 6872316307627393875u:
      sscanf_s(value, "%f\n", &toy->scale);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf_s(value, "%f %f %f %f\n", &toy->albedo.r, &toy->albedo.g, &toy->albedo.b, &toy->albedo.a);
      break;
    /* MATERIAL */
    case 5494753638068011341u:
      sscanf_s(value, "%f %f %f\n", &toy->material.r, &toy->material.g, &toy->material.b);
      break;
    /* EMISSION */
    case 5642809480346946885u:
      sscanf_s(value, "%f %f %f\n", &toy->emission.r, &toy->emission.g, &toy->emission.b);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf_s(value, "%d\n", &toy->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf_s(value, "%f\n", &toy->refractive_index);
      break;
    default:
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "%8.8s (%zu) is not a valid TOY setting.", line, key);
      print_error(error_msg);
      free(error_msg);
      break;
  }
}

static Scene get_default_scene() {
  Scene scene;

  scene.camera.pos.x             = 0.0f;
  scene.camera.pos.y             = 0.0f;
  scene.camera.pos.z             = 0.0f;
  scene.camera.rotation.x        = 0.0f;
  scene.camera.rotation.y        = 0.0f;
  scene.camera.rotation.z        = 0.0f;
  scene.camera.fov               = 1.0f;
  scene.camera.focal_length      = 1.0f;
  scene.camera.aperture_size     = 0.00f;
  scene.camera.exposure          = 1.0f;
  scene.camera.auto_exposure     = 1;
  scene.camera.bloom             = 1;
  scene.camera.bloom_strength    = 0.1f;
  scene.camera.dithering         = 1;
  scene.camera.alpha_cutoff      = 0.0f;
  scene.camera.far_clip_distance = 1000000.0f;
  scene.camera.tonemap           = TONEMAP_ACES;
  scene.camera.filter            = FILTER_NONE;
  scene.camera.wasd_speed        = 1.0f;
  scene.camera.mouse_speed       = 1.0f;
  scene.camera.smooth_movement   = 0;
  scene.camera.smoothing_factor  = 0.1f;

  scene.ocean.active           = 0;
  scene.ocean.emissive         = 0;
  scene.ocean.update           = 0;
  scene.ocean.height           = 0.0f;
  scene.ocean.amplitude        = 0.6f;
  scene.ocean.frequency        = 0.16f;
  scene.ocean.choppyness       = 4.0f;
  scene.ocean.speed            = 1.0f;
  scene.ocean.time             = 0.0f;
  scene.ocean.albedo.r         = 0.0f;
  scene.ocean.albedo.g         = 0.0f;
  scene.ocean.albedo.b         = 0.0f;
  scene.ocean.albedo.a         = 0.9f;
  scene.ocean.refractive_index = 1.333f;

  scene.toy.active           = 0;
  scene.toy.emissive         = 0;
  scene.toy.shape            = TOY_SPHERE;
  scene.toy.position.x       = 0.0f;
  scene.toy.position.y       = 0.0f;
  scene.toy.position.z       = 0.0f;
  scene.toy.rotation.x       = 0.0f;
  scene.toy.rotation.y       = 0.0f;
  scene.toy.rotation.z       = 0.0f;
  scene.toy.scale            = 1.0f;
  scene.toy.refractive_index = 1.0f;
  scene.toy.albedo.r         = 0.9f;
  scene.toy.albedo.g         = 0.9f;
  scene.toy.albedo.b         = 0.9f;
  scene.toy.albedo.a         = 1.0f;
  scene.toy.material.r       = 0.3f;
  scene.toy.material.g       = 0.0f;
  scene.toy.material.b       = 1.0f;
  scene.toy.material.a       = 0.0f;
  scene.toy.emission.r       = 0.0f;
  scene.toy.emission.g       = 0.0f;
  scene.toy.emission.b       = 0.0f;
  scene.toy.emission.a       = 0.0f;

  scene.sky.sun_color.r      = 1.0f;
  scene.sky.sun_color.g      = 0.9f;
  scene.sky.sun_color.b      = 0.8f;
  scene.sky.altitude         = 0.5f;
  scene.sky.azimuth          = 3.141f;
  scene.sky.sun_strength     = 40.0f;
  scene.sky.base_density     = 1.0f;
  scene.sky.rayleigh_falloff = 0.125f;
  scene.sky.mie_falloff      = 0.833333f;

  return scene;
}

static void convert_wavefront_to_internal(Wavefront_Content content, Scene* scene) {
  scene->triangles_length = convert_wavefront_content(&scene->triangles, content);

  Node2* initial_nodes = build_bvh_structure(&scene->triangles, &scene->triangles_length, &scene->nodes_length);

  scene->nodes = collapse_bvh(initial_nodes, scene->nodes_length, &scene->triangles, scene->triangles_length, &scene->nodes_length);

  free(initial_nodes);

  scene->traversal_triangles = malloc(sizeof(TraversalTriangle) * scene->triangles_length);

  for (unsigned int i = 0; i < scene->triangles_length; i++) {
    Triangle triangle    = scene->triangles[i];
    TraversalTriangle tt = {
      .vertex = {.x = triangle.vertex.x, .y = triangle.vertex.y, .z = triangle.vertex.z},
      .edge1  = {.x = triangle.edge1.x, .y = triangle.edge1.y, .z = triangle.edge1.z},
      .edge2  = {.x = triangle.edge2.x, .y = triangle.edge2.y, .z = triangle.edge2.z}};
    scene->traversal_triangles[i] = tt;
    scene->triangles[i]           = triangle;
  }

  scene->materials_length    = content.materials_length;
  scene->texture_assignments = get_texture_assignments(content);
}

RaytraceInstance* load_scene(const char* filename) {
  FILE* file;
  fopen_s(&file, filename, "rb");

  char* line = (char*) malloc(LINE_SIZE);

  sprintf(line, "Scene file \"%s\" could not be opened.", filename);

  assert((unsigned long long) file, line, 1);

  fgets(line, LINE_SIZE, file);

  assert(!validate_filetype(line), "Scene file is not a Luminary scene file!", 1);

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v' || line[0] == 'V') {
    int version = 0;
    sscanf_s(line, "%*c %d\n", &version);
    assert(version == CURRENT_VERSION, "Incompatible Scene version! Update the file or use an older version of Luminary!", 1);
  }
  else {
    print_error("Scene file has no version information, assuming correct version!")
  }

  Scene scene = get_default_scene();

  General general = {
    .width             = 1280,
    .height            = 720,
    .max_ray_depth     = 5,
    .samples           = 16,
    .denoiser          = 1,
    .output_path       = malloc(LINE_SIZE),
    .mesh_files        = malloc(sizeof(char*) * 10),
    .mesh_files_count  = 0,
    .mesh_files_length = 10};

  strncpy_s(general.output_path, LINE_SIZE, "output.png", 11);

  Wavefront_Content content = create_wavefront_content();

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'G') {
      parse_general_settings(&general, &content, line + 7 + 1);
    }
    else if (line[0] == 'C') {
      parse_camera_settings(&scene.camera, line + 6 + 1);
    }
    else if (line[0] == 'S') {
      parse_sky_settings(&scene.sky, line + 3 + 1);
    }
    else if (line[0] == 'O') {
      parse_ocean_settings(&scene.ocean, line + 5 + 1);
    }
    else if (line[0] == 'T') {
      parse_toy_settings(&scene.toy, line + 3 + 1);
    }
    else if (line[0] == '#' || line[0] == 10) {
      continue;
    }
    else {
      char* error_msg = (char*) malloc(LINE_SIZE);
      sprintf(error_msg, "Scene file contains unknown line!\n Content: %s", line);
      print_error(error_msg);
      free(error_msg);
    }
  }

  fclose(file);
  free(line);

  assert(general.mesh_files_count, "No mesh files where loaded.", 1);

  convert_wavefront_to_internal(content, &scene);

  process_lights(&scene);

  void* albedo_atlas      = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  void* illuminance_atlas = initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  void* material_atlas    = initialize_textures(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  return instance;
}

RaytraceInstance* load_obj_as_scene(char* filename) {
  Scene scene = get_default_scene();

  General general = {
    .width             = 1280,
    .height            = 720,
    .max_ray_depth     = 5,
    .samples           = 16,
    .denoiser          = 1,
    .output_path       = malloc(LINE_SIZE),
    .mesh_files        = malloc(sizeof(char*) * 10),
    .mesh_files_count  = 0,
    .mesh_files_length = 10};

  general.mesh_files[0] = malloc(LINE_SIZE);
  strcpy_s(general.mesh_files[0], LINE_SIZE, filename);

  strncpy_s(general.output_path, LINE_SIZE, "output.png", 11);

  Wavefront_Content content = create_wavefront_content();

  assert(!read_wavefront_file(filename, &content), "Mesh file could not be loaded.", 1);

  general.mesh_files[general.mesh_files_count++] = filename;

  convert_wavefront_to_internal(content, &scene);

  process_lights(&scene);

  void* albedo_atlas      = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  void* illuminance_atlas = initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  void* material_atlas    = initialize_textures(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  return instance;
}

void serialize_scene(RaytraceInstance* instance) {
  FILE* file;
  fopen_s(&file, "generated.lum", "wb");

  if (!file) {
    print_error("Could not export settings!");
    return;
  }

  char* line = malloc(LINE_SIZE);

  sprintf_s(line, LINE_SIZE, "Luminary\n");
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "V %d\n", CURRENT_VERSION);
  fputs(line, file);

  sprintf_s(
    line, LINE_SIZE,
    "#===============================\n# This file was automatically\n# created by Luminary.\n#\n# Please read the documentation\n# before "
    "changing any settings.\n#===============================\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# General Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "GENERAL WIDTH___ %d\n", instance->settings.width);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "GENERAL HEIGHT__ %d\n", instance->settings.height);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "GENERAL BOUNCES_ %d\n", instance->settings.max_ray_depth);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "GENERAL SAMPLES_ %d\n", instance->settings.samples);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "GENERAL DENOISER %d\n", instance->settings.denoiser);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "GENERAL OUTPUTFN %s\n", instance->settings.output_path);
  fputs(line, file);
  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    sprintf_s(line, LINE_SIZE, "GENERAL MESHFILE %s\n", instance->settings.mesh_files[i]);
    fputs(line, file);
  }

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Camera Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(
    line, LINE_SIZE, "CAMERA POSITION %f %f %f\n", instance->scene_gpu.camera.pos.x, instance->scene_gpu.camera.pos.y,
    instance->scene_gpu.camera.pos.z);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "CAMERA ROTATION %f %f %f\n", instance->scene_gpu.camera.rotation.x, instance->scene_gpu.camera.rotation.y,
    instance->scene_gpu.camera.rotation.z);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA FOV_____ %f\n", instance->scene_gpu.camera.fov);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA FOCALLEN %f\n", instance->scene_gpu.camera.focal_length);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA APERTURE %f\n", instance->scene_gpu.camera.aperture_size);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA EXPOSURE %f\n", instance->scene_gpu.camera.exposure);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA BLOOM___ %d\n", instance->scene_gpu.camera.bloom);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA BLOOMSTR %f\n", instance->scene_gpu.camera.bloom_strength);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA DITHER__ %d\n", instance->scene_gpu.camera.dithering);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA FARCLIPD %f\n", instance->scene_gpu.camera.far_clip_distance);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA TONEMAP_ %d\n", instance->scene_gpu.camera.tonemap);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA ALPHACUT %f\n", instance->scene_gpu.camera.alpha_cutoff);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA AUTOEXP_ %d\n", instance->scene_gpu.camera.auto_exposure);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CAMERA FILTER__ %d\n", instance->scene_gpu.camera.filter);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Sky Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(
    line, LINE_SIZE, "SKY SUNCOLOR %f %f %f\n", instance->scene_gpu.sky.sun_color.r, instance->scene_gpu.sky.sun_color.g,
    instance->scene_gpu.sky.sun_color.b);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY AZIMUTH_ %f\n", instance->scene_gpu.sky.azimuth);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY ALTITUDE %f\n", instance->scene_gpu.sky.altitude);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY STRENGTH %f\n", instance->scene_gpu.sky.sun_strength);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY DENSITY_ %f\n", instance->scene_gpu.sky.base_density);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY RAYLEIGH %f\n", instance->scene_gpu.sky.rayleigh_falloff);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY MIE_____ %f\n", instance->scene_gpu.sky.mie_falloff);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Ocean Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "OCEAN ACTIVE__ %d\n", instance->scene_gpu.ocean.active);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN HEIGHT__ %f\n", instance->scene_gpu.ocean.height);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN AMPLITUD %f\n", instance->scene_gpu.ocean.amplitude);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN FREQUENC %f\n", instance->scene_gpu.ocean.frequency);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN CHOPPY__ %f\n", instance->scene_gpu.ocean.choppyness);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN SPEED___ %f\n", instance->scene_gpu.ocean.speed);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN ANIMATED %d\n", instance->scene_gpu.ocean.update);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "OCEAN COLOR___ %f %f %f %f\n", instance->scene_gpu.ocean.albedo.r, instance->scene_gpu.ocean.albedo.g,
    instance->scene_gpu.ocean.albedo.b, instance->scene_gpu.ocean.albedo.a);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN EMISSIVE %d\n", instance->scene_gpu.ocean.emissive);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "OCEAN REFRACT_ %f\n", instance->scene_gpu.ocean.refractive_index);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Toy Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "TOY ACTIVE__ %d\n", instance->scene_gpu.toy.active);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "TOY POSITION %f %f %f\n", instance->scene_gpu.toy.position.x, instance->scene_gpu.toy.position.y,
    instance->scene_gpu.toy.position.z);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "TOY ROTATION %f %f %f\n", instance->scene_gpu.toy.rotation.x, instance->scene_gpu.toy.rotation.y,
    instance->scene_gpu.toy.rotation.z);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "TOY SHAPE___ %d\n", instance->scene_gpu.toy.shape);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "TOY SCALE___ %f\n", instance->scene_gpu.toy.scale);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "TOY COLOR___ %f %f %f %f\n", instance->scene_gpu.toy.albedo.r, instance->scene_gpu.toy.albedo.g,
    instance->scene_gpu.toy.albedo.b, instance->scene_gpu.toy.albedo.a);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "TOY MATERIAL %f %f %f\n", instance->scene_gpu.toy.material.r, instance->scene_gpu.toy.material.g,
    instance->scene_gpu.toy.material.b);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "TOY EMISSION %f %f %f\n", instance->scene_gpu.toy.emission.r, instance->scene_gpu.toy.emission.g,
    instance->scene_gpu.toy.emission.b);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "TOY EMISSIVE %d\n", instance->scene_gpu.toy.emissive);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "TOY REFRACT_ %f\n", instance->scene_gpu.toy.refractive_index);
  fputs(line, file);

  free(line);

  fclose(file);
}

void free_atlases(RaytraceInstance* instance) {
  free_textures_atlas(instance->albedo_atlas, instance->albedo_atlas_length);
  free_textures_atlas(instance->illuminance_atlas, instance->illuminance_atlas_length);
  free_textures_atlas(instance->material_atlas, instance->material_atlas_length);
}

void free_strings(RaytraceInstance* instance) {
  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    free(instance->settings.mesh_files[i]);
  }
  free(instance->settings.mesh_files);
  free(instance->settings.output_path);
}

void free_scene(Scene scene) {
  free(scene.triangles);
  free(scene.traversal_triangles);
  free(scene.nodes);
  free(scene.lights);
  free(scene.texture_assignments);
}
