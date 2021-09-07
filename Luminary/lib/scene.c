#include "scene.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static void parse_general_settings(
  int* width, int* height, int* bounces, int* samples, char* output_path, int* denoiser, Wavefront_Content* content, char* line) {
  const uint64_t key = *((uint64_t*) line);

  switch (key) {
    /* MESHFILE */
    case 4993446653056992589u:
      char* source = (char*) malloc(LINE_SIZE);
      sscanf_s(line, "%*s %s\n", source, LINE_SIZE);
      if (read_wavefront_file(source, content)) {
        print_error("Mesh file could not be loaded!");
      }
      free(source);
      break;
    /* WIDTH___ */
    case 6872316320646711639u:
      sscanf_s(line, "%*s %d\n", width);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf_s(line, "%*s %d\n", height);
      break;
    /* BOUNCES_ */
    case 6868910012049477442u:
      sscanf_s(line, "%*s %d\n", bounces);
      break;
    /* SAMPLES_ */
    case 6868910050737209683u:
      sscanf_s(line, "%*s %d\n", samples);
      break;
    /* DENOISER */
    case 5928236058831373636u:
      sscanf_s(line, "%*s %d\n", denoiser);
      break;
    /* OUTPUTFN */
    case 5640288308724782415u:
      sscanf_s(line, "%*s %s\n", output_path, LINE_SIZE);
      break;
    default:
      break;
  }
}

static void parse_camera_settings(Camera* camera, char* line) {
  const uint64_t key = *((uint64_t*) line);

  switch (key) {
    /* POSITION */
    case 5642809484474797904u:
      sscanf_s(line, "%*s %f %f %f\n", &camera->pos.x, &camera->pos.y, &camera->pos.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf_s(line, "%*s %f %f %f\n", &camera->rotation.x, &camera->rotation.y, &camera->rotation.z);
      break;
    /* FOV_____ */
    case 6872316419616689990u:
      sscanf_s(line, "%*s %f\n", &camera->fov);
      break;
    /* FOCALLEN */
    case 5639997998747569990u:
      sscanf_s(line, "%*s %f\n", &camera->focal_length);
      break;
    /* APERTURE */
    case 4995148757353189441u:
      sscanf_s(line, "%*s %f\n", &camera->aperture_size);
      break;
    /* AUTOEXP_ */
    case 6868086486446921025u:
      sscanf_s(line, "%*s %f\n", &camera->auto_exposure);
      break;
    /* EXPOSURE */
    case 4995148753008613445u:
      sscanf_s(line, "%*s %f\n", &camera->exposure);
      break;
    /* BLOOM___ */
    case 6872316342038383682u:
      sscanf_s(line, "%*s %d\n", &camera->bloom);
      break;
    /* BLOOMSTR */
    case 5932458200661969986u:
      sscanf_s(line, "%*s %f\n", &camera->bloom_strength);
      break;
    /* DITHER__ */
    case 6872302013910370628u:
      sscanf_s(line, "%*s %d\n", &camera->dithering);
      break;
    /* FARCLIPD */
    case 4922514984611758406u:
      sscanf_s(line, "%*s %f\n", &camera->far_clip_distance);
      break;
    /* TONEMAP_ */
    case 6868061231871053652u:
      sscanf_s(line, "%*s %d\n", &camera->tonemap);
      break;
    /* ALPHACUT */
    case 6076837219871509569u:
      sscanf_s(line, "%*s %f\n", &camera->alpha_cutoff);
      break;
    default:
      break;
  }
}

static void parse_sky_settings(Sky* sky, char* line) {
  const uint64_t key = *((uint64_t*) line);

  switch (key) {
    /* SUNCOLOR */
    case 5931043137585567059u:
      sscanf_s(line, "%*s %f %f %f\n", &sky->sun_color.r, &sky->sun_color.g, &sky->sun_color.b);
      break;
    /* STRENGTH */
    case 5211869070270551123u:
      sscanf_s(line, "%*s %f\n", &sky->sun_strength);
      break;
    /* AZIMUTH_ */
    case 6865830357271927361u:
      sscanf_s(line, "%*s %f\n", &sky->azimuth);
      break;
    /* ALTITUDE */
    case 4991208107529227329u:
      sscanf_s(line, "%*s %f\n", &sky->altitude);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf_s(line, "%*s %f\n", &sky->base_density);
      break;
    /* RAYLEIGH */
    case 5208212056059756882u:
      sscanf_s(line, "%*s %f\n", &sky->rayleigh_falloff);
      break;
    /* MIE_____ */
    case 6872316419615574349u:
      sscanf_s(line, "%*s %f\n", &sky->mie_falloff);
      break;
    default:
      break;
  }
}

static void parse_ocean_settings(Ocean* ocean, char* line) {
  const uint64_t key = *((uint64_t*) line);

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(line, "%*s %d\n", &ocean->active);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf_s(line, "%*s %f\n", &ocean->height);
      break;
    /* AMPLITUD */
    case 4923934441389182273u:
      sscanf_s(line, "%*s %f\n", &ocean->amplitude);
      break;
    /* FREQUENC */
    case 4849890081462637126u:
      sscanf_s(line, "%*s %f\n", &ocean->frequency);
      break;
    /* CHOPPY__ */
    case 6872309757870295107u:
      sscanf_s(line, "%*s %f\n", &ocean->choppyness);
      break;
    /* SPEED___ */
    case 6872316303215251539u:
      sscanf_s(line, "%*s %f\n", &ocean->speed);
      break;
    /* ANIMATED */
    case 4919430807418392129u:
      sscanf_s(line, "%*s %d\n", &ocean->update);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf_s(line, "%*s %f %f %f %f\n", &ocean->albedo.r, &ocean->albedo.g, &ocean->albedo.b, &ocean->albedo.a);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf_s(line, "%*s %d\n", &ocean->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf_s(line, "%*s %f\n", &ocean->refractive_index);
      break;
    default:
      break;
  }
}

static void parse_toy_settings(Toy* toy, char* line) {
  const uint64_t key = *((uint64_t*) line);

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(line, "%*s %d\n", &toy->active);
      break;
    /* POSITION */
    case 5642809484474797904u:
      sscanf_s(line, "%*s %f %f %f\n", &toy->position.x, &toy->position.y, &toy->position.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf_s(line, "%*s %f %f %f\n", &toy->rotation.x, &toy->rotation.y, &toy->rotation.z);
      break;
    /* SHAPE___ */
    case 6872316307694504019u:
      sscanf_s(line, "%*s %d\n", &toy->shape);
      break;
    /* SCALE__ */
    case 6872316307627393875u:
      sscanf_s(line, "%*s %f\n", &toy->scale);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf_s(line, "%*s %f %f %f %f\n", &toy->albedo.r, &toy->albedo.g, &toy->albedo.b, &toy->albedo.a);
      break;
    /* MATERIAL */
    case 5494753638068011341u:
      sscanf_s(line, "%*s %f %f %f\n", &toy->material.r, &toy->material.g, &toy->material.b);
      break;
    /* EMISSION */
    case 5642809480346946885u:
      sscanf_s(line, "%*s %f %f %f\n", &toy->emission.r, &toy->emission.g, &toy->emission.b);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf_s(line, "%*s %d\n", &toy->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf_s(line, "%*s %f\n", &toy->refractive_index);
      break;
    default:
      break;
  }
}

Scene load_scene(const char* filename, RaytraceInstance** instance, char** output_name) {
  FILE* file;
  fopen_s(&file, filename, "rb");

  char* line = (char*) malloc(LINE_SIZE);

  sprintf(line, "Scene file \"%s\" could not be opened.", filename);

  assert((unsigned long long) file, line, 1);

  fgets(line, LINE_SIZE, file);

  assert(!validate_filetype(line), "Scene file is not a Luminary scene file!", 1);

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v') {
    int version = 0;
    sscanf_s(line, "%*c %d\n", &version);
    assert(version == CURRENT_VERSION, "Incompatible Scene version! Update the file or use an older version of Luminary!", 1);
  }
  else {
    print_error("Scene file has no version information, assuming correct version!")
  }

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

  int width   = 1280;
  int height  = 720;
  int bounces = 5;
  int samples = 16;

  int denoiser = 1;

  Wavefront_Content content = create_wavefront_content();

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'G') {
      parse_general_settings(&width, &height, &bounces, &samples, *output_name, &denoiser, &content, line + 7 + 1);
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

  Triangle* triangles;

  unsigned int triangle_count = convert_wavefront_content(&triangles, content);

  int nodes_length;

  Node2* initial_nodes = build_bvh_structure(&triangles, &triangle_count, &nodes_length);

  Node8* nodes = collapse_bvh(initial_nodes, nodes_length, &triangles, triangle_count, &nodes_length);

  free(initial_nodes);

  Traversal_Triangle* traversal_triangles = malloc(sizeof(Traversal_Triangle) * triangle_count);

  for (unsigned int i = 0; i < triangle_count; i++) {
    Triangle triangle     = triangles[i];
    Traversal_Triangle tt = {
      .vertex = {.x = triangle.vertex.x, .y = triangle.vertex.y, .z = triangle.vertex.z},
      .edge1  = {.x = triangle.edge1.x, .y = triangle.edge1.y, .z = triangle.edge1.z},
      .edge2  = {.x = triangle.edge2.x, .y = triangle.edge2.y, .z = triangle.edge2.z}};
    traversal_triangles[i] = tt;
    triangles[i]           = triangle;
  }

  scene.triangles           = triangles;
  scene.traversal_triangles = traversal_triangles;
  scene.triangles_length    = triangle_count;
  scene.nodes               = nodes;
  scene.nodes_length        = nodes_length;
  scene.materials_length    = content.materials_length;
  scene.texture_assignments = get_texture_assignments(content);

  process_lights(&scene);

  void* albedo_atlas      = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  void* illuminance_atlas = initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  void* material_atlas    = initialize_textures(content.material_maps, content.material_maps_length);

  *instance = init_raytracing(
    width, height, bounces, samples, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length,
    material_atlas, content.material_maps_length, scene, denoiser);

  free_wavefront_content(content);

  return scene;
}

void free_scene(Scene scene, RaytraceInstance* instance) {
  free_textures(instance->albedo_atlas, instance->albedo_atlas_length);
  free_textures(instance->illuminance_atlas, instance->illuminance_atlas_length);
  free_textures(instance->material_atlas, instance->material_atlas_length);

  free(scene.triangles);
  free(scene.nodes);
  free(scene.lights);
}
