#include "scene.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "light.h"
#include "log.h"
#include "png.h"
#include "raytrace.h"
#include "stars.h"
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
    case 4993446653056992589u: {
      char* source = (char*) malloc(LINE_SIZE);
      sscanf_s(value, "%s\n", source, LINE_SIZE);
      if (read_wavefront_file(source, content)) {
        error_message("Mesh file could not be loaded!");
      }
      if (general->mesh_files_count == general->mesh_files_length) {
        general->mesh_files_length *= 2;
        general->mesh_files = safe_realloc(general->mesh_files, sizeof(char*) * general->mesh_files_length);
      }
      general->mesh_files[general->mesh_files_count++] = source;
      break;
    }
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
      error_message("%8.8s (%zu) is not a valid GENERAL setting.", line, key);
      break;
  }
}

static void parse_material_settings(GlobalMaterial* material, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* LIGHTSON */
    case 5642820479573510476u:
      sscanf_s(value, "%d\n", &material->lights_active);
      break;
    /* SMOOTHNE */
    case 4994008563745508691u:
      sscanf_s(value, "%f\n", &material->default_material.r);
      break;
    /* METALLIC */
    case 4848490364238316877u:
      sscanf_s(value, "%f\n", &material->default_material.g);
      break;
    /* EMISSION */
    case 5642809480346946885u:
      sscanf_s(value, "%f\n", &material->default_material.b);
      break;
    /* DIFFUSE_ */
    case 6864984832712526148u:
      sscanf_s(value, "%d\n", &material->diffuse);
      break;
    /* FRESNEL_ */
    case 6866939734539981382u:
      sscanf_s(value, "%d\n", &material->fresnel);
      break;
    default:
      error_message("%8.8s (%zu) is not a valid MATERIAL setting.", line, key);
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
    /* BLOOMTHR */
    case 5929081600453069890u:
      sscanf_s(value, "%f\n", &camera->bloom_threshold);
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
      error_message("%8.8s (%zu) is not a valid CAMERA setting.", line, key);
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
    /* OFFSET__ */
    case 6872304213117257295u:
      sscanf_s(value, "%f %f %f\n", &sky->geometry_offset.x, &sky->geometry_offset.y, &sky->geometry_offset.z);
      break;
    /* MOONALTI */
    case 5283932106182840141u:
      sscanf_s(value, "%f\n", &sky->moon_altitude);
      break;
    /* MOONAZIM */
    case 5569081650753523533u:
      sscanf_s(value, "%f\n", &sky->moon_azimuth);
      break;
    /* MOONALBE */
    case 4990635180450336589u:
      sscanf_s(value, "%f\n", &sky->moon_albedo);
      break;
    /* INTENSIT */
    case 6073477168121531977u:
      sscanf_s(value, "%f\n", &sky->sky_intensity);
      break;
    /* SUNSTREN */
    case 5640004630479787347u:
      sscanf_s(value, "%f\n", &sky->sun_strength);
      break;
    /* OZONEABS */
    case 5999429419533294159u:
      sscanf_s(value, "%d\n", &sky->ozone_absorption);
      break;
    /* STEPS___ */
    case 6872316367824311379u:
      sscanf_s(value, "%d\n", &sky->steps);
      break;
    /* SHASTEPS */
    case 6003374531761227859u:
      sscanf_s(value, "%d\n", &sky->shadow_steps);
      break;
    /* STARSEED */
    case 4919414392136750163u:
      sscanf_s(value, "%d\n", &sky->stars_seed);
      break;
    /* STARINTE */
    case 4995703963480314963u:
      sscanf_s(value, "%f\n", &sky->stars_intensity);
      break;
    /* STARNUM_ */
    case 6867238801685697619u:
      sscanf_s(value, "%d\n", &sky->settings_stars_count);
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
    default:
      error_message("%8.8s (%zu) is not a valid SKY setting.", line, key);
      break;
  }
}

static void parse_cloud_settings(Cloud* cloud, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(value, "%d\n", &cloud->active);
      break;
    /* SEED___ */
    case 6872316419162588499u:
      sscanf_s(value, "%d\n", &cloud->seed);
      break;
    /* OFFSET__ */
    case 6872304213117257295u:
      sscanf_s(value, "%f %f\n", &cloud->offset_x, &cloud->offset_z);
      break;
    /* HEIGHTMA */
    case 4705509855082399048u:
      sscanf_s(value, "%f\n", &cloud->height_max);
      break;
    /* HEIGHTMI */
    case 5281970607385822536u:
      sscanf_s(value, "%f\n", &cloud->height_min);
      break;
    /* SHASCALE */
    case 4993437844262438995u:
      sscanf_s(value, "%f\n", &cloud->noise_shape_scale);
      break;
    /* DETSCALE */
    case 4993437844263683396u:
      sscanf_s(value, "%f\n", &cloud->noise_detail_scale);
      break;
    /* WEASCALE */
    case 4993437844262438231u:
      sscanf_s(value, "%f\n", &cloud->noise_weather_scale);
      break;
    /* CURSCALE */
    case 4993437844263556419u:
      sscanf_s(value, "%f\n", &cloud->noise_curl_scale);
      break;
    /* COVERAGE */
    case 4992030533569892163u:
      sscanf_s(value, "%f\n", &cloud->coverage);
      break;
    /* COVERMIN */
    case 5641125024004198211u:
      sscanf_s(value, "%f\n", &cloud->coverage_min);
      break;
    /* ANVIL___ */
    case 6872316337643212353u:
      sscanf_s(value, "%f\n", &cloud->anvil);
      break;
    /* FWDSCATT */
    case 6076553554645243718u:
      sscanf_s(value, "%f\n", &cloud->forward_scattering);
      break;
    /* BWDSCATT */
    case 6076553554645243714u:
      sscanf_s(value, "%f\n", &cloud->backward_scattering);
      break;
    /* SCATLERP */
    case 5787764665257902931u:
      sscanf_s(value, "%f\n", &cloud->lobe_lerp);
      break;
    /* WETNESS_ */
    case 6868925413802132823u:
      sscanf_s(value, "%f\n", &cloud->wetness);
      break;
    /* POWDER__ */
    case 6872302013843459920u:
      sscanf_s(value, "%f\n", &cloud->powder);
      break;
    /* SHASTEPS */
    case 6003374531761227859u:
      sscanf_s(value, "%d\n", &cloud->shadow_steps);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf_s(value, "%f\n", &cloud->density);
      break;
    default:
      error_message("%8.8s (%zu) is not a valid CLOUD setting.", line, key);
      break;
  }
}

static void parse_fog_settings(Fog* fog, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf_s(value, "%d\n", &fog->active);
      break;
    /* ABSORPTI */
    case 5283936577260831297u:
      sscanf_s(value, "%f\n", &fog->absorption);
      break;
    /* SCATTERI */
    case 5283361541352145747u:
      sscanf_s(value, "%f\n", &fog->scattering);
      break;
    /* ANISOTRO */
    case 5715723576763043393u:
      sscanf_s(value, "%f\n", &fog->anisotropy);
      break;
    /* DISTANCE */
    case 4990918854551226692u:
      sscanf_s(value, "%f\n", &fog->dist);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf_s(value, "%f\n", &fog->height);
      break;
    /* FALLOFF_ */
    case 6865251988369326406u:
      sscanf_s(value, "%f\n", &fog->falloff);
      break;
    default:
      error_message("%8.8s (%zu) is not a valid FOG setting.", line, key);
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
      error_message("%8.8s (%zu) is not a valid OCEAN setting.", line, key);
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
      error_message("%8.8s (%zu) is not a valid TOY setting.", line, key);
      break;
  }
}

static Scene get_default_scene() {
  Scene scene;

  memset(&scene, 0, sizeof(Scene));

  scene.material.lights_active      = 0;
  scene.material.default_material.r = 0.3f;
  scene.material.default_material.g = 0.0f;
  scene.material.default_material.b = 1.0f;
  scene.material.fresnel            = FDEZ_AGUERA;
  scene.material.diffuse            = LAMBERTIAN;

  scene.camera.pos.x                 = 0.0f;
  scene.camera.pos.y                 = 0.0f;
  scene.camera.pos.z                 = 0.0f;
  scene.camera.rotation.x            = 0.0f;
  scene.camera.rotation.y            = 0.0f;
  scene.camera.rotation.z            = 0.0f;
  scene.camera.fov                   = 1.0f;
  scene.camera.focal_length          = 1.0f;
  scene.camera.aperture_size         = 0.00f;
  scene.camera.exposure              = 1.0f;
  scene.camera.auto_exposure         = 1;
  scene.camera.bloom                 = 1;
  scene.camera.bloom_strength        = 1.0f;
  scene.camera.bloom_threshold       = 1.0f;
  scene.camera.dithering             = 1;
  scene.camera.alpha_cutoff          = 0.0f;
  scene.camera.far_clip_distance     = 50000.0f;
  scene.camera.tonemap               = TONEMAP_ACES;
  scene.camera.filter                = FILTER_NONE;
  scene.camera.wasd_speed            = 1.0f;
  scene.camera.mouse_speed           = 1.0f;
  scene.camera.smooth_movement       = 0;
  scene.camera.smoothing_factor      = 0.1f;
  scene.camera.temporal_blend_factor = 0.15f;
  scene.camera.purkinje              = 1;
  scene.camera.purkinje_kappa1       = 0.2f;
  scene.camera.purkinje_kappa2       = 0.29f;

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
  scene.toy.position.y       = 10.0f;
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

  scene.sky.geometry_offset.x         = 0.0f;
  scene.sky.geometry_offset.y         = 0.0f;
  scene.sky.geometry_offset.z         = 0.0f;
  scene.sky.sun_color.r               = 1.0f;
  scene.sky.sun_color.g               = 1.0f;
  scene.sky.sun_color.b               = 1.0f;
  scene.sky.altitude                  = 0.5f;
  scene.sky.azimuth                   = 3.141f;
  scene.sky.moon_altitude             = -0.5f;
  scene.sky.moon_azimuth              = 0.0f;
  scene.sky.moon_albedo               = 0.12f;
  scene.sky.sky_intensity             = 20.0f;
  scene.sky.sun_strength              = 10.0f;
  scene.sky.base_density              = 0.6f;
  scene.sky.steps                     = 16;
  scene.sky.shadow_steps              = 128;
  scene.sky.ozone_absorption          = 0;
  scene.sky.stars_seed                = 0;
  scene.sky.stars_intensity           = 1.0f;
  scene.sky.settings_stars_count      = 10000;
  scene.sky.cloud.active              = 0;
  scene.sky.cloud.initialized         = 0;
  scene.sky.cloud.seed                = 1;
  scene.sky.cloud.offset_x            = 0.0f;
  scene.sky.cloud.offset_z            = 0.0f;
  scene.sky.cloud.height_max          = 4000.0f;
  scene.sky.cloud.height_min          = 1500.0f;
  scene.sky.cloud.noise_shape_scale   = 1.0f;
  scene.sky.cloud.noise_detail_scale  = 1.0f;
  scene.sky.cloud.noise_weather_scale = 1.0f;
  scene.sky.cloud.noise_curl_scale    = 1.0f;
  scene.sky.cloud.coverage            = 2.0f;
  scene.sky.cloud.anvil               = 0.0f;
  scene.sky.cloud.coverage_min        = 1.05f;
  scene.sky.cloud.forward_scattering  = 0.8f;
  scene.sky.cloud.backward_scattering = -0.2f;
  scene.sky.cloud.lobe_lerp           = 0.5f;
  scene.sky.cloud.wetness             = 0.0f;
  scene.sky.cloud.powder              = 0.5f;
  scene.sky.cloud.shadow_steps        = 16;
  scene.sky.cloud.density             = 1.0f;

  scene.fog.active     = 0;
  scene.fog.absorption = 1.0f;
  scene.fog.scattering = 1.0f;
  scene.fog.anisotropy = 0.0f;
  scene.fog.height     = 1000.0f;
  scene.fog.dist       = 100.0f;
  scene.fog.falloff    = 10.0f;

  return scene;
}

static void convert_wavefront_to_internal(Wavefront_Content content, Scene* scene) {
  scene->triangles_length = convert_wavefront_content(&scene->triangles, content);

  Node2* initial_nodes = build_bvh_structure(&scene->triangles, &scene->triangles_length, &scene->nodes_length);

  if (!scene->triangles_length) {
    crash_message("No triangles are left. Did the scene not contain any faces?");
  }

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
    error_message("Scene file has no version information, assuming correct version!");
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
    else if (line[0] == 'M') {
      parse_material_settings(&scene.material, line + 8 + 1);
    }
    else if (line[0] == 'C' && line[1] == 'A') {
      parse_camera_settings(&scene.camera, line + 6 + 1);
    }
    else if (line[0] == 'S') {
      parse_sky_settings(&scene.sky, line + 3 + 1);
    }
    else if (line[0] == 'C' && line[1] == 'L') {
      parse_cloud_settings(&scene.sky.cloud, line + 5 + 1);
    }
    else if (line[0] == 'F') {
      parse_fog_settings(&scene.fog, line + 3 + 1);
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
      error_message("Scene file contains unknown line!\n Content: %s", line);
    }
  }

  fclose(file);
  free(line);

  assert(general.mesh_files_count, "No mesh files where loaded.", 1);

  convert_wavefront_to_internal(content, &scene);

  process_lights(&scene);

  DeviceBuffer* albedo_atlas      = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  DeviceBuffer* illuminance_atlas = initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  DeviceBuffer* material_atlas    = initialize_textures(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  generate_stars(instance);
  generate_clouds(instance);

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

  DeviceBuffer* albedo_atlas      = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  DeviceBuffer* illuminance_atlas = initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  DeviceBuffer* material_atlas    = initialize_textures(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  generate_stars(instance);
  generate_clouds(instance);

  return instance;
}

void serialize_scene(RaytraceInstance* instance) {
  FILE* file;
  fopen_s(&file, "generated.lum", "wb");

  if (!file) {
    error_message("Could not export settings!");
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
  sprintf_s(line, LINE_SIZE, "CAMERA BLOOMTHR %f\n", instance->scene_gpu.camera.bloom_threshold);
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

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# MATERIAL Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "MATERIAL LIGHTSON %d\n", instance->scene_gpu.material.lights_active);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "MATERIAL SMOOTHNE %f\n", instance->scene_gpu.material.default_material.r);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "MATERIAL METALLIC %f\n", instance->scene_gpu.material.default_material.g);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "MATERIAL EMISSION %f\n", instance->scene_gpu.material.default_material.b);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "MATERIAL DIFFUSE_ %d\n", instance->scene_gpu.material.diffuse);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "MATERIAL FRESNEL_ %d\n", instance->scene_gpu.material.fresnel);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Sky Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(
    line, LINE_SIZE, "SKY SUNCOLOR %f %f %f\n", instance->scene_gpu.sky.sun_color.r, instance->scene_gpu.sky.sun_color.g,
    instance->scene_gpu.sky.sun_color.b);
  fputs(line, file);
  sprintf_s(
    line, LINE_SIZE, "SKY OFFSET__ %f %f %f\n", instance->scene_gpu.sky.geometry_offset.x, instance->scene_gpu.sky.geometry_offset.y,
    instance->scene_gpu.sky.geometry_offset.z);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY AZIMUTH_ %f\n", instance->scene_gpu.sky.azimuth);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY ALTITUDE %f\n", instance->scene_gpu.sky.altitude);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY MOONALTI %f\n", instance->scene_gpu.sky.moon_altitude);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY MOONAZIM %f\n", instance->scene_gpu.sky.moon_azimuth);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY MOONALBE %f\n", instance->scene_gpu.sky.moon_albedo);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY INTENSIT %f\n", instance->scene_gpu.sky.sky_intensity);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY SUNSTREN %f\n", instance->scene_gpu.sky.sun_strength);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY DENSITY_ %f\n", instance->scene_gpu.sky.base_density);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY OZONEABS %d\n", instance->scene_gpu.sky.ozone_absorption);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY STEPS___ %d\n", instance->scene_gpu.sky.steps);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY SHASTEPS %d\n", instance->scene_gpu.sky.shadow_steps);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY STARSEED %d\n", instance->scene_gpu.sky.stars_seed);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY STARINTE %f\n", instance->scene_gpu.sky.stars_intensity);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "SKY STARNUM_ %d\n", instance->scene_gpu.sky.settings_stars_count);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Cloud Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "CLOUD ACTIVE__ %d\n", instance->scene_gpu.sky.cloud.active);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD SEED____ %d\n", instance->scene_gpu.sky.cloud.seed);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD OFFSET__ %f %f\n", instance->scene_gpu.sky.cloud.offset_x, instance->scene_gpu.sky.cloud.offset_z);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD HEIGHTMA %f\n", instance->scene_gpu.sky.cloud.height_max);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD HEIGHTMI %f\n", instance->scene_gpu.sky.cloud.height_min);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD SHASCALE %f\n", instance->scene_gpu.sky.cloud.noise_shape_scale);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD DETSCALE %f\n", instance->scene_gpu.sky.cloud.noise_detail_scale);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD WEASCALE %f\n", instance->scene_gpu.sky.cloud.noise_weather_scale);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD CURSCALE %f\n", instance->scene_gpu.sky.cloud.noise_curl_scale);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD COVERAGE %f\n", instance->scene_gpu.sky.cloud.coverage);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD COVERMIN %f\n", instance->scene_gpu.sky.cloud.coverage_min);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD ANVIL___ %f\n", instance->scene_gpu.sky.cloud.anvil);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD FWDSCATT %f\n", instance->scene_gpu.sky.cloud.forward_scattering);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD BWDSCATT %f\n", instance->scene_gpu.sky.cloud.backward_scattering);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD SCATLERP %f\n", instance->scene_gpu.sky.cloud.lobe_lerp);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD WETNESS_ %f\n", instance->scene_gpu.sky.cloud.wetness);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD POWDER__ %f\n", instance->scene_gpu.sky.cloud.powder);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD SHASTEPS %d\n", instance->scene_gpu.sky.cloud.shadow_steps);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "CLOUD DENSITY_ %f\n", instance->scene_gpu.sky.cloud.density);
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "\n#===============================\n# Fog Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf_s(line, LINE_SIZE, "FOG ACTIVE__ %d\n", instance->scene_gpu.fog.active);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG ABSORPTI %f\n", instance->scene_gpu.fog.absorption);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG SCATTERI %f\n", instance->scene_gpu.fog.scattering);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG ANISOTRO %f\n", instance->scene_gpu.fog.anisotropy);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG DISTANCE %f\n", instance->scene_gpu.fog.dist);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG HEIGHT__ %f\n", instance->scene_gpu.fog.height);
  fputs(line, file);
  sprintf_s(line, LINE_SIZE, "FOG FALLOFF_ %f\n", instance->scene_gpu.fog.falloff);
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
