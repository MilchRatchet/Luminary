#include "lum.h"

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "wavefront.h"

static const int LINE_SIZE       = 4096;
static const int CURRENT_VERSION = 4;

static void parse_general_settings(General* general, WavefrontContent* content, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* MESHFILE */
    case 4993446653056992589u: {
      char* source = (char*) malloc(LINE_SIZE);
      sscanf(value, "%s\n", source);
      if (wavefront_read_file(content, source)) {
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
      sscanf(value, "%d\n", &general->width);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf(value, "%d\n", &general->height);
      break;
    /* BOUNCES_ */
    case 6868910012049477442u:
      sscanf(value, "%d\n", &general->max_ray_depth);
      break;
    /* SAMPLES_ */
    case 6868910050737209683u:
      sscanf(value, "%d\n", &general->samples);
      break;
    /* NUMLIGHT */
    case 6073182477647435086u:
      sscanf(value, "%d\n", &general->num_light_ray);
      break;
    /* DENOISER */
    case 5928236058831373636u:
      sscanf(value, "%d\n", &general->denoiser);
      break;
    /* OUTPUTFN */
    case 5640288308724782415u:
      sscanf(value, "%s\n", general->output_path);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid GENERAL setting.", line, key);
      break;
  }
}

static void parse_material_settings(GlobalMaterial* material, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* LIGHTSON */
    case 5642820479573510476u:
      sscanf(value, "%d\n", &material->lights_active);
      break;
    /* OVERRIDE */
    case 4991194904949773903u:
      sscanf(value, "%d\n", &material->override_materials);
      break;
    /* SMOOTHNE */
    case 4994008563745508691u:
      sscanf(value, "%f\n", &material->default_material.r);
      break;
    /* METALLIC */
    case 4848490364238316877u:
      sscanf(value, "%f\n", &material->default_material.g);
      break;
    /* EMISSION */
    case 5642809480346946885u:
      sscanf(value, "%f\n", &material->default_material.b);
      break;
    /* ALPHACUT */
    case 6076837219871509569u:
      sscanf(value, "%f\n", &material->alpha_cutoff);
      break;
    /* COLORTRA */
    case 4706917273050042179u:
      sscanf(value, "%d\n", &material->colored_transparency);
      break;
    /* IORSHADO */
    case 5711762006303985481u:
      sscanf(value, "%d\n", &material->enable_ior_shadowing);
      break;
    /* INTERTRO */
    case 5715723589413916233u:
      sscanf(value, "%d\n", &material->invert_roughness);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid MATERIAL setting.", line, key);
      break;
  }
}

static void parse_camera_settings(Camera* camera, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* POSITION */
    case 5642809484474797904u:
      sscanf(value, "%f %f %f\n", &camera->pos.x, &camera->pos.y, &camera->pos.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf(value, "%f %f %f\n", &camera->rotation.x, &camera->rotation.y, &camera->rotation.z);
      break;
    /* FOV_____ */
    case 6872316419616689990u:
      sscanf(value, "%f\n", &camera->fov);
      break;
    /* FOCALLEN */
    case 5639997998747569990u:
      sscanf(value, "%f\n", &camera->focal_length);
      break;
    /* APERTURE */
    case 4995148757353189441u:
      sscanf(value, "%f\n", &camera->aperture_size);
      break;
    /* APESHAPE */
    case 4994563765644382273u:
      sscanf(value, "%d\n", &camera->aperture_shape);
      break;
    /* APEBLACO */
    case 5711480548221079617u:
      sscanf(value, "%d\n", &camera->aperture_blade_count);
      break;
    /* AUTOEXP_ */
    case 6868086486446921025u:
      sscanf(value, "%d\n", &camera->auto_exposure);
      break;
    /* EXPOSURE */
    case 4995148753008613445u:
      sscanf(value, "%f\n", &camera->exposure);
      break;
    /* MINEXPOS */
    case 6003105168358263117u:
      sscanf(value, "%f\n", &camera->min_exposure);
      break;
    /* MAXEXPOS */
    case 6003105168358916429u:
      sscanf(value, "%f\n", &camera->max_exposure);
      break;
    /* BLOOM___ */
    case 6872316342038383682u:
      sscanf(value, "%d\n", &camera->bloom);
      break;
    /* BLOOMBLE */
    case 4993438986657549378u:
      sscanf(value, "%f\n", &camera->bloom_blend);
      break;
    /* LENSFLAR */
    case 5927102449525343564u:
      sscanf(value, "%d\n", &camera->lens_flare);
      break;
    /* LENSFTHR */
    case 5929081570455340364u:
      sscanf(value, "%f\n", &camera->lens_flare_threshold);
      break;
    /* DITHER__ */
    case 6872302013910370628u:
      sscanf(value, "%d\n", &camera->dithering);
      break;
    /* FARCLIPD */
    case 4922514984611758406u:
      sscanf(value, "%f\n", &camera->far_clip_distance);
      break;
    /* TONEMAP_ */
    case 6868061231871053652u:
      sscanf(value, "%d\n", &camera->tonemap);
      break;
    /* AGXSLOPE */
    case 4994579175988283201u:
      sscanf(value, "%f\n", &camera->agx_custom_slope);
      break;
    /* AGXPOWER */
    case 5928240482665121601u:
      sscanf(value, "%f\n", &camera->agx_custom_power);
      break;
    /* AGXSATUR */
    case 5932740723678398273u:
      sscanf(value, "%f\n", &camera->agx_custom_saturation);
      break;
    /* FILTER__ */
    case 6872302014111172934u:
      sscanf(value, "%d\n", &camera->filter);
      break;
    /* PURKINJE */
    case 4992889213596882256u:
      sscanf(value, "%d\n", &camera->purkinje);
      break;
    /* RUSSIANR */
    case 5930749542479910226u:
      sscanf(value, "%f\n", &camera->russian_roulette_threshold);
      break;
    /* FIREFLYC */
    case 4852993938162862406u:
      sscanf(value, "%d\n", &camera->do_firefly_clamping);
      break;
    /* FILMGRAI */
    case 5278590704447932742u:
      sscanf(value, "%f\n", &camera->film_grain);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid CAMERA setting.", line, key);
      break;
  }
}

static void parse_sky_settings(Sky* sky, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* OFFSET__ */
    case 6872304213117257295u:
      sscanf(value, "%f %f %f\n", &sky->geometry_offset.x, &sky->geometry_offset.y, &sky->geometry_offset.z);
      break;
    /* MOONALTI */
    case 5283932106182840141u:
      sscanf(value, "%f\n", &sky->moon_altitude);
      break;
    /* MOONAZIM */
    case 5569081650753523533u:
      sscanf(value, "%f\n", &sky->moon_azimuth);
      break;
    /* MOONTEXO */
    case 5717395955340234573u:
      sscanf(value, "%f\n", &sky->moon_tex_offset);
      break;
    /* SUNSTREN */
    case 5640004630479787347u:
      sscanf(value, "%f\n", &sky->sun_strength);
      break;
    /* OZONEABS */
    case 5999429419533294159u:
      sscanf(value, "%d\n", &sky->ozone_absorption);
      break;
    /* STEPS___ */
    case 6872316367824311379u:
      sscanf(value, "%d\n", &sky->steps);
      break;
    /* STARSEED */
    case 4919414392136750163u:
      sscanf(value, "%d\n", &sky->stars_seed);
      break;
    /* STARINTE */
    case 4995703963480314963u:
      sscanf(value, "%f\n", &sky->stars_intensity);
      break;
    /* STARNUM_ */
    case 6867238801685697619u:
      sscanf(value, "%d\n", &sky->settings_stars_count);
      break;
    /* AZIMUTH_ */
    case 6865830357271927361u:
      sscanf(value, "%f\n", &sky->azimuth);
      break;
    /* ALTITUDE */
    case 4991208107529227329u:
      sscanf(value, "%f\n", &sky->altitude);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf(value, "%f\n", &sky->base_density);
      break;
    /* RAYLEDEN */
    case 5639989172775764306u:
      sscanf(value, "%f\n", &sky->rayleigh_density);
      break;
    /* MIEDENSI */
    case 5283652847240825165u:
      sscanf(value, "%f\n", &sky->mie_density);
      break;
    /* OZONEDEN */
    case 5639989172808669775u:
      sscanf(value, "%f\n", &sky->ozone_density);
      break;
    /* RAYLEFAL */
    case 5494750283816321362u:
      sscanf(value, "%f\n", &sky->rayleigh_falloff);
      break;
    /* MIEFALLO */
    case 5714025870461847885u:
      sscanf(value, "%f\n", &sky->mie_falloff);
      break;
    /* GROUNDVI */
    case 5284486315995255367u:
      sscanf(value, "%f\n", &sky->ground_visibility);
      break;
    /* DIAMETER */
    case 5928237141128726852u:
      sscanf(value, "%f\n", &sky->mie_diameter);
      break;
    /* OZONETHI */
    case 5280563219735206479u:
      sscanf(value, "%f\n", &sky->ozone_layer_thickness);
      break;
    /* MSFACTOR */
    case 5931051882104902477u:
      sscanf(value, "%f\n", &sky->multiscattering_factor);
      break;
    /* AERIALPE */
    case 4994575830040593729u:
      sscanf(value, "%d\n", &sky->aerial_perspective);
      break;
    /* HDRIACTI */
    case 5283922210494497864u:
      sscanf(value, "%d\n", &sky->hdri_active);
      break;
    /* HDRIDIM_ */
    case 6867225564446606408u:
      sscanf(value, "%d\n", &sky->settings_hdri_dim);
      sscanf(value, "%d\n", &sky->hdri_dim);
      break;
      /* HDRISAMP */
    case 5786352922209174600u:
      sscanf(value, "%d\n", &sky->hdri_samples);
      break;
      /* HDRIMIPB */
    case 4778399800931533896u:
      sscanf(value, "%f\n", &sky->hdri_mip_bias);
      break;
      /* HDRIORIG */
    case 5136727350478783560u:
      sscanf(value, "%f %f %f\n", &sky->hdri_origin.x, &sky->hdri_origin.y, &sky->hdri_origin.z);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid SKY setting.", line, key);
      break;
  }
}

static void parse_cloud_settings(Cloud* cloud, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%d\n", &cloud->active);
      break;
    /* INSCATTE */
    case 4995710525939863113u:
      sscanf(value, "%d\n", &cloud->atmosphere_scattering);
      break;
    /* MIPMAPBI */
    case 5278869954631846221u:
      sscanf(value, "%f\n", &cloud->mipmap_bias);
      break;
    /* SEED___ */
    case 6872316419162588499u:
      sscanf(value, "%d\n", &cloud->seed);
      break;
    /* OFFSET__ */
    case 6872304213117257295u:
      sscanf(value, "%f %f\n", &cloud->offset_x, &cloud->offset_z);
      break;
    /* SHASCALE */
    case 4993437844262438995u:
      sscanf(value, "%f\n", &cloud->noise_shape_scale);
      break;
    /* DETSCALE */
    case 4993437844263683396u:
      sscanf(value, "%f\n", &cloud->noise_detail_scale);
      break;
    /* WEASCALE */
    case 4993437844262438231u:
      sscanf(value, "%f\n", &cloud->noise_weather_scale);
      break;
    /* DIAMETER */
    case 5928237141128726852u:
      sscanf(value, "%f\n", &cloud->droplet_diameter);
      break;
    /* SHASTEPS */
    case 6003374531761227859u:
      sscanf(value, "%d\n", &cloud->shadow_steps);
      break;
    /* STEPS___ */
    case 6872316367824311379u:
      sscanf(value, "%d\n", &cloud->steps);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf(value, "%f\n", &cloud->density);
      break;
    /* LOWACTIV */
    case 6217593408397463372u:
      sscanf(value, "%d\n", &cloud->low.active);
      break;
    /* LOWCOVER */
    case 5928239382935326540u:
      sscanf(value, "%f %f\n", &cloud->low.coverage_min, &cloud->low.coverage);
      break;
    /* LOWTYPE_ */
    case 6864981551593508684u:
      sscanf(value, "%f %f\n", &cloud->low.type_min, &cloud->low.type);
      break;
    /* LOWHEIGH */
    case 5208212055992520524u:
      sscanf(value, "%f %f\n", &cloud->low.height_min, &cloud->low.height_max);
      break;
    /* LOWWIND_ */
    case 6864697808924397388u:
      sscanf(value, "%f %f\n", &cloud->low.wind_speed, &cloud->low.wind_angle);
      break;
    /* MIDACTIV */
    case 6217593408396216653u:
      sscanf(value, "%d\n", &cloud->mid.active);
      break;
    /* MIDCOVER */
    case 5928239382934079821u:
      sscanf(value, "%f %f\n", &cloud->mid.coverage_min, &cloud->mid.coverage);
      break;
    /* MIDTYPE_ */
    case 6864981551592261965u:
      sscanf(value, "%f %f\n", &cloud->mid.type_min, &cloud->mid.type);
      break;
    /* MIDHEIGH */
    case 5208212055991273805u:
      sscanf(value, "%f %f\n", &cloud->mid.height_min, &cloud->mid.height_max);
      break;
    /* MIDWIND_ */
    case 6864697808923150669u:
      sscanf(value, "%f %f\n", &cloud->mid.wind_speed, &cloud->mid.wind_angle);
      break;
    /* TOPACTIV */
    case 6217593408397004628u:
      sscanf(value, "%d\n", &cloud->top.active);
      break;
    /* TOPCOVER */
    case 5928239382934867796u:
      sscanf(value, "%f %f\n", &cloud->top.coverage_min, &cloud->top.coverage);
      break;
    /* TOPTYPE_ */
    case 6864981551593049940u:
      sscanf(value, "%f %f\n", &cloud->top.type_min, &cloud->top.type);
      break;
    /* TOPHEIGH */
    case 5208212055992061780u:
      sscanf(value, "%f %f\n", &cloud->top.height_min, &cloud->top.height_max);
      break;
    /* TOPWIND_ */
    case 6864697808923938644u:
      sscanf(value, "%f %f\n", &cloud->top.wind_speed, &cloud->top.wind_angle);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid CLOUD setting.", line, key);
      break;
  }
}

static void parse_fog_settings(Fog* fog, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%d\n", &fog->active);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf(value, "%f\n", &fog->density);
      break;
    /* DIAMETER */
    case 5928237141128726852u:
      sscanf(value, "%f\n", &fog->droplet_diameter);
      break;
    /* DISTANCE */
    case 4990918854551226692u:
      sscanf(value, "%f\n", &fog->dist);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf(value, "%f\n", &fog->height);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid FOG setting.", line, key);
      break;
  }
}

static void parse_ocean_settings(Ocean* ocean, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%d\n", &ocean->active);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf(value, "%f\n", &ocean->height);
      break;
    /* AMPLITUD */
    case 4923934441389182273u:
      sscanf(value, "%f\n", &ocean->amplitude);
      break;
    /* FREQUENC */
    case 4849890081462637126u:
      sscanf(value, "%f\n", &ocean->frequency);
      break;
    /* CHOPPY__ */
    case 6872309757870295107u:
      sscanf(value, "%f\n", &ocean->choppyness);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf(value, "%f\n", &ocean->refractive_index);
      break;
    /* WATERTYP */
    case 5789751508288684375u:
      sscanf(value, "%d\n", &ocean->water_type);
      break;
    /* CAUSACTI */
    case 5283922210662465859u:
      sscanf(value, "%d\n", &ocean->caustics_active);
      break;
    /* CAUSRISS */
    case 6004223346149245251u:
      sscanf(value, "%d\n", &ocean->caustics_ris_sample_count);
      break;
    /* CAUSREGU */
    case 6144956436458062147u:
      sscanf(value, "%f\n", &ocean->caustics_regularization);
      break;
    /* CAUSSCAL */
    case 5494747045528158531u:
      sscanf(value, "%f\n", &ocean->caustics_domain_scale);
      break;
    /* MULTISCA */
    case 4702694010316936525:
      sscanf(value, "%d\n", &ocean->multiscattering);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid OCEAN setting.", line, key);
      break;
  }
}

static void parse_particle_settings(Particles* particle, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%d\n", &particle->active);
      break;
    /* SCALE___ */
    case 6872316307627393875u:
      sscanf(value, "%f\n", &particle->scale);
      break;
    /* ALBEDO__ */
    case 6872298711029009473u:
      sscanf(value, "%f %f %f\n", &particle->albedo.r, &particle->albedo.g, &particle->albedo.b);
      break;
    /* DIRECTIO */
    case 5713190250198747460u:
      sscanf(value, "%f %f\n", &particle->direction_altitude, &particle->direction_azimuth);
      break;
    /* SPEED___ */
    case 6872316303215251539u:
      sscanf(value, "%f\n", &particle->speed);
      break;
    /* PHASEDIA */
    case 4704366350305413200u:
      sscanf(value, "%f\n", &particle->phase_diameter);
      break;
    /* SEED____ */
    case 6872316419162588499u:
      sscanf(value, "%d\n", &particle->seed);
      break;
    /* COUNT___ */
    case 6872316372086771523u:
      sscanf(value, "%d\n", &particle->count);
      break;
    /* SIZE____ */
    case 6872316419180742995u:
      sscanf(value, "%f\n", &particle->size);
      break;
    /* SIZEVARI */
    case 5283357151645550931u:
      sscanf(value, "%f\n", &particle->size_variation);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid PARTICLE setting.", line, key);
      break;
  }
}

static void parse_toy_settings(Toy* toy, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%d\n", &toy->active);
      break;
    /* POSITION */
    case 5642809484474797904u:
      sscanf(value, "%f %f %f\n", &toy->position.x, &toy->position.y, &toy->position.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf(value, "%f %f %f\n", &toy->rotation.x, &toy->rotation.y, &toy->rotation.z);
      break;
    /* SHAPE___ */
    case 6872316307694504019u:
      sscanf(value, "%d\n", &toy->shape);
      break;
    /* SCALE__ */
    case 6872316307627393875u:
      sscanf(value, "%f\n", &toy->scale);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf(value, "%f %f %f %f\n", &toy->albedo.r, &toy->albedo.g, &toy->albedo.b, &toy->albedo.a);
      break;
    /* MATERIAL */
    case 5494753638068011341u:
      sscanf(value, "%f %f %f\n", &toy->material.r, &toy->material.g, &toy->material.b);
      break;
    /* EMISSION */
    case 5642809480346946885u:
      sscanf(value, "%f %f %f\n", &toy->emission.r, &toy->emission.g, &toy->emission.b);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf(value, "%d\n", &toy->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf(value, "%f\n", &toy->refractive_index);
      break;
    /* FLASHLIG */
    case 5136720723510905926u:
      sscanf(value, "%d\n", &toy->flashlight_mode);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid TOY setting.", line, key);
      break;
  }
}

/*
 * Determines whether the file is a supported *.lum file and on success puts the file accessor past the header.
 * @param file File handle.
 * @result 0 if file is supported, non zero value else.
 */
int lum_validate_file(FILE* file) {
  if (!file)
    return -1;

  char* line = (char*) malloc(LINE_SIZE);

  fgets(line, LINE_SIZE, file);

  {
    int result = 0;

    result += line[0] ^ 'L';
    result += line[1] ^ 'u';
    result += line[2] ^ 'm';
    result += line[3] ^ 'i';
    result += line[4] ^ 'n';
    result += line[5] ^ 'a';
    result += line[6] ^ 'r';
    result += line[7] ^ 'y';

    if (result) {
      error_message("Scene file does not identify as Luminary file.");
      return -1;
    }
  }

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v' || line[0] == 'V') {
    int version = 0;
    sscanf(line, "%*c %d\n", &version);

    if (version != CURRENT_VERSION) {
      error_message("Incompatible Scene file version! Update the file or use an older version of Luminary!");
      return -1;
    }
  }
  else {
    error_message(
      "Scene file has no version information, assuming correct version!\nMake sure that the second line of *.lum file reads as \"V %d\".",
      CURRENT_VERSION);
  }

  free(line);

  return 0;
}

/*
 * Parses *.lum and writes values into structure instances. Assumes file handle to point past header.
 * @param file File handle.
 * @param scene Scene instance.
 * @param general General instance.
 * @param content WavefrontContent instance.
 */
void lum_parse_file(FILE* file, Scene* scene, General* general, WavefrontContent* content) {
  char* line = (char*) malloc(LINE_SIZE);

  while (1) {
    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;

    if (line[0] == 'G') {
      parse_general_settings(general, content, line + 7 + 1);
    }
    else if (line[0] == 'M') {
      parse_material_settings(&scene->material, line + 8 + 1);
    }
    else if (line[0] == 'C' && line[1] == 'A') {
      parse_camera_settings(&scene->camera, line + 6 + 1);
    }
    else if (line[0] == 'S') {
      parse_sky_settings(&scene->sky, line + 3 + 1);
    }
    else if (line[0] == 'C' && line[1] == 'L') {
      parse_cloud_settings(&scene->sky.cloud, line + 5 + 1);
    }
    else if (line[0] == 'F') {
      parse_fog_settings(&scene->fog, line + 3 + 1);
    }
    else if (line[0] == 'O') {
      parse_ocean_settings(&scene->ocean, line + 5 + 1);
    }
    else if (line[0] == 'P') {
      parse_particle_settings(&scene->particles, line + 8 + 1);
    }
    else if (line[0] == 'T') {
      parse_toy_settings(&scene->toy, line + 3 + 1);
    }
    else if (line[0] == '#' || line[0] == 10) {
      continue;
    }
    else {
      warn_message("Scene file contains unknown line!\n Content: %s", line);
    }
  }

  free(line);
}

/*
 * Write a *.lum file using the values present in instance.
 * @param file File handle.
 * @param instance RaytraceInstance whose values are written.
 */
void lum_write_file(FILE* file, RaytraceInstance* instance) {
  char* line = malloc(LINE_SIZE);

  sprintf(line, "Luminary\n");
  fputs(line, file);
  sprintf(line, "V %d\n", CURRENT_VERSION);
  fputs(line, file);

  sprintf(
    line,
    "#===============================\n# This file was automatically\n# created by Luminary.\n#\n# Please read the documentation\n# before "
    "changing any settings.\n#===============================\n");
  fputs(line, file);

  sprintf(line, "\n#===============================\n# General Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "GENERAL WIDTH___ %d\n", instance->settings.width);
  fputs(line, file);
  sprintf(line, "GENERAL HEIGHT__ %d\n", instance->settings.height);
  fputs(line, file);
  sprintf(line, "GENERAL BOUNCES_ %d\n", instance->max_ray_depth);
  fputs(line, file);
  sprintf(line, "GENERAL SAMPLES_ %d\n", instance->settings.samples);
  fputs(line, file);
  sprintf(line, "GENERAL NUMLIGHT %d\n", instance->ris_settings.num_light_rays);
  fputs(line, file);
  sprintf(line, "GENERAL DENOISER %d\n", instance->settings.denoiser);
  fputs(line, file);
  sprintf(line, "GENERAL OUTPUTFN %s\n", instance->settings.output_path);
  fputs(line, file);
  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    sprintf(line, "GENERAL MESHFILE %s\n", instance->settings.mesh_files[i]);
    fputs(line, file);
  }

  sprintf(line, "\n#===============================\n# Camera Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "CAMERA POSITION %f %f %f\n", instance->scene.camera.pos.x, instance->scene.camera.pos.y, instance->scene.camera.pos.z);
  fputs(line, file);
  sprintf(
    line, "CAMERA ROTATION %f %f %f\n", instance->scene.camera.rotation.x, instance->scene.camera.rotation.y,
    instance->scene.camera.rotation.z);
  fputs(line, file);
  sprintf(line, "CAMERA FOV_____ %f\n", instance->scene.camera.fov);
  fputs(line, file);
  sprintf(line, "CAMERA FOCALLEN %f\n", instance->scene.camera.focal_length);
  fputs(line, file);
  sprintf(line, "CAMERA APERTURE %f\n", instance->scene.camera.aperture_size);
  fputs(line, file);
  sprintf(line, "CAMERA APESHAPE %d\n", instance->scene.camera.aperture_shape);
  fputs(line, file);
  sprintf(line, "CAMERA APEBLACO %d\n", instance->scene.camera.aperture_blade_count);
  fputs(line, file);
  sprintf(line, "CAMERA EXPOSURE %f\n", instance->scene.camera.exposure);
  fputs(line, file);
  sprintf(line, "CAMERA MINEXPOS %f\n", instance->scene.camera.min_exposure);
  fputs(line, file);
  sprintf(line, "CAMERA MAXEXPOS %f\n", instance->scene.camera.max_exposure);
  fputs(line, file);
  sprintf(line, "CAMERA BLOOM___ %d\n", instance->scene.camera.bloom);
  fputs(line, file);
  sprintf(line, "CAMERA BLOOMBLE %f\n", instance->scene.camera.bloom_blend);
  fputs(line, file);
  sprintf(line, "CAMERA LENSFLAR %d\n", instance->scene.camera.lens_flare);
  fputs(line, file);
  sprintf(line, "CAMERA LENSFTHR %f\n", instance->scene.camera.lens_flare_threshold);
  fputs(line, file);
  sprintf(line, "CAMERA DITHER__ %d\n", instance->scene.camera.dithering);
  fputs(line, file);
  sprintf(line, "CAMERA FARCLIPD %f\n", instance->scene.camera.far_clip_distance);
  fputs(line, file);
  sprintf(line, "CAMERA TONEMAP_ %d\n", instance->scene.camera.tonemap);
  fputs(line, file);
  sprintf(line, "CAMERA AGXSLOPE %f\n", instance->scene.camera.agx_custom_slope);
  fputs(line, file);
  sprintf(line, "CAMERA AGXPOWER %f\n", instance->scene.camera.agx_custom_power);
  fputs(line, file);
  sprintf(line, "CAMERA AGXSATUR %f\n", instance->scene.camera.agx_custom_saturation);
  fputs(line, file);
  sprintf(line, "CAMERA AUTOEXP_ %d\n", instance->scene.camera.auto_exposure);
  fputs(line, file);
  sprintf(line, "CAMERA FILTER__ %d\n", instance->scene.camera.filter);
  fputs(line, file);
  sprintf(line, "CAMERA PURKINJE %d\n", instance->scene.camera.purkinje);
  fputs(line, file);
  sprintf(line, "CAMERA RUSSIANR %f\n", instance->scene.camera.russian_roulette_threshold);
  fputs(line, file);
  sprintf(line, "CAMERA FIREFLYC %d\n", instance->scene.camera.do_firefly_clamping);
  fputs(line, file);
  sprintf(line, "CAMERA FILMGRAI %f\n", instance->scene.camera.film_grain);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# MATERIAL Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "MATERIAL LIGHTSON %d\n", instance->scene.material.lights_active);
  fputs(line, file);
  sprintf(line, "MATERIAL OVERRIDE %d\n", instance->scene.material.override_materials);
  fputs(line, file);
  sprintf(line, "MATERIAL SMOOTHNE %f\n", instance->scene.material.default_material.r);
  fputs(line, file);
  sprintf(line, "MATERIAL METALLIC %f\n", instance->scene.material.default_material.g);
  fputs(line, file);
  sprintf(line, "MATERIAL EMISSION %f\n", instance->scene.material.default_material.b);
  fputs(line, file);
  sprintf(line, "MATERIAL ALPHACUT %f\n", instance->scene.material.alpha_cutoff);
  fputs(line, file);
  sprintf(line, "MATERIAL COLORTRA %d\n", instance->scene.material.colored_transparency);
  fputs(line, file);
  sprintf(line, "MATERIAL IORSHADO %d\n", instance->scene.material.enable_ior_shadowing);
  fputs(line, file);
  sprintf(line, "MATERIAL INVERTRO %d\n", instance->scene.material.invert_roughness);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Sky Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(
    line, "SKY OFFSET__ %f %f %f\n", instance->scene.sky.geometry_offset.x, instance->scene.sky.geometry_offset.y,
    instance->scene.sky.geometry_offset.z);
  fputs(line, file);
  sprintf(line, "SKY AZIMUTH_ %f\n", instance->scene.sky.azimuth);
  fputs(line, file);
  sprintf(line, "SKY ALTITUDE %f\n", instance->scene.sky.altitude);
  fputs(line, file);
  sprintf(line, "SKY MOONALTI %f\n", instance->scene.sky.moon_altitude);
  fputs(line, file);
  sprintf(line, "SKY MOONAZIM %f\n", instance->scene.sky.moon_azimuth);
  fputs(line, file);
  sprintf(line, "SKY MOONTEXO %f\n", instance->scene.sky.moon_tex_offset);
  fputs(line, file);
  sprintf(line, "SKY SUNSTREN %f\n", instance->scene.sky.sun_strength);
  fputs(line, file);
  sprintf(line, "SKY DENSITY_ %f\n", instance->scene.sky.base_density);
  fputs(line, file);
  sprintf(line, "SKY OZONEABS %d\n", instance->scene.sky.ozone_absorption);
  fputs(line, file);
  sprintf(line, "SKY STEPS___ %d\n", instance->scene.sky.steps);
  fputs(line, file);
  sprintf(line, "SKY RAYLEDEN %f\n", instance->scene.sky.rayleigh_density);
  fputs(line, file);
  sprintf(line, "SKY MIEDENSI %f\n", instance->scene.sky.mie_density);
  fputs(line, file);
  sprintf(line, "SKY OZONEDEN %f\n", instance->scene.sky.ozone_density);
  fputs(line, file);
  sprintf(line, "SKY RAYLEFAL %f\n", instance->scene.sky.rayleigh_falloff);
  fputs(line, file);
  sprintf(line, "SKY MIEFALLO %f\n", instance->scene.sky.mie_falloff);
  fputs(line, file);
  sprintf(line, "SKY DIAMETER %f\n", instance->scene.sky.mie_diameter);
  fputs(line, file);
  sprintf(line, "SKY GROUNDVI %f\n", instance->scene.sky.ground_visibility);
  fputs(line, file);
  sprintf(line, "SKY OZONETHI %f\n", instance->scene.sky.ozone_layer_thickness);
  fputs(line, file);
  sprintf(line, "SKY MSFACTOR %f\n", instance->scene.sky.multiscattering_factor);
  fputs(line, file);
  sprintf(line, "SKY STARSEED %d\n", instance->scene.sky.stars_seed);
  fputs(line, file);
  sprintf(line, "SKY STARINTE %f\n", instance->scene.sky.stars_intensity);
  fputs(line, file);
  sprintf(line, "SKY STARNUM_ %d\n", instance->scene.sky.settings_stars_count);
  fputs(line, file);
  sprintf(line, "SKY AERIALPE %d\n", instance->scene.sky.aerial_perspective);
  fputs(line, file);
  sprintf(line, "SKY HDRIACTI %d\n", instance->scene.sky.hdri_active);
  fputs(line, file);
  sprintf(line, "SKY HDRIDIM_ %d\n", instance->scene.sky.settings_hdri_dim);
  fputs(line, file);
  sprintf(line, "SKY HDRISAMP %d\n", instance->scene.sky.hdri_samples);
  fputs(line, file);
  sprintf(line, "SKY HDRIMIPB %f\n", instance->scene.sky.hdri_mip_bias);
  fputs(line, file);
  sprintf(
    line, "SKY HDRIORIG %f %f %f\n", instance->scene.sky.hdri_origin.x, instance->scene.sky.hdri_origin.y,
    instance->scene.sky.hdri_origin.z);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Cloud Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "CLOUD ACTIVE__ %d\n", instance->scene.sky.cloud.active);
  fputs(line, file);
  sprintf(line, "CLOUD INSCATTE %d\n", instance->scene.sky.cloud.atmosphere_scattering);
  fputs(line, file);
  sprintf(line, "CLOUD MIPMAPBI %f\n", instance->scene.sky.cloud.mipmap_bias);
  fputs(line, file);
  sprintf(line, "CLOUD SEED____ %d\n", instance->scene.sky.cloud.seed);
  fputs(line, file);
  sprintf(line, "CLOUD OFFSET__ %f %f\n", instance->scene.sky.cloud.offset_x, instance->scene.sky.cloud.offset_z);
  fputs(line, file);
  sprintf(line, "CLOUD SHASCALE %f\n", instance->scene.sky.cloud.noise_shape_scale);
  fputs(line, file);
  sprintf(line, "CLOUD DETSCALE %f\n", instance->scene.sky.cloud.noise_detail_scale);
  fputs(line, file);
  sprintf(line, "CLOUD WEASCALE %f\n", instance->scene.sky.cloud.noise_weather_scale);
  fputs(line, file);
  sprintf(line, "CLOUD DIAMETER %f\n", instance->scene.sky.cloud.droplet_diameter);
  fputs(line, file);
  sprintf(line, "CLOUD STEPS___ %d\n", instance->scene.sky.cloud.steps);
  fputs(line, file);
  sprintf(line, "CLOUD SHASTEPS %d\n", instance->scene.sky.cloud.shadow_steps);
  fputs(line, file);
  sprintf(line, "CLOUD DENSITY_ %f\n", instance->scene.sky.cloud.density);
  fputs(line, file);
  sprintf(line, "CLOUD LOWACTIV %d\n", instance->scene.sky.cloud.low.active);
  fputs(line, file);
  sprintf(line, "CLOUD LOWCOVER %f %f\n", instance->scene.sky.cloud.low.coverage_min, instance->scene.sky.cloud.low.coverage);
  fputs(line, file);
  sprintf(line, "CLOUD LOWTYPE_ %f %f\n", instance->scene.sky.cloud.low.type_min, instance->scene.sky.cloud.low.type);
  fputs(line, file);
  sprintf(line, "CLOUD LOWHEIGH %f %f\n", instance->scene.sky.cloud.low.height_min, instance->scene.sky.cloud.low.height_max);
  fputs(line, file);
  sprintf(line, "CLOUD LOWWIND_ %f %f\n", instance->scene.sky.cloud.low.wind_speed, instance->scene.sky.cloud.low.wind_angle);
  fputs(line, file);
  sprintf(line, "CLOUD MIDACTIV %d\n", instance->scene.sky.cloud.mid.active);
  fputs(line, file);
  sprintf(line, "CLOUD MIDCOVER %f %f\n", instance->scene.sky.cloud.mid.coverage_min, instance->scene.sky.cloud.mid.coverage);
  fputs(line, file);
  sprintf(line, "CLOUD MIDTYPE_ %f %f\n", instance->scene.sky.cloud.mid.type_min, instance->scene.sky.cloud.mid.type);
  fputs(line, file);
  sprintf(line, "CLOUD MIDHEIGH %f %f\n", instance->scene.sky.cloud.mid.height_min, instance->scene.sky.cloud.mid.height_max);
  fputs(line, file);
  sprintf(line, "CLOUD MIDWIND_ %f %f\n", instance->scene.sky.cloud.mid.wind_speed, instance->scene.sky.cloud.mid.wind_angle);
  fputs(line, file);
  sprintf(line, "CLOUD TOPACTIV %d\n", instance->scene.sky.cloud.top.active);
  fputs(line, file);
  sprintf(line, "CLOUD TOPCOVER %f %f\n", instance->scene.sky.cloud.top.coverage_min, instance->scene.sky.cloud.top.coverage);
  fputs(line, file);
  sprintf(line, "CLOUD TOPTYPE_ %f %f\n", instance->scene.sky.cloud.top.type_min, instance->scene.sky.cloud.top.type);
  fputs(line, file);
  sprintf(line, "CLOUD TOPHEIGH %f %f\n", instance->scene.sky.cloud.top.height_min, instance->scene.sky.cloud.top.height_max);
  fputs(line, file);
  sprintf(line, "CLOUD TOPWIND_ %f %f\n", instance->scene.sky.cloud.top.wind_speed, instance->scene.sky.cloud.top.wind_angle);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Fog Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "FOG ACTIVE__ %d\n", instance->scene.fog.active);
  fputs(line, file);
  sprintf(line, "FOG DENSITY_ %f\n", instance->scene.fog.density);
  fputs(line, file);
  sprintf(line, "FOG DIAMETER %f\n", instance->scene.fog.droplet_diameter);
  fputs(line, file);
  sprintf(line, "FOG DISTANCE %f\n", instance->scene.fog.dist);
  fputs(line, file);
  sprintf(line, "FOG HEIGHT__ %f\n", instance->scene.fog.height);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Ocean Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "OCEAN ACTIVE__ %d\n", instance->scene.ocean.active);
  fputs(line, file);
  sprintf(line, "OCEAN HEIGHT__ %f\n", instance->scene.ocean.height);
  fputs(line, file);
  sprintf(line, "OCEAN AMPLITUD %f\n", instance->scene.ocean.amplitude);
  fputs(line, file);
  sprintf(line, "OCEAN FREQUENC %f\n", instance->scene.ocean.frequency);
  fputs(line, file);
  sprintf(line, "OCEAN CHOPPY__ %f\n", instance->scene.ocean.choppyness);
  fputs(line, file);
  sprintf(line, "OCEAN REFRACT_ %f\n", instance->scene.ocean.refractive_index);
  fputs(line, file);
  sprintf(line, "OCEAN WATERTYP %d\n", instance->scene.ocean.water_type);
  fputs(line, file);
  sprintf(line, "OCEAN CAUSACTI %d\n", instance->scene.ocean.caustics_active);
  fputs(line, file);
  sprintf(line, "OCEAN CAUSRISS %d\n", instance->scene.ocean.caustics_ris_sample_count);
  fputs(line, file);
  sprintf(line, "OCEAN CAUSREGU %f\n", instance->scene.ocean.caustics_regularization);
  fputs(line, file);
  sprintf(line, "OCEAN CAUSSCAL %f\n", instance->scene.ocean.caustics_domain_scale);
  fputs(line, file);
  sprintf(line, "OCEAN MULTISCA %d\n", instance->scene.ocean.multiscattering);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Particle Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "PARTICLE ACTIVE__ %d\n", instance->scene.particles.active);
  fputs(line, file);
  sprintf(line, "PARTICLE SCALE___ %f\n", instance->scene.particles.scale);
  fputs(line, file);
  sprintf(
    line, "PARTICLE ALBEDO__ %f %f %f\n", instance->scene.particles.albedo.r, instance->scene.particles.albedo.g,
    instance->scene.particles.albedo.b);
  fputs(line, file);
  sprintf(line, "PARTICLE DIRECTIO %f %f\n", instance->scene.particles.direction_altitude, instance->scene.particles.direction_azimuth);
  fputs(line, file);
  sprintf(line, "PARTICLE SPEED___ %f\n", instance->scene.particles.speed);
  fputs(line, file);
  sprintf(line, "PARTICLE PHASEDIA %f\n", instance->scene.particles.phase_diameter);
  fputs(line, file);
  sprintf(line, "PARTICLE SEED____ %d\n", instance->scene.particles.seed);
  fputs(line, file);
  sprintf(line, "PARTICLE COUNT___ %d\n", instance->scene.particles.count);
  fputs(line, file);
  sprintf(line, "PARTICLE SIZE____ %f\n", instance->scene.particles.size);
  fputs(line, file);
  sprintf(line, "PARTICLE SIZEVARI %f\n", instance->scene.particles.size_variation);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Toy Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "TOY ACTIVE__ %d\n", instance->scene.toy.active);
  fputs(line, file);
  sprintf(line, "TOY POSITION %f %f %f\n", instance->scene.toy.position.x, instance->scene.toy.position.y, instance->scene.toy.position.z);
  fputs(line, file);
  sprintf(line, "TOY ROTATION %f %f %f\n", instance->scene.toy.rotation.x, instance->scene.toy.rotation.y, instance->scene.toy.rotation.z);
  fputs(line, file);
  sprintf(line, "TOY SHAPE___ %d\n", instance->scene.toy.shape);
  fputs(line, file);
  sprintf(line, "TOY SCALE___ %f\n", instance->scene.toy.scale);
  fputs(line, file);
  sprintf(
    line, "TOY COLOR___ %f %f %f %f\n", instance->scene.toy.albedo.r, instance->scene.toy.albedo.g, instance->scene.toy.albedo.b,
    instance->scene.toy.albedo.a);
  fputs(line, file);
  sprintf(line, "TOY MATERIAL %f %f %f\n", instance->scene.toy.material.r, instance->scene.toy.material.g, instance->scene.toy.material.b);
  fputs(line, file);
  sprintf(line, "TOY EMISSION %f %f %f\n", instance->scene.toy.emission.r, instance->scene.toy.emission.g, instance->scene.toy.emission.b);
  fputs(line, file);
  sprintf(line, "TOY EMISSIVE %d\n", instance->scene.toy.emissive);
  fputs(line, file);
  sprintf(line, "TOY REFRACT_ %f\n", instance->scene.toy.refractive_index);
  fputs(line, file);
  sprintf(line, "TOY FLASHLIG %d\n", instance->scene.toy.flashlight_mode);
  fputs(line, file);

  free(line);
}
