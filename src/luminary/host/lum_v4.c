#include "internal_error.h"
#include "lum.h"

#define LINE_SIZE 4096
#define CURRENT_VERSION 4

static LuminaryResult parse_general_settings(RendererSettings* settings, ARRAYPTR char*** obj_file_path_strings, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* MESHFILE */
    case 4993446653056992589u: {
      char* obj_file_path;
      __FAILURE_HANDLE(host_malloc(&obj_file_path, LINE_SIZE));

      sscanf(value, "%s\n", obj_file_path);

      __FAILURE_HANDLE(array_push(obj_file_path_strings, &obj_file_path));
      break;
    }
    /* WIDTH___ */
    case 6872316320646711639u:
      sscanf(value, "%u\n", &settings->width);
      break;
    /* HEIGHT__ */
    case 6872304225801028936u:
      sscanf(value, "%u\n", &settings->height);
      break;
    /* BOUNCES_ */
    case 6868910012049477442u:
      sscanf(value, "%u\n", &settings->max_ray_depth);
      break;
#if 0
    /* SAMPLES_ */
    case 6868910050737209683u:
      sscanf(value, "%u\n", &general->samples);
      break;
#endif
    /* NUMLIGHT */
    case 6073182477647435086u:
      sscanf(value, "%u\n", &settings->light_num_rays);
      break;
    /* DENOISER */
    case 5928236058831373636u:
      sscanf(value, "%u\n", &bool_uint);
      settings->use_denoiser = bool_uint;
      break;
#if 0
    /* OUTPUTFN */
    case 5640288308724782415u:
      sscanf(value, "%s\n", general->output_path);
      break;
#endif
    default:
      warn_message("%8.8s (%zu) is not a valid GENERAL setting.", line, key);
      break;
  }

  return LUMINARY_SUCCESS;
}

// Not supported
#if 0
static void parse_material_settings(GlobalMaterial* material, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;

  switch (key) {
    /* LIGHTSON */
    case 5642820479573510476u:
      sscanf(value, "%u\n", &material->lights_active);
      break;
    /* OVERRIDE */
    case 4991194904949773903u:
      sscanf(value, "%u\n", &material->override_materials);
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
      sscanf(value, "%u\n", &material->colored_transparency);
      break;
    /* IORSHADO */
    case 5711762006303985481u:
      sscanf(value, "%u\n", &material->enable_ior_shadowing);
      break;
    /* INTERTRO */
    case 5715723589413916233u:
      sscanf(value, "%u\n", &material->invert_roughness);
      break;
    /* ROUGHCLA */
    case 4705209688408805202u:
      sscanf(value, "%f\n", &material->caustic_roughness_clamp);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid MATERIAL setting.", line, key);
      break;
  }
}
#endif

static LuminaryResult parse_camera_settings(Camera* camera, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

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
      sscanf(value, "%u\n", &camera->aperture_shape);
      break;
    /* APEBLACO */
    case 5711480548221079617u:
      sscanf(value, "%u\n", &camera->aperture_blade_count);
      break;
    /* AUTOEXP_ */
    case 6868086486446921025u:
      sscanf(value, "%u\n", &bool_uint);
      camera->auto_exposure = bool_uint;
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
      sscanf(value, "%u\n", &bool_uint);
      camera->bloom = bool_uint;
      break;
    /* BLOOMBLE */
    case 4993438986657549378u:
      sscanf(value, "%f\n", &camera->bloom_blend);
      break;
    /* LENSFLAR */
    case 5927102449525343564u:
      sscanf(value, "%u\n", &bool_uint);
      camera->lens_flare = bool_uint;
      break;
    /* LENSFTHR */
    case 5929081570455340364u:
      sscanf(value, "%f\n", &camera->lens_flare_threshold);
      break;
    /* DITHER__ */
    case 6872302013910370628u:
      sscanf(value, "%u\n", &bool_uint);
      camera->dithering = bool_uint;
      break;
    /* TONEMAP_ */
    case 6868061231871053652u:
      sscanf(value, "%u\n", &camera->tonemap);
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
      sscanf(value, "%u\n", &camera->filter);
      break;
    /* PURKINJE */
    case 4992889213596882256u:
      sscanf(value, "%u\n", &bool_uint);
      camera->purkinje = bool_uint;
      break;
    /* RUSSIANR */
    case 5930749542479910226u:
      sscanf(value, "%f\n", &camera->russian_roulette_threshold);
      break;
    /* FIREFLYC */
    case 4852993938162862406u:
      sscanf(value, "%u\n", &bool_uint);
      camera->do_firefly_clamping = bool_uint;
      break;
    /* FILMGRAI */
    case 5278590704447932742u:
      sscanf(value, "%f\n", &camera->film_grain);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid CAMERA setting.", line, key);
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_sky_settings(Sky* sky, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* MODE____ */
    case 6872316419179302733u:
      sscanf(value, "%u\n", &sky->mode);
      break;
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
      sscanf(value, "%u\n", &bool_uint);
      sky->ozone_absorption = bool_uint;
      break;
    /* STEPS___ */
    case 6872316367824311379u:
      sscanf(value, "%u\n", &sky->steps);
      break;
    /* STARSEED */
    case 4919414392136750163u:
      sscanf(value, "%u\n", &sky->stars_seed);
      break;
    /* STARINTE */
    case 4995703963480314963u:
      sscanf(value, "%f\n", &sky->stars_intensity);
      break;
    /* STARNUM_ */
    case 6867238801685697619u:
      sscanf(value, "%u\n", &sky->settings_stars_count);
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
      sscanf(value, "%u\n", &bool_uint);
      sky->aerial_perspective = bool_uint;
      break;
    /* HDRIDIM_ */
    case 6867225564446606408u:
      sscanf(value, "%u\n", &sky->settings_hdri_dim);
      sscanf(value, "%u\n", &sky->hdri_dim);
      break;
    /* HDRISAMP */
    case 5786352922209174600u:
      sscanf(value, "%u\n", &sky->hdri_samples);
      break;
    /* HDRIMIPB */
    case 4778399800931533896u:
      sscanf(value, "%f\n", &sky->hdri_mip_bias);
      break;
    /* HDRIORIG */
    case 5136727350478783560u:
      sscanf(value, "%f %f %f\n", &sky->hdri_origin.x, &sky->hdri_origin.y, &sky->hdri_origin.z);
      break;
    /* COLORCON */
    case 5642802878915301187u:
      sscanf(value, "%f %f %f\n", &sky->constant_color.r, &sky->constant_color.g, &sky->constant_color.b);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid SKY setting.", line, key);
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_cloud_settings(Cloud* cloud, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%u\n", &bool_uint);
      cloud->active = bool_uint;
      break;
    /* INSCATTE */
    case 4995710525939863113u:
      sscanf(value, "%u\n", &bool_uint);
      cloud->atmosphere_scattering = bool_uint;
      break;
    /* MIPMAPBI */
    case 5278869954631846221u:
      sscanf(value, "%f\n", &cloud->mipmap_bias);
      break;
    /* SEED___ */
    case 6872316419162588499u:
      sscanf(value, "%u\n", &cloud->seed);
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
      sscanf(value, "%u\n", &cloud->shadow_steps);
      break;
    /* STEPS___ */
    case 6872316367824311379u:
      sscanf(value, "%u\n", &cloud->steps);
      break;
    /* DENSITY_ */
    case 6870615380437386564u:
      sscanf(value, "%f\n", &cloud->density);
      break;
    /* LOWACTIV */
    case 6217593408397463372u:
      sscanf(value, "%u\n", &bool_uint);
      cloud->low.active = bool_uint;
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
      sscanf(value, "%u\n", &bool_uint);
      cloud->mid.active = bool_uint;
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
      sscanf(value, "%u\n", &bool_uint);
      cloud->top.active = bool_uint;
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

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_fog_settings(Fog* fog, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%u\n", &bool_uint);
      fog->active = bool_uint;
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

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_ocean_settings(Ocean* ocean, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%u\n", &bool_uint);
      ocean->active = bool_uint;
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
      sscanf(value, "%u\n", &ocean->water_type);
      break;
    /* CAUSACTI */
    case 5283922210662465859u:
      sscanf(value, "%u\n", &bool_uint);
      ocean->caustics_active = bool_uint;
      break;
    /* CAUSRISS */
    case 6004223346149245251u:
      sscanf(value, "%u\n", &ocean->caustics_ris_sample_count);
      break;
    /* CAUSSCAL */
    case 5494747045528158531u:
      sscanf(value, "%f\n", &ocean->caustics_domain_scale);
      break;
    /* MULTISCA */
    case 4702694010316936525u:
      sscanf(value, "%u\n", &bool_uint);
      ocean->multiscattering = bool_uint;
      break;
    /* LIGHTSON */
    case 5642820479573510476u:
      sscanf(value, "%u\n", &bool_uint);
      ocean->triangle_light_contribution = bool_uint;
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid OCEAN setting.", line, key);
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_particle_settings(Particles* particle, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%u\n", &bool_uint);
      particle->active = bool_uint;
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
      sscanf(value, "%u\n", &particle->seed);
      break;
    /* COUNT___ */
    case 6872316372086771523u:
      sscanf(value, "%u\n", &particle->count);
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

  return LUMINARY_SUCCESS;
}

static LuminaryResult parse_toy_settings(Toy* toy, char* line) {
  const uint64_t key = *((uint64_t*) line);
  char* value        = line + 9;
  uint32_t bool_uint = 0;

  switch (key) {
    /* ACTIVE__ */
    case 6872287793290429249u:
      sscanf(value, "%u\n", &bool_uint);
      toy->active = bool_uint;
      break;
    /* POSITION */
    case 5642809484474797904u:
      sscanf(value, "%f %f %f\n", &toy->position.x, &toy->position.y, &toy->position.z);
      break;
    /* ROTATION */
    case 5642809484340645714u:
      sscanf(value, "%f %f %f\n", &toy->rotation.x, &toy->rotation.y, &toy->rotation.z);
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
      sscanf(value, "%u\n", &bool_uint);
      toy->emissive = bool_uint;
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf(value, "%f\n", &toy->refractive_index);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid TOY setting.", line, key);
      break;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parse_file_v4(FILE* file, LumFileContent* content) {
  char* line;

  __FAILURE_HANDLE(host_malloc(&line, LINE_SIZE));

  while (1) {
    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;

    if (line[0] == 'G') {
      __FAILURE_HANDLE(parse_general_settings(&content->settings, &content->obj_file_path_strings, line + 7 + 1));
    }
    else if (line[0] == 'M') {
      // parse_material_settings(&content->material, line + 8 + 1);
    }
    else if (line[0] == 'C' && line[1] == 'A') {
      __FAILURE_HANDLE(parse_camera_settings(&content->camera, line + 6 + 1));
    }
    else if (line[0] == 'S') {
      __FAILURE_HANDLE(parse_sky_settings(&content->sky, line + 3 + 1));
    }
    else if (line[0] == 'C' && line[1] == 'L') {
      __FAILURE_HANDLE(parse_cloud_settings(&content->cloud, line + 5 + 1));
    }
    else if (line[0] == 'F') {
      __FAILURE_HANDLE(parse_fog_settings(&content->fog, line + 3 + 1));
    }
    else if (line[0] == 'O') {
      __FAILURE_HANDLE(parse_ocean_settings(&content->ocean, line + 5 + 1));
    }
    else if (line[0] == 'P') {
      __FAILURE_HANDLE(parse_particle_settings(&content->particles, line + 8 + 1));
    }
    else if (line[0] == 'T') {
      __FAILURE_HANDLE(parse_toy_settings(&content->toy, line + 3 + 1));
    }
    else if (line[0] == '#' || line[0] == 10) {
      continue;
    }
    else {
      warn_message("Scene file contains unknown line!\n Content: %s", line);
    }
  }

  __FAILURE_HANDLE(host_free(&line));

  return LUMINARY_SUCCESS;
}
