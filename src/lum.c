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
    /* FRESNEL_ */
    case 6866939734539981382u:
      sscanf(value, "%d\n", &material->fresnel);
      break;
    /* ALPHACUT */
    case 6076837219871509569u:
      sscanf(value, "%f\n", &material->alpha_cutoff);
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
    /* AUTOEXP_ */
    case 6868086486446921025u:
      sscanf(value, "%d\n", &camera->auto_exposure);
      break;
    /* EXPOSURE */
    case 4995148753008613445u:
      sscanf(value, "%f\n", &camera->exposure);
      break;
    /* BLOOM___ */
    case 6872316342038383682u:
      sscanf(value, "%d\n", &camera->bloom);
      break;
    /* BLOOMSTR */
    case 5932458200661969986u:
      sscanf(value, "%f\n", &camera->bloom_strength);
      break;
    /* BLOOMTHR */
    case 5929081600453069890u:
      sscanf(value, "%f\n", &camera->bloom_threshold);
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
      sscanf(value, "%f\n", &camera->russian_roulette_bias);
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
    /* MOONALBE */
    case 4990635180450336589u:
      sscanf(value, "%f\n", &sky->moon_albedo);
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
    /* MIE_PH_G */
    case 5142908809513355597u:
      sscanf(value, "%f\n", &sky->mie_g);
      break;
    /* GROUNDVI */
    case 5284486315995255367u:
      sscanf(value, "%f\n", &sky->ground_visibility);
      break;
    /* OZONETHI */
    case 5280563219735206479u:
      sscanf(value, "%f\n", &sky->ozone_layer_thickness);
      break;
    /* MSFACTOR */
    case 5931051882104902477u:
      sscanf(value, "%f\n", &sky->multiscattering_factor);
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
    /* SEED___ */
    case 6872316419162588499u:
      sscanf(value, "%d\n", &cloud->seed);
      break;
    /* OFFSET__ */
    case 6872304213117257295u:
      sscanf(value, "%f %f\n", &cloud->offset_x, &cloud->offset_z);
      break;
    /* HEIGHTMA */
    case 4705509855082399048u:
      sscanf(value, "%f\n", &cloud->height_max);
      break;
    /* HEIGHTMI */
    case 5281970607385822536u:
      sscanf(value, "%f\n", &cloud->height_min);
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
    /* CURSCALE */
    case 4993437844263556419u:
      sscanf(value, "%f\n", &cloud->noise_curl_scale);
      break;
    /* COVERAGE */
    case 4992030533569892163u:
      sscanf(value, "%f\n", &cloud->coverage);
      break;
    /* COVERMIN */
    case 5641125024004198211u:
      sscanf(value, "%f\n", &cloud->coverage_min);
      break;
    /* ANVIL___ */
    case 6872316337643212353u:
      sscanf(value, "%f\n", &cloud->anvil);
      break;
    /* FWDSCATT */
    case 6076553554645243718u:
      sscanf(value, "%f\n", &cloud->forward_scattering);
      break;
    /* BWDSCATT */
    case 6076553554645243714u:
      sscanf(value, "%f\n", &cloud->backward_scattering);
      break;
    /* SCATLERP */
    case 5787764665257902931u:
      sscanf(value, "%f\n", &cloud->lobe_lerp);
      break;
    /* WETNESS_ */
    case 6868925413802132823u:
      sscanf(value, "%f\n", &cloud->wetness);
      break;
    /* POWDER__ */
    case 6872302013843459920u:
      sscanf(value, "%f\n", &cloud->powder);
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
    /* ANISOTRO */
    case 5715723576763043393u:
      sscanf(value, "%f\n", &fog->anisotropy);
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
    /* SPEED___ */
    case 6872316303215251539u:
      sscanf(value, "%f\n", &ocean->speed);
      break;
    /* ANIMATED */
    case 4919430807418392129u:
      sscanf(value, "%d\n", &ocean->update);
      break;
    /* COLOR___ */
    case 6872316363513024323u:
      sscanf(value, "%f %f %f %f\n", &ocean->albedo.r, &ocean->albedo.g, &ocean->albedo.b, &ocean->albedo.a);
      break;
    /* EMISSIVE */
    case 4996261458842570053u:
      sscanf(value, "%d\n", &ocean->emissive);
      break;
    /* REFRACT_ */
    case 6869189279479121234u:
      sscanf(value, "%f\n", &ocean->refractive_index);
      break;
    default:
      warn_message("%8.8s (%zu) is not a valid OCEAN setting.", line, key);
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
  sprintf(line, "GENERAL BOUNCES_ %d\n", instance->settings.max_ray_depth);
  fputs(line, file);
  sprintf(line, "GENERAL SAMPLES_ %d\n", instance->settings.samples);
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
  sprintf(line, "CAMERA EXPOSURE %f\n", instance->scene.camera.exposure);
  fputs(line, file);
  sprintf(line, "CAMERA BLOOM___ %d\n", instance->scene.camera.bloom);
  fputs(line, file);
  sprintf(line, "CAMERA BLOOMSTR %f\n", instance->scene.camera.bloom_strength);
  fputs(line, file);
  sprintf(line, "CAMERA BLOOMTHR %f\n", instance->scene.camera.bloom_threshold);
  fputs(line, file);
  sprintf(line, "CAMERA DITHER__ %d\n", instance->scene.camera.dithering);
  fputs(line, file);
  sprintf(line, "CAMERA FARCLIPD %f\n", instance->scene.camera.far_clip_distance);
  fputs(line, file);
  sprintf(line, "CAMERA TONEMAP_ %d\n", instance->scene.camera.tonemap);
  fputs(line, file);
  sprintf(line, "CAMERA AUTOEXP_ %d\n", instance->scene.camera.auto_exposure);
  fputs(line, file);
  sprintf(line, "CAMERA FILTER__ %d\n", instance->scene.camera.filter);
  fputs(line, file);
  sprintf(line, "CAMERA PURKINJE %d\n", instance->scene.camera.purkinje);
  fputs(line, file);
  sprintf(line, "CAMERA RUSSIANR %f\n", instance->scene.camera.russian_roulette_bias);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# MATERIAL Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "MATERIAL LIGHTSON %d\n", instance->scene.material.lights_active);
  fputs(line, file);
  sprintf(line, "MATERIAL SMOOTHNE %f\n", instance->scene.material.default_material.r);
  fputs(line, file);
  sprintf(line, "MATERIAL METALLIC %f\n", instance->scene.material.default_material.g);
  fputs(line, file);
  sprintf(line, "MATERIAL EMISSION %f\n", instance->scene.material.default_material.b);
  fputs(line, file);
  sprintf(line, "MATERIAL FRESNEL_ %d\n", instance->scene.material.fresnel);
  fputs(line, file);
  sprintf(line, "MATERIAL ALPHACUT %f\n", instance->scene.material.alpha_cutoff);
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
  sprintf(line, "SKY MOONALBE %f\n", instance->scene.sky.moon_albedo);
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
  sprintf(line, "SKY MIE_PH_G %f\n", instance->scene.sky.mie_g);
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

  sprintf(line, "\n#===============================\n# Cloud Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "CLOUD ACTIVE__ %d\n", instance->scene.sky.cloud.active);
  fputs(line, file);
  sprintf(line, "CLOUD SEED____ %d\n", instance->scene.sky.cloud.seed);
  fputs(line, file);
  sprintf(line, "CLOUD OFFSET__ %f %f\n", instance->scene.sky.cloud.offset_x, instance->scene.sky.cloud.offset_z);
  fputs(line, file);
  sprintf(line, "CLOUD HEIGHTMA %f\n", instance->scene.sky.cloud.height_max);
  fputs(line, file);
  sprintf(line, "CLOUD HEIGHTMI %f\n", instance->scene.sky.cloud.height_min);
  fputs(line, file);
  sprintf(line, "CLOUD SHASCALE %f\n", instance->scene.sky.cloud.noise_shape_scale);
  fputs(line, file);
  sprintf(line, "CLOUD DETSCALE %f\n", instance->scene.sky.cloud.noise_detail_scale);
  fputs(line, file);
  sprintf(line, "CLOUD WEASCALE %f\n", instance->scene.sky.cloud.noise_weather_scale);
  fputs(line, file);
  sprintf(line, "CLOUD CURSCALE %f\n", instance->scene.sky.cloud.noise_curl_scale);
  fputs(line, file);
  sprintf(line, "CLOUD COVERAGE %f\n", instance->scene.sky.cloud.coverage);
  fputs(line, file);
  sprintf(line, "CLOUD COVERMIN %f\n", instance->scene.sky.cloud.coverage_min);
  fputs(line, file);
  sprintf(line, "CLOUD ANVIL___ %f\n", instance->scene.sky.cloud.anvil);
  fputs(line, file);
  sprintf(line, "CLOUD FWDSCATT %f\n", instance->scene.sky.cloud.forward_scattering);
  fputs(line, file);
  sprintf(line, "CLOUD BWDSCATT %f\n", instance->scene.sky.cloud.backward_scattering);
  fputs(line, file);
  sprintf(line, "CLOUD SCATLERP %f\n", instance->scene.sky.cloud.lobe_lerp);
  fputs(line, file);
  sprintf(line, "CLOUD WETNESS_ %f\n", instance->scene.sky.cloud.wetness);
  fputs(line, file);
  sprintf(line, "CLOUD POWDER__ %f\n", instance->scene.sky.cloud.powder);
  fputs(line, file);
  sprintf(line, "CLOUD STEPS___ %d\n", instance->scene.sky.cloud.steps);
  fputs(line, file);
  sprintf(line, "CLOUD SHASTEPS %d\n", instance->scene.sky.cloud.shadow_steps);
  fputs(line, file);
  sprintf(line, "CLOUD DENSITY_ %f\n", instance->scene.sky.cloud.density);
  fputs(line, file);

  sprintf(line, "\n#===============================\n# Fog Settings\n#===============================\n\n");
  fputs(line, file);

  sprintf(line, "FOG ACTIVE__ %d\n", instance->scene.fog.active);
  fputs(line, file);
  sprintf(line, "FOG DENSITY_ %f\n", instance->scene.fog.density);
  fputs(line, file);
  sprintf(line, "FOG ANISOTRO %f\n", instance->scene.fog.anisotropy);
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
  sprintf(line, "OCEAN SPEED___ %f\n", instance->scene.ocean.speed);
  fputs(line, file);
  sprintf(line, "OCEAN ANIMATED %d\n", instance->scene.ocean.update);
  fputs(line, file);
  sprintf(
    line, "OCEAN COLOR___ %f %f %f %f\n", instance->scene.ocean.albedo.r, instance->scene.ocean.albedo.g, instance->scene.ocean.albedo.b,
    instance->scene.ocean.albedo.a);
  fputs(line, file);
  sprintf(line, "OCEAN EMISSIVE %d\n", instance->scene.ocean.emissive);
  fputs(line, file);
  sprintf(line, "OCEAN REFRACT_ %f\n", instance->scene.ocean.refractive_index);
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
