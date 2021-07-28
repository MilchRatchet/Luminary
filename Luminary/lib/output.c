#include "utils.h"
#include "SDL/SDL.h"
#include "processing.h"
#include "raytrace.h"
#include "denoiser.h"
#include "png.h"
#include "output.h"
#include "frametime.h"
#include "UI/UI.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

void offline_output(
  Scene scene, RaytraceInstance* instance, char* output_name, int progress, clock_t time) {
  clock_t start_of_rt = clock();
  trace_scene(instance, 0, 0xffffffff);
  for (int i = 1; i < instance->offline_samples; i++) {
    trace_scene(instance, i, 0x0);
    const double progress     = ((double) i) / instance->offline_samples;
    const double time_elapsed = ((double) (clock() - start_of_rt)) / CLOCKS_PER_SEC;
    const double time_left    = (time_elapsed / progress) - time_elapsed;
    printf(
      "\r                                                                                          "
      "                \rProgress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - "
      "Performance: %.1f Mrays/s",
      100.0 * progress, time_elapsed, time_left,
      0.000001 * instance->max_ray_depth * instance->width * instance->height * i / time_elapsed);
  }

  printf(
    "\r                                                                                            "
    "                  \r");

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_inputs(instance);

  if (instance->denoiser) {
    denoise_with_optix(instance);
    printf("[%.3fs] Applied Optix Denoiser.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);
  }

  apply_bloom(instance, instance->frame_output_gpu);

  initialize_8bit_frame(instance, instance->width, instance->height);
  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);
  copy_framebuffer_to_8bit(
    frame, instance->width, instance->height, instance->frame_output_gpu, instance);

  store_as_png(
    output_name, (uint8_t*) frame, sizeof(RGB8) * instance->width * instance->height,
    instance->width, instance->height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free(frame);
  free_outputs(instance);

  printf("[Done] Luminary can now be closed.\n");
  getchar();
}

static vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
  const vec3 u = {.x = q.x, .y = q.y, .z = q.z};

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  const vec3 cross = {
    .x = u.y * v.z - u.z * v.y, .y = u.z * v.x - u.x * v.z, .z = u.x * v.y - u.y * v.x};

  const vec3 result = {
    .x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x,
    .y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y,
    .z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z};

  return result;
}

static RealtimeInstance* init_realtime_instance(RaytraceInstance* instance) {
  RealtimeInstance* realtime = (RealtimeInstance*) malloc(sizeof(RealtimeInstance));

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Rect rect;
  SDL_Rect screen_size;

  SDL_GetDisplayUsableBounds(0, &rect);
  SDL_GetDisplayBounds(0, &screen_size);

  rect.y += screen_size.h - rect.h;
  rect.h -= screen_size.h - rect.h;

  // It seems SDL window dimensions must be a multiple of 4
  rect.h = rect.h & 0xfffffffc;

  rect.w = rect.h * ((float) instance->width) / instance->height;

  rect.w = rect.w & 0xfffffffc;

  if (rect.w > screen_size.w) {
    rect.w = screen_size.w;
    rect.h = rect.w * ((float) instance->height) / instance->width;
  }

  realtime->width  = rect.w;
  realtime->height = rect.h;

  realtime->window =
    SDL_CreateWindow("Luminary", SDL_WINDOWPOS_CENTERED, rect.y, rect.w, rect.h, SDL_WINDOW_SHOWN);

  realtime->window_surface = SDL_GetWindowSurface(realtime->window);

  Uint32 rmask, gmask, bmask, amask;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
  rmask = 0xff000000;
  gmask = 0x00ff0000;
  bmask = 0x0000ff00;
  amask = 0x000000ff;
#else
  rmask = 0x000000ff;
  gmask = 0x0000ff00;
  bmask = 0x00ff0000;
  amask = 0xff000000;
#endif

  realtime->surface = SDL_CreateRGBSurface(0, rect.w, rect.h, 24, rmask, gmask, bmask, amask);

  realtime->buffer = (RGB8*) realtime->surface->pixels;
  initialize_8bit_frame(instance, rect.w, rect.h);

  return realtime;
}

static void free_realtime_instance(RealtimeInstance* realtime) {
  free(realtime->buffer);
  SDL_DestroyWindow(realtime->window);
  SDL_Quit();
  free(realtime);
}

static char* SHADING_MODE_STRING[4] = {"", "[ALBEDO] ", "[DEPTH] ", "[NORMAL] "};

void realtime_output(Scene scene, RaytraceInstance* instance, const int filters) {
  RealtimeInstance* realtime = init_realtime_instance(instance);

  int exit = 0;

  Frametime frametime = init_frametime();
  UI ui               = init_UI();
  render_UI(&ui);

  int temporal_frames      = 0;
  int information_mode     = 0;
  unsigned int update_mask = 0xffffffff;
  int update_ocean         = 0;
  int use_bloom            = 0;
  int use_denoiser         = 1;
  int show_menu            = 1;

  int make_png  = 0;
  int png_count = 0;

  char* title = (char*) malloc(4096);

  void* optix_setup = initialize_optix_denoise_for_realtime(instance);

  SDL_SetRelativeMouseMode(SDL_TRUE);

  while (!exit) {
    SDL_Event event;

    trace_scene(instance, temporal_frames, update_mask);
    update_mask = 0;

    if (instance->denoiser && use_denoiser) {
      RGBF* denoised_image = denoise_with_optix_realtime(optix_setup);
      if (use_bloom)
        apply_bloom(instance, denoised_image);
      copy_framebuffer_to_8bit(
        realtime->buffer, realtime->width, realtime->height, denoised_image, instance);
    }
    else {
      copy_framebuffer_to_8bit(
        realtime->buffer, realtime->width, realtime->height, instance->frame_output_gpu, instance);
    }

    if (show_menu)
      blit_UI(&ui, (uint8_t*) realtime->buffer, realtime->width, realtime->height);

    if (make_png) {
      make_png       = 0;
      char* filename = malloc(4096);
      sprintf(filename, "%d.png", png_count++);
      store_as_png(
        filename, (uint8_t*) realtime->buffer, sizeof(RGB8) * realtime->width * realtime->height,
        realtime->width, realtime->height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);
      free(filename);
    }

    SDL_BlitSurface(realtime->surface, 0, realtime->window_surface, 0);

    temporal_frames++;
    sample_frametime(&frametime);
    const double time = get_frametime(&frametime);

    if (information_mode == 0) {
      sprintf(
        title, "Luminary %s- FPS: %.0f - Frametime: %.2fms",
        SHADING_MODE_STRING[instance->shading_mode], 1000.0 / time, time);
    }
    else if (information_mode == 1) {
      sprintf(
        title, "Luminary %s- Pos: (%.2f,%.2f,%.2f) Rot: (%.2f,%.2f,%.2f) FOV: %.2f",
        SHADING_MODE_STRING[instance->shading_mode], instance->scene_gpu.camera.pos.x,
        instance->scene_gpu.camera.pos.y, instance->scene_gpu.camera.pos.z,
        instance->scene_gpu.camera.rotation.x, instance->scene_gpu.camera.rotation.y,
        instance->scene_gpu.camera.rotation.z, instance->scene_gpu.camera.fov);
    }
    else if (information_mode == 2) {
      sprintf(
        title, "Luminary %s- Focal Length: %.2f Aperture Size: %.2f Exposure: %.2f %s",
        SHADING_MODE_STRING[instance->shading_mode], instance->scene_gpu.camera.focal_length,
        instance->scene_gpu.camera.aperture_size, instance->scene_gpu.camera.exposure,
        (instance->scene_gpu.camera.auto_exposure && instance->denoiser) ? "[AutoExp]" : "");
    }
    else if (information_mode == 3) {
      sprintf(
        title, "Luminary %s- Azimuth: %.3f Altitude: %.3f",
        SHADING_MODE_STRING[instance->shading_mode], instance->scene_gpu.azimuth,
        instance->scene_gpu.altitude);
    }
    else if (information_mode == 4) {
      sprintf(
        title, "Luminary %s- Ocean Height: %.3f Amplitude: %.3f Frequency: %.3f Choppyness: %.3f",
        SHADING_MODE_STRING[instance->shading_mode], instance->scene_gpu.ocean.height,
        instance->scene_gpu.ocean.amplitude, instance->scene_gpu.ocean.frequency,
        instance->scene_gpu.ocean.choppyness);
    }
    else if (information_mode == 5) {
      sprintf(
        title, "Luminary %s- Sky Density: %.3f Rayleigh: %.3f Mie: %.3f",
        SHADING_MODE_STRING[instance->shading_mode], instance->scene_gpu.sky.base_density,
        instance->scene_gpu.sky.rayleigh_falloff, instance->scene_gpu.sky.mie_falloff);
    }

    const double normalized_time = time / 16.66667;

    SDL_SetWindowTitle(realtime->window, title);
    SDL_UpdateWindowSurface(realtime->window);

    SDL_PumpEvents();
    const uint8_t* keystate = SDL_GetKeyboardState((int*) 0);

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        if (keystate[SDL_SCANCODE_F]) {
          instance->scene_gpu.camera.focal_length += 0.01f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_G]) {
          instance->scene_gpu.camera.aperture_size += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_E]) {
          instance->scene_gpu.camera.exposure +=
            max(1.0f, instance->scene_gpu.camera.exposure) * 0.005f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_L]) {
          instance->scene_gpu.ocean.height += 0.005f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_K]) {
          instance->scene_gpu.ocean.amplitude += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_J]) {
          instance->scene_gpu.ocean.frequency += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_H]) {
          instance->scene_gpu.ocean.choppyness += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_Y]) {
          instance->scene_gpu.sky.base_density += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_U]) {
          instance->scene_gpu.sky.rayleigh_falloff += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_I]) {
          instance->scene_gpu.sky.mie_falloff += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else if (keystate[SDL_SCANCODE_M]) {
          instance->default_material.g += 0.001f * event.motion.xrel;
          update_mask |= 0b10000;
        }
        else if (keystate[SDL_SCANCODE_N]) {
          instance->default_material.r += 0.001f * event.motion.xrel;
          update_mask |= 0b10000;
        }
        else if (keystate[SDL_SCANCODE_C]) {
          instance->scene_gpu.camera.alpha_cutoff += 0.001f * event.motion.xrel;
          update_mask |= 0b1;
        }
        else {
          instance->scene_gpu.camera.rotation.y += event.motion.xrel * (-0.005f);
          instance->scene_gpu.camera.rotation.x += event.motion.yrel * (-0.005f);
          update_mask |= 0b1000;
        }

        if (event.motion.xrel || event.motion.yrel)
          temporal_frames = 0;
      }
      else if (event.type == SDL_MOUSEWHEEL) {
        instance->scene_gpu.camera.fov -= event.wheel.y * 0.005f * normalized_time;
        update_mask |= 0b1001;
        temporal_frames = 0;
      }
      else if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.scancode == SDL_SCANCODE_T) {
          information_mode = (information_mode + 1) % 6;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_V) {
          instance->shading_mode = (instance->shading_mode + 1) % 4;
          update_mask |= 0b10;
          temporal_frames = 0;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_O) {
          update_ocean ^= 0b1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_B) {
          use_bloom ^= 0b1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_F12) {
          make_png = 1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_GRAVE) {
          use_denoiser ^= 1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_E) {
          show_menu ^= 1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_R) {
          instance->scene_gpu.camera.auto_exposure ^= 0b1;
        }
      }
      else if (event.type == SDL_QUIT) {
        exit = 1;
      }
    }

    const float alpha = instance->scene_gpu.camera.rotation.x;
    const float beta  = instance->scene_gpu.camera.rotation.y;
    const float gamma = instance->scene_gpu.camera.rotation.z;

    const float cy = cosf(gamma * 0.5f);
    const float sy = sinf(gamma * 0.5f);
    const float cp = cosf(beta * 0.5f);
    const float sp = sinf(beta * 0.5f);
    const float cr = cosf(alpha * 0.5f);
    const float sr = sinf(alpha * 0.5f);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    vec3 movement_vector = {.x = 0.0f, .y = 0.0f, .z = 0.0f};

    int shift_pressed = 1;

    if (keystate[SDL_SCANCODE_LEFT]) {
      instance->scene_gpu.azimuth += 0.005f * normalized_time;
      update_mask |= 0b100;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
      instance->scene_gpu.azimuth -= 0.005f * normalized_time;
      update_mask |= 0b100;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_UP]) {
      instance->scene_gpu.altitude += 0.005f * normalized_time;
      update_mask |= 0b100;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
      instance->scene_gpu.altitude -= 0.005f * normalized_time;
      update_mask |= 0b100;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_W]) {
      movement_vector.z -= 1.0f;
      update_mask |= 0b1;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_A]) {
      movement_vector.x -= 1.0f;
      update_mask |= 0b1;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_S]) {
      movement_vector.z += 1.0f;
      update_mask |= 0b1;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_D]) {
      movement_vector.x += 1.0f;
      update_mask |= 0b1;
      temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_LSHIFT]) {
      shift_pressed = 2;
    }

    clamp(instance->scene_gpu.camera.aperture_size, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.camera.focal_length, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.camera.exposure, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.camera.alpha_cutoff, 0.0f, 1.0f);
    clamp(instance->scene_gpu.ocean.choppyness, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.ocean.amplitude, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.ocean.frequency, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.ocean.choppyness, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.sky.base_density, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.sky.rayleigh_falloff, 0.0f, FLT_MAX);
    clamp(instance->scene_gpu.sky.mie_falloff, 0.0f, FLT_MAX);
    clamp(instance->default_material.r, 0.0f, 1.0f);
    clamp(instance->default_material.g, 0.0f, 1.0f);

    const float movement_speed = 0.5f * shift_pressed * normalized_time;

    movement_vector = rotate_vector_by_quaternion(movement_vector, q);
    instance->scene_gpu.camera.pos.x += movement_speed * movement_vector.x;
    instance->scene_gpu.camera.pos.y += movement_speed * movement_vector.y;
    instance->scene_gpu.camera.pos.z += movement_speed * movement_vector.z;

    if (update_ocean) {
      temporal_frames = 0;
      update_mask |= 0b1;
      instance->scene_gpu.ocean.time += time * 0.001f;
    }

    if (instance->scene_gpu.camera.auto_exposure) {
      instance->scene_gpu.camera.exposure = get_auto_exposure_from_optix(optix_setup);
      update_mask |= 0b1;
    }
  }

  free(title);
  free_8bit_frame(instance);
  free_realtime_denoise(optix_setup);
  free_realtime_instance(realtime);
  free_UI(&ui);
}
