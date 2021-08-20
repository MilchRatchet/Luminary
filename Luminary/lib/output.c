#include "output.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "SDL/SDL.h"
#include "UI/UI.h"
#include "denoiser.h"
#include "frametime.h"
#include "png.h"
#include "raytrace.h"
#include "realtime.h"
#include "utils.h"

void offline_output(Scene scene, RaytraceInstance* instance, char* output_name, clock_t time) {
  clock_t start_of_rt = clock();
  trace_scene(instance, 0);
  for (int i = 1; i < instance->offline_samples; i++) {
    trace_scene(instance, i);
    const double progress     = ((double) i) / instance->offline_samples;
    const double time_elapsed = ((double) (clock() - start_of_rt)) / CLOCKS_PER_SEC;
    const double time_left    = (time_elapsed / progress) - time_elapsed;
    printf(
      "\r                                                                                                          \rProgress: "
      "%2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - Performance: %.1f Mrays/s",
      100.0 * progress, time_elapsed, time_left,
      0.000001 * instance->max_ray_depth * instance->width * instance->height * i / time_elapsed);
  }

  printf("\r                                                                                                              \r");

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_inputs(instance);

  if (instance->denoiser) {
    denoise_with_optix(instance);
    printf("[%.3fs] Applied Optix Denoiser.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);
  }

  apply_bloom(instance, instance->frame_output_gpu);

  initialize_8bit_frame(instance, instance->width, instance->height);
  XRGB8* frame = (XRGB8*) malloc(sizeof(XRGB8) * instance->width * instance->height);
  copy_framebuffer_to_8bit(frame, instance->width, instance->height, instance->frame_output_gpu, instance);

  store_XRGB8_png(output_name, frame, instance->width, instance->height);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free(frame);

  printf("[Done] Luminary can now be closed.\n");
  system("Pause");
}

static vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
  const vec3 u = {.x = q.x, .y = q.y, .z = q.z};

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  const vec3 cross = {.x = u.y * v.z - u.z * v.y, .y = u.z * v.x - u.x * v.z, .z = u.x * v.y - u.y * v.x};

  const vec3 result = {
    .x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x,
    .y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y,
    .z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z};

  return result;
}

void realtime_output(Scene scene, RaytraceInstance* instance) {
  RealtimeInstance* realtime = init_realtime_instance(instance);

  int exit = 0;

  Frametime frametime_trace = init_frametime();
  Frametime frametime_UI    = init_frametime();
  Frametime frametime_post  = init_frametime();
  Frametime frametime_total = init_frametime();
  UI ui                     = init_UI(instance, realtime);

  instance->temporal_frames = 0;

  int make_png = 0;

  char* title = (char*) malloc(4096);

  void* optix_setup = initialize_optix_denoise_for_realtime(instance);

  while (!exit) {
    SDL_Event event;

    start_frametime(&frametime_trace);
    update_scene(instance);
    trace_scene(instance, instance->temporal_frames);
    sample_frametime(&frametime_trace);

    start_frametime(&frametime_post);
    if (instance->denoiser && instance->use_denoiser) {
      RGBF* denoised_image = denoise_with_optix_realtime(optix_setup);
      if (instance->scene_gpu.camera.bloom)
        apply_bloom(instance, denoised_image);
      copy_framebuffer_to_8bit(realtime->buffer, realtime->width, realtime->height, denoised_image, instance);
    }
    else {
      copy_framebuffer_to_8bit(realtime->buffer, realtime->width, realtime->height, instance->frame_output_gpu, instance);
    }
    sample_frametime(&frametime_post);

    instance->temporal_frames++;

    SDL_PumpEvents();

    int mwheel  = 0;
    int mmotion = 0;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        if (ui.active) {
          mmotion += event.motion.xrel;
        }
        else {
          instance->scene_gpu.camera.rotation.y += event.motion.xrel * (-0.005f);
          instance->scene_gpu.camera.rotation.x += event.motion.yrel * (-0.005f);

          if (event.motion.xrel || event.motion.yrel)
            instance->temporal_frames = 0;
        }
      }
      else if (event.type == SDL_MOUSEWHEEL) {
        mwheel += event.wheel.y;
      }
      else if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.scancode == SDL_SCANCODE_F12) {
          make_png = 1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_E) {
          toggle_UI(&ui);
        }
      }
      else if (event.type == SDL_QUIT) {
        exit = 1;
      }
    }

    start_frametime(&frametime_UI);
    set_input_events_UI(&ui, mmotion, mwheel);
    handle_mouse_UI(&ui);
    render_UI(&ui);
    blit_UI(&ui, (uint8_t*) realtime->buffer, realtime->width, realtime->height);
    sample_frametime(&frametime_UI);

    if (make_png) {
      make_png         = 0;
      char* filename   = malloc(4096);
      char* timestring = malloc(4096);
      time_t rawtime;
      struct tm timeinfo;

      time(&rawtime);
      localtime_s(&timeinfo, &rawtime);
      strftime(timestring, 4096, "%Y-%m-%d-%H-%M-%S", &timeinfo);

      sprintf(filename, "Snap-%s.png", timestring);
      store_XRGB8_png(filename, realtime->buffer, realtime->width, realtime->height);

      free(filename);
      free(timestring);
    }

    sample_frametime(&frametime_total);
    start_frametime(&frametime_total);
    const double trace_time = get_frametime(&frametime_trace);
    const double ui_time    = get_frametime(&frametime_UI);
    const double post_time  = get_frametime(&frametime_post);
    const double total_time = get_frametime(&frametime_total);

    sprintf(
      title, "Luminary - FPS: %.0f - Frametime: %.2fms Trace: %.2fms UI: %.2fms Post: %.2fms", 1000.0 / total_time, total_time, trace_time,
      ui_time, post_time);

    const double normalized_time = total_time / 16.66667;

    SDL_SetWindowTitle(realtime->window, title);
    SDL_UpdateWindowSurface(realtime->window);

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

    const uint8_t* keystate = SDL_GetKeyboardState((int*) 0);

    if (keystate[SDL_SCANCODE_LEFT]) {
      instance->scene_gpu.sky.azimuth += 0.005f * normalized_time;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
      instance->scene_gpu.sky.azimuth -= 0.005f * normalized_time;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_UP]) {
      instance->scene_gpu.sky.altitude += 0.005f * normalized_time;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
      instance->scene_gpu.sky.altitude -= 0.005f * normalized_time;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_W]) {
      movement_vector.z -= 1.0f;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_A]) {
      movement_vector.x -= 1.0f;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_S]) {
      movement_vector.z += 1.0f;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_D]) {
      movement_vector.x += 1.0f;
      instance->temporal_frames = 0;
    }
    if (keystate[SDL_SCANCODE_LSHIFT]) {
      shift_pressed = 2;
    }

    const float movement_speed = 0.5f * shift_pressed * normalized_time;

    movement_vector = rotate_vector_by_quaternion(movement_vector, q);
    instance->scene_gpu.camera.pos.x += movement_speed * movement_vector.x;
    instance->scene_gpu.camera.pos.y += movement_speed * movement_vector.y;
    instance->scene_gpu.camera.pos.z += movement_speed * movement_vector.z;

    if (instance->scene_gpu.ocean.update) {
      instance->temporal_frames = 0;
      instance->scene_gpu.ocean.time += total_time * 0.001f * instance->scene_gpu.ocean.speed;
    }

    if (instance->scene_gpu.camera.auto_exposure) {
      instance->scene_gpu.camera.exposure = get_auto_exposure_from_optix(optix_setup, instance->scene_gpu.camera.exposure);
    }
  }

  free(title);
  free_8bit_frame(instance);
  free_realtime_denoise(optix_setup);
  free_realtime_instance(realtime);
  free_UI(&ui);
}
