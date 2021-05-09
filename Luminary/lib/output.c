#include "utils.h"
#include "SDL/SDL.h"
#include "processing.h"
#include "raytrace.h"
#include "denoiser.h"
#include "png.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void offline_output(
  Scene scene, raytrace_instance* instance, char* output_name, int progress, clock_t time) {
  trace_scene(scene, instance, progress);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_inputs(instance);
  copy_framebuffer_to_cpu(instance);

  switch (instance->denoiser) {
  case 0: {
    post_median_filter(instance, 0.9f);
    printf("[%.3fs] Applied Median Filter.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);
    break;
  }
  case 1: {
    denoise_with_optix(instance);
    printf("[%.3fs] Applied Optix Denoiser.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);
    break;
  }
  }

  post_bloom(instance, 3.0f);

  printf("[%.3fs] Applied Bloom.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  post_tonemapping(instance);

  printf("[%.3fs] Applied Tonemapping.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);
  frame_buffer_to_8bit_image(scene.camera, instance, frame);

  printf(
    "[%.3fs] Converted frame buffer to image.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    output_name, (uint8_t*) frame, sizeof(RGB8) * instance->width * instance->height,
    instance->width, instance->height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);
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

void realtime_output(Scene scene, raytrace_instance* instance, const int filters) {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window* window = SDL_CreateWindow(
    "Luminary", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, instance->width, instance->height,
    SDL_WINDOW_SHOWN);

  SDL_Surface* window_surface = SDL_GetWindowSurface(window);

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

  SDL_Surface* surface =
    SDL_CreateRGBSurface(0, instance->width, instance->height, 24, rmask, gmask, bmask, amask);

  RGB8* buffer = (RGB8*) surface->pixels;

  int exit = 0;

  clock_t time = clock();

  int frame_count = 0;

  char* title = (char*) malloc(4096);

  initiliaze_realtime(instance);

  SDL_SetRelativeMouseMode(SDL_TRUE);

  while (!exit) {
    SDL_Event event;

    trace_scene(scene, instance, 0);

    copy_framebuffer_to_8bit(buffer, instance);

    SDL_BlitSurface(surface, 0, window_surface, 0);

    frame_count++;
    const double frame_time = 1000.0 * ((double) (clock() - time)) / CLOCKS_PER_SEC;
    time                    = clock();

    sprintf(title, "Luminary - FPS: %.1f - FPS: %.0fms", 1000.0 / frame_time, frame_time);

    const double normalized_time = frame_time / 16.66667;

    SDL_SetWindowTitle(window, title);
    SDL_UpdateWindowSurface(window);

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        instance->scene_gpu.camera.rotation.y += event.motion.xrel * (-0.005f);
        instance->scene_gpu.camera.rotation.x += event.motion.yrel * (-0.005f);
      }
      else if (event.type == SDL_MOUSEWHEEL) {
        instance->scene_gpu.camera.fov -= event.wheel.y * 0.005f * normalized_time;
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

    SDL_PumpEvents();
    const uint8_t* keystate = SDL_GetKeyboardState((int*) 0);

    int shift_pressed = 1;

    if (keystate[SDL_SCANCODE_LEFT]) {
      instance->scene_gpu.azimuth += 0.005f * normalized_time;
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
      instance->scene_gpu.azimuth -= 0.005f * normalized_time;
    }
    if (keystate[SDL_SCANCODE_UP]) {
      instance->scene_gpu.altitude += 0.005f * normalized_time;
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
      instance->scene_gpu.altitude -= 0.005f * normalized_time;
    }
    if (keystate[SDL_SCANCODE_W]) {
      movement_vector.z -= 1.0f;
    }
    if (keystate[SDL_SCANCODE_A]) {
      movement_vector.x -= 1.0f;
    }
    if (keystate[SDL_SCANCODE_S]) {
      movement_vector.z += 1.0f;
    }
    if (keystate[SDL_SCANCODE_D]) {
      movement_vector.x += 1.0f;
    }
    if (keystate[SDL_SCANCODE_LSHIFT]) {
      shift_pressed = 2;
    }

    const float movement_speed = 0.5f * shift_pressed * normalized_time;

    movement_vector = rotate_vector_by_quaternion(movement_vector, q);
    instance->scene_gpu.camera.pos.x += movement_speed * movement_vector.x;
    instance->scene_gpu.camera.pos.y += movement_speed * movement_vector.y;
    instance->scene_gpu.camera.pos.z += movement_speed * movement_vector.z;
  }

  free(title);
  free_realtime(instance);

  SDL_DestroyWindow(window);
  SDL_Quit();
}
