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

void offline_output(Scene scene, raytrace_instance* instance, char* output_name, clock_t time) {
  trace_scene(scene, instance, 1);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

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

  while (!exit) {
    SDL_Event event;

    instance->scene_gpu.azimuth += 0.1f;

    trace_scene(scene, instance, 0);

    if (filters) {
      post_median_filter(instance, 0.9f);
      post_tonemapping(instance);
    }

    frame_buffer_to_8bit_image(scene.camera, instance, buffer);

    SDL_BlitSurface(surface, 0, window_surface, 0);

    frame_count++;
    const int FPS = (int) (frame_count / (((double) (clock() - time)) / CLOCKS_PER_SEC));
    sprintf(title, "Luminary - Frame: %d - FPS: %d", frame_count, FPS);

    SDL_SetWindowTitle(window, title);
    SDL_UpdateWindowSurface(window);

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        exit = 1;
      }
    }
  }

  free(title);

  SDL_DestroyWindow(window);
  SDL_Quit();
}
