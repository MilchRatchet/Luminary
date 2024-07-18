#include "window_instance.h"

#include <cuda_gl_interop.h>
#include <stdlib.h>

#include "buffer.h"
#include "raytrace.h"

static PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers           = 0;
static PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer           = 0;
static PFNGLBLITFRAMEBUFFERPROC glBlitFramebuffer           = 0;
static PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = 0;
static PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers     = 0;

// TODO: If window is not maximum height then position the window in the middle rather then the top.
WindowInstance* window_instance_init(RaytraceInstance* instance) {
  WindowInstance* window = (WindowInstance*) calloc(1, sizeof(WindowInstance));

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Rect rect;
  SDL_Rect screen_size;

  SDL_GetDisplayUsableBounds(0, &rect);
  SDL_GetDisplayBounds(0, &screen_size);

  // max_width also acts as leading dimension so it must be slightly larger
  window->max_width  = screen_size.w + 32;
  window->max_height = screen_size.h;

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

  window->width  = (unsigned int) rect.w;
  window->height = (unsigned int) rect.h;

  // Request newest OpenGL version. Luminary requires very modern Nvidia GPUs already so this should always be supported.
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);
  SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);

  window->window = SDL_CreateWindow(
    "Luminary", SDL_WINDOWPOS_CENTERED, rect.y, rect.w, rect.h, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);

  if (!window->window) {
    log_message("SDL Error: %s", SDL_GetError());
    crash_message("Failed to create SDL window.");
  }

  SDL_GLContext sdl_gl_context = SDL_GL_CreateContext(window->window);

  if (!sdl_gl_context) {
    log_message("SDL Error: %s", SDL_GetError());
    crash_message("Failed to create OpenGL context.");
  }

  // No VSync, this is not a game.
  SDL_GL_SetSwapInterval(0);

  glGenFramebuffers      = SDL_GL_GetProcAddress("glGenFramebuffers");
  glBindFramebuffer      = SDL_GL_GetProcAddress("glBindFramebuffer");
  glBlitFramebuffer      = SDL_GL_GetProcAddress("glBlitFramebuffer");
  glFramebufferTexture2D = SDL_GL_GetProcAddress("glFramebufferTexture2D");
  glDeleteFramebuffers   = SDL_GL_GetProcAddress("glDeleteFramebuffers");

  if (!glGenFramebuffers) {
    crash_message("Failed to gather glGenFramebuffers");
  }
  if (!glBindFramebuffer) {
    crash_message("Failed to gather glBindFramebuffer");
  }
  if (!glBlitFramebuffer) {
    crash_message("Failed to gather glBlitFramebuffer");
  }
  if (!glFramebufferTexture2D) {
    crash_message("Failed to gather glFramebufferTexture2D");
  }
  if (!glDeleteFramebuffers) {
    crash_message("Failed to gather glDeleteFramebuffers");
  }

  glGenFramebuffers(1, &window->framebuffer);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, window->framebuffer);

  glGenTextures(1, &window->tex);
  glBindTexture(GL_TEXTURE_2D, window->tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rect.w, rect.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    error_message("OpenGL error: %u", err);
  }

  gpuErrchk(cudaGraphicsGLRegisterImage(&window->resource, window->tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

  device_buffer_init(&window->buffer);
  window_instance_resize_buffer(window);

  return window;
}

void window_instance_resize_buffer(WindowInstance* window) {
  // TODO: Probably need to redo the render buffer
  SDL_GetWindowSize(window->window, (int*) &(window->width), (int*) &(window->height));

  // Round up to nearest multiple of 128
  window->ld = ((window->width + 127) / 128) * 128;

  size_t required_buffer_size = window->ld * window->height * sizeof(XRGB8);

  if (device_buffer_get_size(window->buffer) < required_buffer_size) {
    device_buffer_free(window->buffer);
    device_buffer_malloc(window->buffer, sizeof(XRGB8), window->ld * window->height);
  }
}

int window_instance_is_visible(const WindowInstance* window) {
  if (!window)
    return 0;

  return !(SDL_GetWindowFlags(window->window) & SDL_WINDOW_MINIMIZED);
}

void window_instance_update(WindowInstance* window) {
  gpuErrchk(cudaGraphicsMapResources(1, &window->resource, cudaStreamDefault));

  cudaMipmappedArray_t mipmapped_array;
  gpuErrchk(cudaGraphicsResourceGetMappedMipmappedArray(&mipmapped_array, window->resource));

  cudaArray_t array;
  gpuErrchk(cudaGetMipmappedArrayLevel(&array, mipmapped_array, 0));

  gpuErrchk(cudaMemcpy2DToArray(
    array, 0, 0, device_buffer_get_pointer(window->buffer), window->ld, window->width, window->height, cudaMemcpyDeviceToDevice));

  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaGraphicsUnmapResources(1, &window->resource, cudaStreamDefault));

  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, window->tex, 0);
  glBlitFramebuffer(0, 0, window->width, window->height, 0, 0, window->width, window->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  SDL_GL_SwapWindow(window->window);

  GLenum err;
  while ((err = glGetError()) != GL_NO_ERROR) {
    error_message("OpenGL error: %u", err);
  }
}

void window_instance_free(WindowInstance* window) {
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &window->framebuffer);
  glDeleteTextures(1, &window->tex);
  cudaGraphicsUnregisterResource(window->resource);
  SDL_DestroyWindow(window->window);
  SDL_Quit();
  free(window);
}
