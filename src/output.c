#include "output.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "SDL.h"
#include "UI/UI.h"
#include "bench.h"
#include "buffer.h"
#include "denoise.h"
#include "device.h"
#include "frametime.h"
#include "log.h"
#include "png.h"
#include "qoi.h"
#include "raytrace.h"
#include "utils.h"
#include "window_instance.h"

void offline_exit_post_process_menu(RaytraceInstance* instance) {
  instance->post_process_menu = 0;
}

static void offline_post_process_menu(RaytraceInstance* instance) {
  WindowInstance* window = window_instance_init(instance);

  Frametime frametime = init_frametime();

  int exit = 0;

  UI ui = init_post_process_UI(instance, window);

  char* title = (char*) malloc(4096);

  RGBF* gpu_output   = device_buffer_get_pointer(instance->frame_post);
  XRGB8* gpu_scratch = device_buffer_get_pointer(window->gpu_buffer);

  while (!exit) {
    SDL_Event event;

    raytrace_update_device_scene(instance);

    DeviceBuffer* base_output_image = instance->frame_final;
    if (instance->denoiser) {
      base_output_image = denoise_apply(instance, device_buffer_get_pointer(base_output_image));
    }

    device_buffer_copy(base_output_image, instance->frame_post);
    device_camera_post_apply(instance, device_buffer_get_pointer(base_output_image), gpu_output);

    device_copy_framebuffer_to_8bit(
      gpu_output, gpu_scratch, window->buffer, window->width, window->height, window->ld, OUTPUT_VARIABLE_BEAUTY);

    SDL_PumpEvents();

    int mwheel  = 0;
    int mmotion = 0;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        mmotion += event.motion.xrel;
      }
      else if (event.type == SDL_MOUSEWHEEL) {
        mwheel += event.wheel.y;
      }
      else if (event.type == SDL_QUIT) {
        exit = 1;
      }
    }

    window_instance_update_pointer(window);

    set_input_events_UI(&ui, mmotion, mwheel);
    handle_mouse_UI(&ui, instance);
    render_UI(&ui);

    blit_UI(&ui, (uint8_t*) window->buffer, window->width, window->height, window->ld);

    sample_frametime(&frametime);
    start_frametime(&frametime);
    const double total_time = get_frametime(&frametime);

    sprintf(
      title, "Luminary - Post Process Menu - FPS: %.0f - Frametime: %.2fms - Mem: %.2f/%.2fGB", 1000.0 / total_time, total_time,
      device_memory_usage() * (1.0 / (1024.0 * 1024.0 * 1024.0)), device_memory_limit() * (1.0 / (1024.0 * 1024.0 * 1024.0)));

    SDL_SetWindowTitle(window->window, title);
    SDL_UpdateWindowSurface(window->window);

    if (!(instance->post_process_menu))
      exit = 1;
  }

  free(title);
  window_instance_free(window);
  free_UI(&ui);
}

void offline_output(RaytraceInstance* instance) {
  raytrace_prepare(instance);
  const clock_t start_of_rt       = clock();
  clock_t last_frame              = start_of_rt;
  double moving_average_frametime = 0.0;
  bench_tic("Raytracing");
  for (instance->temporal_frames = 0; instance->temporal_frames < instance->offline_samples; raytrace_increment(instance)) {
    raytrace_execute(instance);
    raytrace_update_ray_emitter(instance);

    const clock_t current_frame = clock();
    const double progress       = ((double) (instance->temporal_frames + 1.0f)) / instance->offline_samples;
    const double time_elapsed   = ((double) (current_frame - start_of_rt)) / CLOCKS_PER_SEC;
    const double frametime      = ((double) (current_frame - last_frame)) / CLOCKS_PER_SEC;
    moving_average_frametime    = (instance->temporal_frames == 0.0f) ? frametime : 0.9 * moving_average_frametime + 0.1 * frametime;
    const double time_left      = (instance->offline_samples - (instance->temporal_frames + 1.0f)) * moving_average_frametime;

    print_info_inline(
      "Progress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - Frametime: %.1f ms", 100.0 * progress, time_elapsed, time_left,
      1000.0f * moving_average_frametime);
    last_frame = current_frame;
  }

  bench_toc();

  raytrace_free_work_buffers(instance);

  // Create final image based on current settings
  DeviceBuffer* base_output_image = instance->frame_final;
  if (instance->denoiser) {
    denoise_create(instance);
    base_output_image = denoise_apply(instance, device_buffer_get_pointer(base_output_image));
  }

  device_buffer_copy(base_output_image, instance->frame_post);
  device_camera_post_apply(instance, device_buffer_get_pointer(base_output_image), device_buffer_get_pointer(instance->frame_post));

  DeviceBuffer* scratch_buffer = (DeviceBuffer*) 0;
  device_buffer_init(&scratch_buffer);
  device_buffer_malloc(scratch_buffer, sizeof(XRGB8), instance->output_width * instance->output_height);
  XRGB8* frame      = (XRGB8*) malloc(sizeof(XRGB8) * instance->output_width * instance->output_height);
  char* output_path = malloc(4096);

  // Post Process Menu stores a backup of the current image, after the user is done the final output image is stored.
  if (instance->post_process_menu) {
    device_copy_framebuffer_to_8bit(
      device_buffer_get_pointer(instance->frame_post), device_buffer_get_pointer(scratch_buffer), frame, instance->output_width,
      instance->output_height, instance->output_width, OUTPUT_VARIABLE_BEAUTY);

    switch (instance->image_format) {
      case IMGFORMAT_QOI:
        sprintf(output_path, "%s.qoi1", instance->settings.output_path);
        store_XRGB8_qoi(output_path, frame, instance->output_width, instance->output_height);
        break;
      case IMGFORMAT_PNG:
      default:
        sprintf(output_path, "%s.png1", instance->settings.output_path);
        png_store_XRGB8(output_path, frame, instance->output_width, instance->output_height);
        break;
    }

    offline_post_process_menu(instance);
  }

  if (instance->denoiser) {
    denoise_free(instance);
  }

  device_copy_framebuffer_to_8bit(
    device_buffer_get_pointer(instance->frame_post), device_buffer_get_pointer(scratch_buffer), frame, instance->output_width,
    instance->output_height, instance->output_width, OUTPUT_VARIABLE_BEAUTY);

  switch (instance->image_format) {
    case IMGFORMAT_QOI:
      sprintf(output_path, "%s.qoi", instance->settings.output_path);
      store_XRGB8_qoi(output_path, frame, instance->output_width, instance->output_height);
      info_message("QOI file created.");
      break;
    case IMGFORMAT_PNG:
    default:
      sprintf(output_path, "%s.png", instance->settings.output_path);
      png_store_XRGB8(output_path, frame, instance->output_width, instance->output_height);
      info_message("PNG file created.");
      break;
  }

  if (instance->aov_mode) {
    for (OutputVariable variable = OUTPUT_VARIABLE_ALBEDO_GUIDANCE; variable < OUTPUT_VARIABLE_COUNT; variable++) {
      device_copy_framebuffer_to_8bit(
        device_buffer_get_pointer(raytrace_get_accumulate_buffer(instance, variable)), device_buffer_get_pointer(scratch_buffer), frame,
        instance->output_width, instance->output_height, instance->output_width, variable);

      const char* var_name = raytrace_get_output_variable_name(variable);

      switch (instance->image_format) {
        case IMGFORMAT_QOI:
          sprintf(output_path, "%s_%s.qoi", instance->settings.output_path, var_name);
          store_XRGB8_qoi(output_path, frame, instance->output_width, instance->output_height);
          info_message("QOI file created. (%s)", var_name);
          break;
        case IMGFORMAT_PNG:
        default:
          sprintf(output_path, "%s_%s.png", instance->settings.output_path, var_name);
          png_store_XRGB8(output_path, frame, instance->output_width, instance->output_height);
          info_message("PNG file created. (%s)", var_name);
          break;
      }
    }
  }

  device_buffer_destroy(&scratch_buffer);

  free(output_path);
  free(frame);

  info_message("Offline render completed.");
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

static void make_snapshot(RaytraceInstance* instance, WindowInstance* window, RGBF* render_output_image) {
  char* filename   = malloc(4096);
  char* timestring = malloc(4096);
  time_t rawtime;
  struct tm timeinfo;

  time(&rawtime);
  timeinfo = *localtime(&rawtime);
  strftime(timestring, 4096, "%Y-%m-%d-%H-%M-%S", &timeinfo);

  XRGB8* buffer;
  int width;
  int height;

  switch (instance->snap_resolution) {
    case SNAP_RESOLUTION_WINDOW:
      buffer = window->buffer;
      width  = window->width;
      height = window->height;
      break;
    case SNAP_RESOLUTION_RENDER:
      width  = (instance->output_variable == OUTPUT_VARIABLE_BEAUTY) ? instance->output_width : instance->width;
      height = (instance->output_variable == OUTPUT_VARIABLE_BEAUTY) ? instance->output_height : instance->height;
      buffer = malloc(sizeof(XRGB8) * width * height);
      void* gpu_scratch;
      device_malloc(&gpu_scratch, sizeof(XRGB8) * width * height);
      device_copy_framebuffer_to_8bit(render_output_image, gpu_scratch, buffer, width, height, width, instance->output_variable);
      device_free(gpu_scratch, sizeof(XRGB8) * width * height);
      break;
    default:
      free(filename);
      free(timestring);
      return;
  }

  switch (instance->image_format) {
    case IMGFORMAT_PNG:
      sprintf(filename, "Snap-%s.png", timestring);
      png_store_XRGB8(filename, buffer, width, height);
      break;
    case IMGFORMAT_QOI:
      sprintf(filename, "Snap-%s.qoi", timestring);
      store_XRGB8_qoi(filename, buffer, width, height);
      break;
    default:
      warn_message("Invalid Image Format, Snapshot was not saved.");
      break;
  }

  if (instance->snap_resolution == SNAP_RESOLUTION_RENDER) {
    free(buffer);
  }

  free(filename);
  free(timestring);
}

void realtime_output(RaytraceInstance* instance) {
  WindowInstance* window = window_instance_init(instance);

  int exit = 0;

  Frametime frametime_trace = init_frametime();
  Frametime frametime_UI    = init_frametime();
  Frametime frametime_post  = init_frametime();
  Frametime frametime_total = init_frametime();
  UI ui                     = init_UI(instance, window);

  instance->temporal_frames = 0.0f;

  float mouse_x_speed = 0.0f;
  float mouse_y_speed = 0.0f;

  int make_image = 0;

  char* title = (char*) malloc(4096);
  denoise_create(instance);

  void* gpu_scratch = device_buffer_get_pointer(window->gpu_buffer);

  while (!exit) {
    SDL_Event event;

    raytrace_prepare(instance);
    start_frametime(&frametime_trace);
    raytrace_execute(instance);
    sample_frametime(&frametime_trace);

    start_frametime(&frametime_post);

    // If window is not minimized
    DeviceBuffer* output_image = (DeviceBuffer*) 0;
    if (!(SDL_GetWindowFlags(window->window) & SDL_WINDOW_MINIMIZED)) {
      DeviceBuffer* base_output_image = raytrace_get_accumulate_buffer(instance, instance->output_variable);

      if (instance->output_variable == OUTPUT_VARIABLE_BEAUTY) {
        if (instance->denoiser) {
          base_output_image = denoise_apply(instance, device_buffer_get_pointer(base_output_image));
        }

        device_buffer_copy(base_output_image, instance->frame_post);
        device_camera_post_apply(instance, device_buffer_get_pointer(base_output_image), device_buffer_get_pointer(instance->frame_post));

        output_image = instance->frame_post;
      }
      else {
        output_image = base_output_image;
      }

      device_copy_framebuffer_to_8bit(
        device_buffer_get_pointer(output_image), gpu_scratch, window->buffer, window->width, window->height, window->ld,
        instance->output_variable);
    }

    sample_frametime(&frametime_post);

    raytrace_increment(instance);

    SDL_PumpEvents();

    int mwheel  = 0;
    int mmotion = 0;

    float mouse_x_diff = 0.0f;
    float mouse_y_diff = 0.0f;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_MOUSEMOTION) {
        if (ui.active) {
          mmotion += event.motion.xrel;
        }
        else {
          mouse_y_diff += event.motion.xrel * (-0.005f) * instance->scene.camera.mouse_speed;
          mouse_x_diff += event.motion.yrel * (-0.005f) * instance->scene.camera.mouse_speed;

          if (event.motion.xrel || event.motion.yrel)
            instance->temporal_frames = 0.0f;
        }
      }
      else if (event.type == SDL_MOUSEWHEEL) {
        mwheel += event.wheel.y;
      }
      else if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.scancode == SDL_SCANCODE_F12) {
          make_image = 1;
        }
        else if (event.key.keysym.scancode == SDL_SCANCODE_E) {
          toggle_UI(&ui);
        }
      }
      else if (event.type == SDL_QUIT) {
        exit = 1;
      }
    }

    if (instance->scene.camera.smooth_movement) {
      mouse_x_speed += mouse_x_diff * instance->scene.camera.smoothing_factor;
      mouse_y_speed += mouse_y_diff * instance->scene.camera.smoothing_factor;
    }
    else {
      mouse_x_speed = mouse_x_diff;
      mouse_y_speed = mouse_y_diff;
    }

    if (mouse_x_speed != 0.0f || mouse_y_speed != 0.0f) {
      instance->scene.camera.rotation.x += mouse_x_speed;
      instance->scene.camera.rotation.y += mouse_y_speed;

      instance->temporal_frames = 0.0f;

      if (instance->scene.camera.smooth_movement) {
        mouse_x_speed -= mouse_x_speed * instance->scene.camera.smoothing_factor * instance->scene.camera.mouse_speed;
        mouse_y_speed -= mouse_y_speed * instance->scene.camera.smoothing_factor * instance->scene.camera.mouse_speed;

        if (fabsf(mouse_x_speed) < 0.0001f * instance->scene.camera.mouse_speed)
          mouse_x_speed = 0.0f;

        if (fabsf(mouse_y_speed) < 0.0001f * instance->scene.camera.mouse_speed)
          mouse_y_speed = 0.0f;
      }
      else {
        mouse_x_speed = 0.0f;
        mouse_y_speed = 0.0f;
      }
    }

    window_instance_update_pointer(window);

    start_frametime(&frametime_UI);

    // If window is not minimized
    if (!(SDL_GetWindowFlags(window->window) & SDL_WINDOW_MINIMIZED)) {
      set_input_events_UI(&ui, mmotion, mwheel);
      handle_mouse_UI(&ui, instance);
      render_UI(&ui);
      blit_UI(&ui, (uint8_t*) window->buffer, window->width, window->height, window->ld);
    }

    sample_frametime(&frametime_UI);

    if (make_image) {
      log_message("Taking snapshot.");
      make_snapshot(instance, window, device_buffer_get_pointer(output_image));
      make_image = 0;
    }

    sample_frametime(&frametime_total);
    start_frametime(&frametime_total);
    const double trace_time = get_frametime(&frametime_trace);
    const double ui_time    = get_frametime(&frametime_UI);
    const double post_time  = get_frametime(&frametime_post);
    const double total_time = get_frametime(&frametime_total);

    sprintf(
      title, "Luminary - FPS: %.0f - Frametime: %.2fms Trace: %.2fms UI: %.2fms Post: %.2fms Mem: %.2f/%.2fGB", 1000.0 / total_time,
      total_time, trace_time, ui_time, post_time, device_memory_usage() * (1.0 / (1024.0 * 1024.0 * 1024.0)),
      device_memory_limit() * (1.0 / (1024.0 * 1024.0 * 1024.0)));

    // TODO: Remove this hack once the UI rework is done.
    const double normalized_time = total_time / (16.66667 * (1 << (2 * (instance->undersampling_setting - instance->undersampling))));

    SDL_SetWindowTitle(window->window, title);
    SDL_UpdateWindowSurface(window->window);

    const float alpha = instance->scene.camera.rotation.x;
    const float beta  = instance->scene.camera.rotation.y;
    const float gamma = instance->scene.camera.rotation.z;

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
      instance->scene.sky.azimuth -= 0.005f * normalized_time;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_RIGHT]) {
      instance->scene.sky.azimuth += 0.005f * normalized_time;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_UP]) {
      instance->scene.sky.altitude += 0.005f * normalized_time;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_DOWN]) {
      instance->scene.sky.altitude -= 0.005f * normalized_time;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_W]) {
      movement_vector.z -= 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_A]) {
      movement_vector.x -= 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_S]) {
      movement_vector.z += 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_D]) {
      movement_vector.x += 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_LCTRL]) {
      movement_vector.y -= 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_SPACE]) {
      movement_vector.y += 1.0f;
      instance->temporal_frames = 0.0f;
    }
    if (keystate[SDL_SCANCODE_LSHIFT]) {
      shift_pressed = 2;
    }

    const float movement_speed = 0.5f * shift_pressed * normalized_time * instance->scene.camera.wasd_speed;

    movement_vector = rotate_vector_by_quaternion(movement_vector, q);
    instance->scene.camera.pos.x += movement_speed * movement_vector.x;
    instance->scene.camera.pos.y += movement_speed * movement_vector.y;
    instance->scene.camera.pos.z += movement_speed * movement_vector.z;

    if (instance->scene.camera.auto_exposure) {
      instance->scene.camera.exposure = denoise_auto_exposure(instance);
    }
  }

  free(title);
  denoise_free(instance);
  window_instance_free(window);
  free_UI(&ui);
}
