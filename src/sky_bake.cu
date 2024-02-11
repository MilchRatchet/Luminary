#include "cuda/kernels_sky_bake.cuh"
#include "cuda/sky.cuh"
#include "device.h"
#include "utils.h"

extern "C" void sky_bake_generate_LUTs(RaytraceInstance* instance) {
  bench_tic((const char*) "Sky LUT Computation");

  if (instance->scene.sky.lut_initialized) {
    texture_free_atlas(instance->sky_tm_luts, 2);
    texture_free_atlas(instance->sky_ms_luts, 2);
  }

  instance->scene.sky.base_density           = instance->atmo_settings.base_density;
  instance->scene.sky.ground_visibility      = instance->atmo_settings.ground_visibility;
  instance->scene.sky.mie_density            = instance->atmo_settings.mie_density;
  instance->scene.sky.mie_falloff            = instance->atmo_settings.mie_falloff;
  instance->scene.sky.mie_diameter           = instance->atmo_settings.mie_diameter;
  instance->scene.sky.ozone_absorption       = instance->atmo_settings.ozone_absorption;
  instance->scene.sky.ozone_density          = instance->atmo_settings.ozone_density;
  instance->scene.sky.ozone_layer_thickness  = instance->atmo_settings.ozone_layer_thickness;
  instance->scene.sky.rayleigh_density       = instance->atmo_settings.rayleigh_density;
  instance->scene.sky.rayleigh_falloff       = instance->atmo_settings.rayleigh_falloff;
  instance->scene.sky.multiscattering_factor = instance->atmo_settings.multiscattering_factor;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_tm_tex[2];
  texture_create(luts_tm_tex + 0, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, TexStorageGPU);
  texture_create(luts_tm_tex + 1, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, TexStorageGPU);
  luts_tm_tex[0].wrap_mode_S = TexModeClamp;
  luts_tm_tex[0].wrap_mode_T = TexModeClamp;
  luts_tm_tex[1].wrap_mode_S = TexModeClamp;
  luts_tm_tex[1].wrap_mode_T = TexModeClamp;

  device_malloc((void**) &luts_tm_tex[0].data, luts_tm_tex[0].height * luts_tm_tex[0].pitch * 4 * sizeof(float));
  device_malloc((void**) &luts_tm_tex[1].data, luts_tm_tex[1].height * luts_tm_tex[1].pitch * 4 * sizeof(float));

  sky_compute_transmittance_lut<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float4*) luts_tm_tex[0].data, (float4*) luts_tm_tex[1].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_tm_luts, luts_tm_tex, 2);

  device_free(luts_tm_tex[0].data, luts_tm_tex[0].height * luts_tm_tex[0].pitch * 4 * sizeof(float));
  device_free(luts_tm_tex[1].data, luts_tm_tex[1].height * luts_tm_tex[1].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  TextureRGBA luts_ms_tex[2];
  texture_create(luts_ms_tex + 0, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, TexStorageGPU);
  texture_create(luts_ms_tex + 1, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, TexStorageGPU);
  luts_ms_tex[0].wrap_mode_S = TexModeClamp;
  luts_ms_tex[0].wrap_mode_T = TexModeClamp;
  luts_ms_tex[1].wrap_mode_S = TexModeClamp;
  luts_ms_tex[1].wrap_mode_T = TexModeClamp;

  device_malloc((void**) &luts_ms_tex[0].data, luts_ms_tex[0].height * luts_ms_tex[0].pitch * 4 * sizeof(float));
  device_malloc((void**) &luts_ms_tex[1].data, luts_ms_tex[1].height * luts_ms_tex[1].pitch * 4 * sizeof(float));

  // We use the z component to signify its special intention
  dim3 threads_ms(SKY_MS_ITER, 1, 1);
  dim3 blocks_ms(SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1);

  sky_compute_multiscattering_lut<<<blocks_ms, threads_ms>>>((float4*) luts_ms_tex[0].data, (float4*) luts_ms_tex[1].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_ms_luts, luts_ms_tex, 2);

  device_free(luts_ms_tex[0].data, luts_ms_tex[0].height * luts_ms_tex[0].pitch * 4 * sizeof(float));
  device_free(luts_ms_tex[1].data, luts_ms_tex[1].height * luts_ms_tex[1].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  instance->scene.sky.lut_initialized = 1;

  bench_toc();
}

extern "C" void sky_bake_hdri_generate_LUT(RaytraceInstance* instance) {
  bench_tic((const char*) "Sky HDRI Computation");

  if (instance->scene.sky.hdri_initialized) {
    texture_free_atlas(instance->sky_hdri_luts, 1);
  }

  instance->scene.sky.hdri_dim = instance->scene.sky.settings_hdri_dim;

  const int dim = instance->scene.sky.hdri_dim;

  if (dim == 0) {
    error_message("Failed to allocated HDRI because resolution was 0. Turned off HDRI.");
    instance->scene.sky.hdri_active = 0;
    return;
  }

  instance->scene.sky.hdri_initialized = 1;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_hdri_tex[1];
  texture_create(luts_hdri_tex + 0, dim, dim, 1, dim, (void*) 0, TexDataFP32, TexStorageGPU);
  luts_hdri_tex[0].wrap_mode_S = TexModeWrap;
  luts_hdri_tex[0].wrap_mode_T = TexModeClamp;
  luts_hdri_tex[0].mipmap      = TexMipmapGenerate;

  device_malloc((void**) &luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));

  RayIterationType iter_type = TYPE_CAMERA;
  device_update_symbol(iteration_type, iter_type);

  sky_hdri_compute_hdri_lut<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float4*) luts_hdri_tex[0].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_hdri_luts, luts_hdri_tex, 1);

  device_free(luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  bench_toc();
}

extern "C" void sky_bake_hdri_set_pos_to_cam(RaytraceInstance* instance) {
  instance->scene.sky.hdri_origin = instance->scene.camera.pos;
}
