#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include "mesh.h"
#include "error.h"
#include "SDL/SDL.h"
#include "cuda/utils.cuh"
#include "cuda/math.cuh"
#include "cuda/sky.cuh"
#include "cuda/brdf.cuh"
#include "cuda/bvh.cuh"
#include "cuda/directives.cuh"
#include "cuda/random.cuh"
#include "cuda/kernels.cuh"
#include "cuda/denoise.cuh"
#include <cuda_runtime_api.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <immintrin.h>

//---------------------------------
// Path Tracing
//---------------------------------

static void update_sun(const Scene scene) {
    vec3 sun;
    sun.x = sinf(scene.azimuth) * cosf(scene.altitude);
    sun.y = sinf(scene.altitude);
    sun.z = cosf(scene.azimuth) * cosf(scene.altitude);
    const float scale = 1.0f / (sqrtf(sun.x * sun.x + sun.y * sun.y + sun.z * sun.z));
    sun.x *= scale;
    sun.y *= scale;
    sun.z *= scale;

    gpuErrchk(cudaMemcpyToSymbol(device_sun, &(sun), sizeof(vec3), 0, cudaMemcpyHostToDevice));

    const vec3 light_source_sun = scale_vector(sun, 149630000000.0f);

    gpuErrchk(cudaMemcpy(scene.lights, &light_source_sun, sizeof(vec3), cudaMemcpyHostToDevice));
}

static void update_camera_pos(const Scene scene, const unsigned int width, const unsigned int height) {
    const float alpha = scene.camera.rotation.x;
    const float beta = scene.camera.rotation.y;
    const float gamma = scene.camera.rotation.z;

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

    gpuErrchk(cudaMemcpyToSymbol(device_camera_rotation, &(q), sizeof(Quaternion), 0, cudaMemcpyHostToDevice));

    const float step = 2.0f * (scene.camera.fov / width);
    const float vfov = step * height / 2.0f;
    const float offset_x = (step / 2.0f);
    const float offset_y = (step / 2.0f);

    gpuErrchk(cudaMemcpyToSymbol(device_step, &(step), sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_vfov, &(vfov), sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_offset_x, &(offset_x), sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_offset_y, &(offset_y), sizeof(float), 0, cudaMemcpyHostToDevice));
}

extern "C" raytrace_instance* init_raytracing(
    const unsigned int width, const unsigned int height, const int reflection_depth,
    const int diffuse_samples, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas,
    int illuminance_atlas_length, void* material_atlas, int material_atlas_length, Scene scene, int denoiser) {

    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)_mm_malloc(sizeof(RGBF) * width * height, 32);

    const unsigned int amount = width * height;

    gpuErrchk(cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height));

    instance->reflection_depth = reflection_depth;
    instance->diffuse_samples = diffuse_samples;

    instance->albedo_atlas = albedo_atlas;
    instance->illuminance_atlas = illuminance_atlas;
    instance->material_atlas = material_atlas;

    instance->albedo_atlas_length = albedo_atlas_length;
    instance->illuminance_atlas_length = illuminance_atlas_length;
    instance->material_atlas_length = material_atlas_length;

    instance->default_material.r = 0.81f;
    instance->default_material.g = 0.0f;
    instance->default_material.b = 1.0f;

    instance->scene_gpu = scene;
    instance->shading_mode = 0;

    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.texture_assignments), sizeof(texture_assignment) * scene.materials_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.triangles), sizeof(Triangle) * instance->scene_gpu.triangles_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.traversal_triangles), sizeof(Traversal_Triangle) * instance->scene_gpu.triangles_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.nodes), sizeof(Node8) * instance->scene_gpu.nodes_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.lights), sizeof(Light) * instance->scene_gpu.lights_length));

    gpuErrchk(cudaMemcpy(instance->scene_gpu.texture_assignments, scene.texture_assignments, sizeof(texture_assignment) * scene.materials_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.triangles, scene.triangles, sizeof(Triangle) * scene.triangles_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.traversal_triangles, scene.traversal_triangles, sizeof(Traversal_Triangle) * scene.triangles_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.nodes, scene.nodes, sizeof(Node8) * scene.nodes_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.lights, scene.lights, sizeof(Light) * scene.lights_length, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(device_texture_assignments, &(instance->scene_gpu.texture_assignments), sizeof(texture_assignment*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_frame, &(instance->frame_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_width, &(instance->width), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_height, &(instance->height), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_reflection_depth, &(instance->reflection_depth), sizeof(int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_diffuse_samples, &(instance->diffuse_samples), sizeof(int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_albedo_atlas, &(instance->albedo_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_illuminance_atlas, &(instance->illuminance_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_material_atlas, &(instance->material_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_amount, &(amount), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    instance->denoiser = denoiser;

    if (instance->denoiser) {
        gpuErrchk(cudaMalloc((void**) &(instance->albedo_buffer_gpu), sizeof(RGBF) * width * height));
        gpuErrchk(cudaMemcpyToSymbol(device_albedo_buffer, &(instance->albedo_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpyToSymbol(device_denoiser, &(instance->denoiser), sizeof(int), 0, cudaMemcpyHostToDevice));
    }

    size_t samples_length;
    gpuErrchk(cudaMemGetInfo(&samples_length, 0));
    samples_length /= 2;
    samples_length /= sizeof(Sample);

    instance->samples_per_sample = (32 < (unsigned int)samples_length / (width * height)) ? 32 : (unsigned int)samples_length / (width * height);
    assert(instance->samples_per_sample, "Not enough memory to allocate samples buffer", 1);
    instance->samples_per_sample = (instance->samples_per_sample < 10) ? instance->samples_per_sample : 10;
    instance->samples_per_sample = (instance->samples_per_sample < instance->diffuse_samples) ? instance->samples_per_sample : instance->diffuse_samples;
    gpuErrchk(cudaMemcpyToSymbol(device_samples_per_sample, &(instance->samples_per_sample), sizeof(int), 0, cudaMemcpyHostToDevice));
    samples_length = width * height * instance->samples_per_sample;
    unsigned int temp = (instance->diffuse_samples + instance->samples_per_sample - 1)/instance->samples_per_sample;
    gpuErrchk(cudaMemcpyToSymbol(device_iterations_per_sample, &(temp), sizeof(int), 0, cudaMemcpyHostToDevice));

    const unsigned int actual_samples_length = (unsigned int)samples_length;

    gpuErrchk(cudaMalloc((void**) &(instance->samples_gpu), sizeof(Sample) * samples_length));
    gpuErrchk(cudaMemcpyToSymbol(device_active_samples, &(instance->samples_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &(instance->samples_finished_gpu), sizeof(Sample_Result) * samples_length));
    gpuErrchk(cudaMemcpyToSymbol(device_finished_samples, &(instance->samples_finished_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_samples_length, &(actual_samples_length), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &(instance->randoms_gpu), sizeof(curandStateXORWOW_t) * samples_length));
    gpuErrchk(cudaMemcpyToSymbol(device_sample_randoms, &(instance->randoms_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));

    return instance;
}

extern "C" void* initialize_textures(TextureRGBA* textures, const int textures_length) {
    cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t) * textures_length);
    cudaTextureObject_t* textures_gpu;

    gpuErrchk(cudaMalloc((void**) &(textures_gpu), sizeof(cudaTextureObject_t) * textures_length));

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    for (int i = 0; i < textures_length; i++) {
        TextureRGBA texture = textures[i];

        const int num_rows = texture.height;
        const int num_cols = texture.width;
        RGBAF* data = texture.data;
        RGBAF* data_gpu;
        size_t pitch;
        gpuErrchk(cudaMallocPitch((void**) &data_gpu, &pitch, num_cols * sizeof(RGBAF), num_rows));
        gpuErrchk(cudaMemcpy2D(data_gpu, pitch, data, num_cols * sizeof(RGBAF), num_cols * sizeof(RGBAF), num_rows, cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = data_gpu;
        resDesc.res.pitch2D.width = num_cols;
        resDesc.res.pitch2D.height = num_rows;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
        resDesc.res.pitch2D.pitchInBytes = pitch;

        gpuErrchk(cudaCreateTextureObject(textures_cpu + i, &resDesc, &texDesc, NULL));
    }

    gpuErrchk(cudaMemcpy(textures_gpu, textures_cpu, sizeof(cudaTextureObject_t) * textures_length, cudaMemcpyHostToDevice));

    free(textures_cpu);

    return textures_gpu;
}

extern "C" void initialize_device() {
    gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Luminary - %s\n",prop.name);
}

extern "C" void free_textures(void* texture_atlas, const int textures_length) {
    cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t) * textures_length);
    gpuErrchk(cudaMemcpy(textures_cpu, texture_atlas, sizeof(cudaTextureObject_t) * textures_length, cudaMemcpyDeviceToHost));

    for (int i = 0; i < textures_length; i++) {
        gpuErrchk(cudaDestroyTextureObject(textures_cpu[i]));
    }

    gpuErrchk(cudaFree(texture_atlas));
    free(textures_cpu);
}

__device__ __host__
static float linearRGB_to_SRGB(const float value) {
    if (value <= 0.0031308f) {
      return 12.92f * value;
    }
    else {
      return 1.055f * powf(value, 0.416666666667f) - 0.055f;
    }
  }

extern "C" void copy_framebuffer_to_cpu(raytrace_instance* instance) {
    gpuErrchk(cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost));
}

extern "C" void trace_scene(raytrace_instance* instance, const int progress, const int temporal_frames, const unsigned int update_mask) {
    const int total_iterations = instance->width * instance->height * instance->samples_per_sample;

    gpuErrchk(cudaMemcpyToSymbol(device_temporal_frames, &(temporal_frames), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_sample_offset, &(total_iterations), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    if (update_mask & 0b1)
        gpuErrchk(cudaMemcpyToSymbol(device_scene, &(instance->scene_gpu), sizeof(Scene), 0, cudaMemcpyHostToDevice));
    if (update_mask & 0b10)
        gpuErrchk(cudaMemcpyToSymbol(device_shading_mode, &(instance->shading_mode), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    if (update_mask & 0b100)
        update_sun(instance->scene_gpu);
    if (update_mask & 0b1000)
        update_camera_pos(instance->scene_gpu, instance->width, instance->height);
    if (update_mask & 0b10000)
        gpuErrchk(cudaMemcpyToSymbol(device_default_material, &(instance->default_material), sizeof(RGBF), 0, cudaMemcpyHostToDevice));

    clock_t t = clock();

    int32_t curr_progress = total_iterations;
    const float ratio = 1.0f/(total_iterations);

    generate_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

    if (instance->shading_mode) {
        while (curr_progress > 0) {
            trace_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
            special_shade_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
            gpuErrchk(cudaMemcpyFromSymbol(&curr_progress, device_sample_offset, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));

            if (progress == 1) {
                clock_t curr_time = clock();
                const int samples_done = total_iterations - curr_progress;
                const double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
                printf("\r                                                                                                          \rProgress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - Performance: %.1f Mrays/s",
                    (float)samples_done * ratio * 100, time_elapsed,
                    curr_progress * (time_elapsed/samples_done),
                    0.000001 * instance->reflection_depth * (float)(instance->diffuse_samples)/(instance->samples_per_sample) * samples_done / time_elapsed);
            }
        }
    } else {
        while (curr_progress > 0) {
            trace_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
            shade_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

            gpuErrchk(cudaMemcpyFromSymbol(&curr_progress, device_sample_offset, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));

            if (progress == 1) {
                clock_t curr_time = clock();
                const int samples_done = total_iterations - curr_progress;
                const double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
                printf("\r                                                                                                          \rProgress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - Performance: %.1f Mrays/s",
                    (float)samples_done * ratio * 100, time_elapsed,
                    curr_progress * (time_elapsed/samples_done),
                    0.000001 * instance->reflection_depth * (float)(instance->diffuse_samples)/(instance->samples_per_sample) * samples_done / time_elapsed);
            }
        }
    }



    if (progress == 1)
        printf("\r                                                                                                              \r");

    finalize_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void free_inputs(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->scene_gpu.texture_assignments));
    gpuErrchk(cudaFree(instance->scene_gpu.triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.traversal_triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.nodes));
    gpuErrchk(cudaFree(instance->scene_gpu.lights));
    gpuErrchk(cudaFree(instance->samples_gpu));
    gpuErrchk(cudaFree(instance->samples_finished_gpu));
    gpuErrchk(cudaFree(instance->randoms_gpu));
}

extern "C" void free_outputs(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->frame_buffer_gpu));
    _mm_free(instance->frame_buffer);

    if (instance->denoiser) {
        gpuErrchk(cudaFree(instance->albedo_buffer_gpu));
    }

    free(instance);
}

__device__
RGB8* device_frame_8bit;

__device__
RGBF tonemap(RGBF pixel) {
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;

  pixel.r = 1.25f * (pixel.r * (a * pixel.r + b)) / (pixel.r * (c * pixel.r + d) + e);
  pixel.g = 1.25f * (pixel.g * (a * pixel.g + b)) / (pixel.g * (c * pixel.g + d) + e);
  pixel.b = 1.25f * (pixel.b * (a * pixel.b + b)) / (pixel.b * (c * pixel.b + d) + e);

  return pixel;
}

__global__
void convert_RGBF_to_RGB8(const RGBF* source) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int amount = device_width * device_height;

  while (id < amount) {
    int x = id % device_width;
    int y = id / device_width;

    RGBF pixel = source[x + y * device_width];

    pixel = tonemap(pixel);

    pixel.r = fminf(255.9f, 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, 255.9f * linearRGB_to_SRGB(pixel.b));

    RGB8 converted_pixel;
    converted_pixel.r = (uint8_t)pixel.r;
    converted_pixel.g = (uint8_t)pixel.g;
    converted_pixel.b = (uint8_t)pixel.b;

    device_frame_8bit[x + y * device_width] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}

extern "C" void initiliaze_8bit_frame(raytrace_instance* instance) {
    gpuErrchk(cudaMalloc((void**) &(instance->buffer_8bit_gpu), sizeof(RGB8) * instance->width * instance->height));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_8bit, &(instance->buffer_8bit_gpu), sizeof(RGB8*), 0, cudaMemcpyHostToDevice));
}

extern "C" void free_8bit_frame(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->buffer_8bit_gpu));
}

extern "C" void copy_framebuffer_to_8bit(RGB8* buffer, RGBF* source, raytrace_instance* instance) {
    convert_RGBF_to_RGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(source);
    gpuErrchk(cudaMemcpy(buffer, instance->buffer_8bit_gpu, sizeof(RGB8) * instance->width * instance->height, cudaMemcpyDeviceToHost));
}
