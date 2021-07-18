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


extern "C" RaytraceInstance* init_raytracing(
    const unsigned int width, const unsigned int height, const int max_ray_depth, const int samples, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas,
    int illuminance_atlas_length, void* material_atlas, int material_atlas_length, Scene scene, int denoiser) {

    RaytraceInstance* instance = (RaytraceInstance*)malloc(sizeof(RaytraceInstance));

    instance->width = width;
    instance->height = height;
    instance->frame_output = (RGBF*)_mm_malloc(sizeof(RGBF) * width * height, 32);

    const unsigned int amount = width * height;

    gpuErrchk(cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height));
    gpuErrchk(cudaMalloc((void**) &(instance->frame_output_gpu), sizeof(RGBF) * width * height));
    gpuErrchk(cudaMalloc((void**) &(instance->frame_variance_gpu), sizeof(RGBF) * width * height));
    gpuErrchk(cudaMalloc((void**) &(instance->frame_bias_cache_gpu), sizeof(RGBF) * width * height));
    gpuErrchk(cudaMalloc((void**) &(instance->records_gpu), sizeof(RGBF) * width * height));

    instance->max_ray_depth = max_ray_depth;
    instance->offline_samples = samples;

    instance->albedo_atlas = albedo_atlas;
    instance->illuminance_atlas = illuminance_atlas;
    instance->material_atlas = material_atlas;

    instance->albedo_atlas_length = albedo_atlas_length;
    instance->illuminance_atlas_length = illuminance_atlas_length;
    instance->material_atlas_length = material_atlas_length;

    instance->default_material.r = 0.5f;
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
    gpuErrchk(cudaMemcpyToSymbol(device_frame_buffer, &(instance->frame_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_output, &(instance->frame_output_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_variance, &(instance->frame_variance_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_bias_cache, &(instance->frame_bias_cache_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_records, &(instance->records_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_width, &(instance->width), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_height, &(instance->height), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_max_ray_depth, &(instance->max_ray_depth), sizeof(int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_albedo_atlas, &(instance->albedo_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_illuminance_atlas, &(instance->illuminance_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_material_atlas, &(instance->material_atlas), sizeof(cudaTextureObject_t), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_amount, &(amount), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    instance->denoiser = denoiser;

    if (instance->denoiser) {
        gpuErrchk(cudaMalloc((void**) &(instance->albedo_buffer_gpu), sizeof(RGBF) * width * height));
        gpuErrchk(cudaMemcpyToSymbol(device_albedo_buffer, &(instance->albedo_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpyToSymbol(device_denoiser, &(instance->denoiser), sizeof(int), 0, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**) &(instance->bloom_scratch_gpu), sizeof(RGBF) * width * height));
        gpuErrchk(cudaMemcpyToSymbol(device_bloom_scratch, &(instance->bloom_scratch_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice));
    }

    const int thread_count = THREADS_PER_BLOCK * BLOCKS_PER_GRID;
    const int pixels_per_thread = (amount + thread_count - 1) / thread_count;
    gpuErrchk(cudaMemcpyToSymbol(device_pixels_per_thread, &(pixels_per_thread), sizeof(int), 0, cudaMemcpyHostToDevice));

    const int max_task_count = pixels_per_thread * thread_count;

    gpuErrchk(cudaMalloc((void**) &(instance->tasks_gpu), 32 * max_task_count));
    gpuErrchk(cudaMemcpyToSymbol(device_tasks, &(instance->tasks_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &(instance->trace_results_gpu), sizeof(TraceResult) * max_task_count));
    gpuErrchk(cudaMemcpyToSymbol(device_trace_results, &(instance->trace_results_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**) &(instance->task_counts_gpu), 4 * sizeof(uint16_t) * thread_count));
    gpuErrchk(cudaMemcpyToSymbol(device_task_counts, &(instance->task_counts_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**) &(instance->randoms_gpu), sizeof(curandStateXORWOW_t) * thread_count));
    gpuErrchk(cudaMemcpyToSymbol(device_sample_randoms, &(instance->randoms_gpu), sizeof(void*), 0, cudaMemcpyHostToDevice));

    initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

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

extern "C" void copy_framebuffer_to_cpu(RaytraceInstance* instance) {
    gpuErrchk(cudaMemcpy(instance->frame_output, instance->frame_output_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost));
}

extern "C" void trace_scene(RaytraceInstance* instance, const int temporal_frames, const unsigned int update_mask) {
    const int amount = instance->width * instance->height;

    gpuErrchk(cudaMemcpyToSymbol(device_temporal_frames, &(temporal_frames), sizeof(int), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(device_pixels_left, &(amount), sizeof(int), 0, cudaMemcpyHostToDevice));

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

    int pixels_left = amount;
    const float ratio = 1.0f/(amount);

    generate_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

    while (pixels_left > 0) {
        if (pixels_left < amount)
            balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        process_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        process_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
        process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

        gpuErrchk(cudaMemcpyFromSymbol(&(pixels_left), device_pixels_left, sizeof(int), 0, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    finalize_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void apply_bloom(RaytraceInstance* instance, RGBF* image) {
    if (instance->denoiser) {
        //bloom_kernel_split<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(image);
        bloom_kernel_blur_vertical<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(image);
        bloom_kernel_blur_horizontal<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(image);
    }
}

extern "C" void free_inputs(RaytraceInstance* instance) {
    gpuErrchk(cudaFree(instance->scene_gpu.texture_assignments));
    gpuErrchk(cudaFree(instance->scene_gpu.triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.traversal_triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.nodes));
    gpuErrchk(cudaFree(instance->scene_gpu.lights));
    gpuErrchk(cudaFree(instance->tasks_gpu));
    gpuErrchk(cudaFree(instance->frame_buffer_gpu));
    gpuErrchk(cudaFree(instance->frame_variance_gpu));
    gpuErrchk(cudaFree(instance->frame_bias_cache_gpu));
    gpuErrchk(cudaFree(instance->randoms_gpu));
}

extern "C" void free_outputs(RaytraceInstance* instance) {
    gpuErrchk(cudaFree(instance->frame_output_gpu));
    _mm_free(instance->frame_output);

    if (instance->denoiser) {
        gpuErrchk(cudaFree(instance->albedo_buffer_gpu));
        gpuErrchk(cudaFree(instance->bloom_scratch_gpu));
    }

    free(instance);
}

extern "C" void initialize_8bit_frame(RaytraceInstance* instance, const int width, const int height) {
    gpuErrchk(cudaMalloc((void**) &(instance->buffer_8bit_gpu), sizeof(RGB8) * width * height));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_8bit, &(instance->buffer_8bit_gpu), sizeof(RGB8*), 0, cudaMemcpyHostToDevice));
}

extern "C" void free_8bit_frame(RaytraceInstance* instance) {
    gpuErrchk(cudaFree(instance->buffer_8bit_gpu));
}

extern "C" void copy_framebuffer_to_8bit(RGB8* buffer, const int width, const int height, RGBF* source, RaytraceInstance* instance) {
    convert_RGBF_to_RGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(width, height, source);
    gpuErrchk(cudaMemcpy(buffer, instance->buffer_8bit_gpu, sizeof(RGB8) * width * height, cudaMemcpyDeviceToHost));
}
