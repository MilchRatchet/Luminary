#include "scene.h"
#include "raytrace.h"
#include <cuda_runtime_api.h>
#include <float.h>

const int threads_per_block = 256;
const int blocks_per_grid = 32;

__global__
void trace_rays(uint8_t* frame, Scene scene, const unsigned int width, const unsigned int height) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int amount = width * height;

    float step = 2 * (scene.camera.fov / width);

    float vfov = step * height / 2;

    float offset_x = (step / 2);
    float offset_y = (step / 2);

    while (id < amount) {
        int x = id % width;
        int y = (id - x) / width;

        float ray_x = -scene.camera.fov + step * x + offset_x - scene.camera.x;
        float ray_y = -vfov + step * y + offset_y - scene.camera.y;
        float ray_z = -1 - scene.camera.z;

        float length = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);

        ray_x *= length;
        ray_y *= length;
        ray_z *= length;

        float color_multiplier = 0;

        float depth = 0;

        int last_hit = 0;

        for (int i = 0; i < 100; i++) {
            float curr_x = scene.camera.x + ray_x * depth;
            float curr_y = scene.camera.y + ray_y * depth;
            float curr_z = scene.camera.z + ray_z * depth;

            float dist = FLT_MAX;

            for (int j = 0; j < scene.spheres_length; j++) {
                Sphere sphere = scene.spheres[j];

                float d = sqrtf((sphere.x-curr_x)*(sphere.x-curr_x) + (sphere.y-curr_y)*(sphere.y-curr_y) + (sphere.z-curr_z)*(sphere.z-curr_z)) - sphere.radius;

                if (d < dist) {
                    dist = d;
                    last_hit = j;
                }
            }

            if (dist < 0.00001) {
                color_multiplier = 1;
                break;
            }

            depth += dist;

            if (depth > scene.far_clip_distance) {
                break;
            }
        }

        unsigned int ptr = 3 * id;

        frame[ptr] = scene.spheres[last_hit].color.r * color_multiplier;
        frame[ptr + 1] = scene.spheres[last_hit].color.g * color_multiplier;
        frame[ptr + 2] = scene.spheres[last_hit].color.b * color_multiplier;

        id += blockDim.x * gridDim.x;
    }
}

extern "C" uint8_t* scene_to_frame(Scene scene, const unsigned int width, const unsigned int height) {
    uint8_t* frame_cpu = (uint8_t*)malloc(3 * width * height);
    uint8_t* frame_gpu;

    cudaMalloc((void**) &frame_gpu, 3 * width * height);

    Scene scene_gpu = scene;

    cudaMalloc((void**) &(scene_gpu.spheres), sizeof(Sphere) * scene_gpu.spheres_length);

    cudaMemcpy(scene_gpu.spheres, scene.spheres, sizeof(Sphere) * scene.spheres_length, cudaMemcpyHostToDevice);

    trace_rays<<<blocks_per_grid,threads_per_block>>>(frame_gpu,scene_gpu,width,height);

    cudaMemcpy(frame_cpu, frame_gpu, 3 * width * height, cudaMemcpyDeviceToHost);

    cudaFree(frame_gpu);
    cudaFree(scene_gpu.spheres);

    return frame_cpu;
}
