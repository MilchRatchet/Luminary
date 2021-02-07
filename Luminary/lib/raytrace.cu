#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include <cuda_runtime_api.h>
#include <float.h>

const int threads_per_block = 256;
const int blocks_per_grid = 32;

__device__
const float epsilon = 0.00001;

__device__
float sphere_dist(Sphere sphere, const float x, const float y, const float z) {
    return sqrtf((sphere.x-x)*(sphere.x-x) + (sphere.y-y)*(sphere.y-y) + (sphere.z-z)*(sphere.z-z)) - sphere.radius;
}

__device__
vec3 sphere_normal(Sphere sphere, const float x, const float y, const float z) {
    vec3 result;
    result.x = x - sphere.x;
    result.y = y - sphere.y;
    result.z = z - sphere.z;

    float length = rsqrtf(result.x * result.x + result.y * result.y + result.z * result.z);

    result.x *= length;
    result.y *= length;
    result.z *= length;

    return result;
}


__device__
RGB8 trace_light(Light light, Scene scene, const float x, const float y, const float z) {
    float ray_x = light.x - x;
    float ray_y = light.y - y;
    float ray_z = light.z - z;

    float goal_dist = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);

    ray_x /= goal_dist;
    ray_y /= goal_dist;
    ray_z /= goal_dist;

    RGB8 result;
    result.r = 0;
    result.g = 0;
    result.b = 0;

    float depth = 0;

    for (int i = 0; i < 1000; i++) {
        float curr_x = x + ray_x * depth;
        float curr_y = y + ray_y * depth;
        float curr_z = z + ray_z * depth;

        float dist = FLT_MAX;

        for (int j = 0; j < scene.spheres_length; j++) {
            Sphere sphere = scene.spheres[j];

            float d = sphere_dist(sphere,curr_x,curr_y,curr_z);

            if (d < dist) {
                dist = d;
            }
        }

        if (dist > goal_dist) {
            result = light.color;
            break;
        }

        if (dist < epsilon) {
            break;
        }

        goal_dist -= dist;
        depth += dist;

    }

    return result;
}

__device__
RGB8 trace_specular(Scene scene, const float x, const float y, const float z, float ray_x, float ray_y, float ray_z, const unsigned int id) {
    float ray_length = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);

    ray_x *= ray_length;
    ray_y *= ray_length;
    ray_z *= ray_length;

    RGB8 result;
    result.r = 0;
    result.g = 0;
    result.b = 0;

    float depth = 0;

    unsigned int hit_id = 0;

    float curr_x;
    float curr_y;
    float curr_z;

    float dist;

    for (int i = 0; i < 1000; i++) {
        curr_x = x + ray_x * depth;
        curr_y = y + ray_y * depth;
        curr_z = z + ray_z * depth;

        dist = FLT_MAX;

        for (int j = 0; j < scene.spheres_length; j++) {
            Sphere sphere = scene.spheres[j];

            if (id == sphere.id) continue;

            float d = sphere_dist(sphere,curr_x,curr_y,curr_z);

            if (d < dist) {
                dist = d;
                hit_id = sphere.id;
            }
        }

        if (dist < epsilon) {
            break;
        }

        depth += dist;

        if (depth > scene.far_clip_distance) {
            hit_id = 0;
            break;
        }
    }

    if (hit_id != 0) {
        vec3 normal;

        for (int i = 0; i < scene.spheres_length; i++) {
            if (hit_id == scene.spheres[i].id) {
                result = scene.spheres[i].color;
                normal = sphere_normal(scene.spheres[i], curr_x, curr_y, curr_z);
                break;
            }
        }

        unsigned int hits = 0;
        unsigned int red = result.r;
        unsigned int green = result.g;
        unsigned int blue = result.b;

        curr_x += normal.x * epsilon * 2;
        curr_y += normal.y * epsilon * 2;
        curr_z += normal.z * epsilon * 2;

        for (int i = 0; i < scene.lights_length; i++) {
            RGB8 light_color = trace_light(scene.lights[i], scene, curr_x, curr_y, curr_z);
            red += light_color.r;
            green += light_color.g;
            blue += light_color.b;

            if (light_color.r != 0 && light_color.g != 0 && light_color.b != 0) hits++;
        }

        //float inv_hits = 1 / (float)hits;

        for (int i = 0; i < scene.lights_length; i++) {
            result.r = (red > 255) ? 255 : red;
            result.g = (green > 255) ? 255 : green;
            result.b = (blue > 255) ? 255 : blue;
        }
    }

    return result;
}



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

        RGB8 result = trace_specular(scene, scene.camera.x, scene.camera.y, scene.camera.z, ray_x, ray_y, ray_z, 0);

        unsigned int ptr = 3 * id;

        frame[ptr] = result.r;
        frame[ptr + 1] = result.g;
        frame[ptr + 2] = result.b;

        id += blockDim.x * gridDim.x;
    }
}

extern "C" uint8_t* scene_to_frame(Scene scene, const unsigned int width, const unsigned int height) {
    uint8_t* frame_cpu = (uint8_t*)malloc(3 * width * height);
    uint8_t* frame_gpu;

    cudaMalloc((void**) &frame_gpu, 3 * width * height);

    Scene scene_gpu = scene;

    cudaMalloc((void**) &(scene_gpu.spheres), sizeof(Sphere) * scene_gpu.spheres_length);
    cudaMalloc((void**) &(scene_gpu.lights), sizeof(Light) * scene_gpu.lights_length);

    cudaMemcpy(scene_gpu.spheres, scene.spheres, sizeof(Sphere) * scene.spheres_length, cudaMemcpyHostToDevice);
    cudaMemcpy(scene_gpu.lights, scene.lights, sizeof(Light) * scene.lights_length, cudaMemcpyHostToDevice);

    trace_rays<<<blocks_per_grid,threads_per_block>>>(frame_gpu,scene_gpu,width,height);

    cudaMemcpy(frame_cpu, frame_gpu, 3 * width * height, cudaMemcpyDeviceToHost);

    cudaFree(frame_gpu);
    cudaFree(scene_gpu.spheres);
    cudaFree(scene_gpu.lights);

    return frame_cpu;
}
