#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include <cuda_runtime_api.h>
#include <float.h>

const int threads_per_block = 64;
const int blocks_per_grid = 512;

__device__
const float epsilon = 0.0001f;

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
RGBF trace_light(Light light, Scene scene, const float x, const float y, const float z) {
    float ray_x = light.x - x;
    float ray_y = light.y - y;
    float ray_z = light.z - z;

    float goal_dist = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);

    ray_x /= goal_dist;
    ray_y /= goal_dist;
    ray_z /= goal_dist;

    RGBF result;
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
RGBF trace_specular(Scene scene, const float x, const float y, const float z, float ray_x, float ray_y, float ray_z, const unsigned int id) {
    float ray_length = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);

    ray_x *= ray_length;
    ray_y *= ray_length;
    ray_z *= ray_length;

    RGBF result;
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
        float red = result.r;
        float green = result.g;
        float blue = result.b;

        curr_x += normal.x * epsilon * 2;
        curr_y += normal.y * epsilon * 2;
        curr_z += normal.z * epsilon * 2;

        for (int i = 0; i < scene.lights_length; i++) {
            RGBF light_color = trace_light(scene.lights[i], scene, curr_x, curr_y, curr_z);
            red += light_color.r;
            green += light_color.g;
            blue += light_color.b;

            if (light_color.r != 0 && light_color.g != 0 && light_color.b != 0) hits++;
        }

        //float inv_hits = 1 / (float)hits;

        for (int i = 0; i < scene.lights_length; i++) {
            result.r = (red > 1.0) ? 1.0 : red;
            result.g = (green > 1.0) ? 1.0 : green;
            result.b = (blue > 1.0) ? 1.0 : blue;
        }
    }

    return result;
}



__global__
void trace_rays(RGBF* frame, Scene scene, const unsigned int width, const unsigned int height) {
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

        RGBF result = trace_specular(scene, scene.camera.x, scene.camera.y, scene.camera.z, ray_x, ray_y, ray_z, 0);

        frame[id] = result;

        id += blockDim.x * gridDim.x;
    }
}


extern "C" raytrace_instance* init_raytracing(const unsigned int width, const unsigned int height) {
    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)malloc(sizeof(RGBF) * width * height);

    cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height);

    return instance;
}

extern "C" void trace_scene(Scene scene, raytrace_instance* instance) {
    Scene scene_gpu = scene;

    cudaMalloc((void**) &(scene_gpu.spheres), sizeof(Sphere) * scene_gpu.spheres_length);
    cudaMalloc((void**) &(scene_gpu.lights), sizeof(Light) * scene_gpu.lights_length);

    cudaMemcpy(scene_gpu.spheres, scene.spheres, sizeof(Sphere) * scene.spheres_length, cudaMemcpyHostToDevice);
    cudaMemcpy(scene_gpu.lights, scene.lights, sizeof(Light) * scene.lights_length, cudaMemcpyHostToDevice);

    trace_rays<<<blocks_per_grid,threads_per_block>>>(instance->frame_buffer_gpu,scene_gpu,instance->width,instance->height);

    cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost);

    cudaFree(scene_gpu.spheres);
    cudaFree(scene_gpu.lights);
}

extern "C" void frame_buffer_to_image(Camera camera, raytrace_instance* instance, RGB8* image) {
    for (int j = 0; j < instance->height; j++) {
        for (int i = 0; i < instance->width; i++) {
            RGB8 pixel;
            RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

            pixel.r = (int)(pixel_float.r * 255);
            pixel.g = (int)(pixel_float.r * 255);
            pixel.b = (int)(pixel_float.r * 255);


            image[i + instance->width * j] = pixel;
        }
    }
}

extern "C" void free_raytracing(raytrace_instance* instance) {
    cudaFree(instance->frame_buffer_gpu);

    free(instance->frame_buffer);

    free(instance);
}
