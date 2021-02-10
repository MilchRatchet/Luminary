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
float sphere_dist(Sphere sphere, const vec3 pos) {
    return sphere.sign * (sqrtf((sphere.pos.x-pos.x)*(sphere.pos.x-pos.x) + (sphere.pos.y-pos.y)*(sphere.pos.y-pos.y) + (sphere.pos.z-pos.z)*(sphere.pos.z-pos.z)) - sphere.radius);
}

__device__
vec3 sphere_normal(Sphere sphere, const vec3 pos) {
    vec3 result;
    result.x = pos.x - sphere.pos.x;
    result.y = pos.y - sphere.pos.y;
    result.z = pos.z - sphere.pos.z;

    float length = rsqrtf(result.x * result.x + result.y * result.y + result.z * result.z) * sphere.sign;

    result.x *= length;
    result.y *= length;
    result.z *= length;

    return result;
}


__device__
RGBF trace_light(Light light, Scene scene, const vec3 origin) {
    vec3 ray;

    ray.x = light.pos.x - origin.x;
    ray.y = light.pos.y - origin.y;
    ray.z = light.pos.z - origin.z;

    float goal_dist = rsqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

    ray.x *= goal_dist;
    ray.y *= goal_dist;
    ray.z *= goal_dist;

    goal_dist = 1.0f/goal_dist;

    RGBF result;
    result.r = 0.0f;
    result.g = 0.0f;
    result.b = 0.0f;

    float depth = 0;

    for (int i = 0; i < 1000; i++) {
        vec3 curr;

        curr.x = origin.x + ray.x * depth;
        curr.y = origin.y + ray.y * depth;
        curr.z = origin.z + ray.z * depth;

        float dist = FLT_MAX;

        for (int j = 0; j < scene.spheres_length; j++) {
            Sphere sphere = scene.spheres[j];

            float d = sphere_dist(sphere,curr);

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
RGBF trace_specular(Scene scene, const vec3 origin, vec3 ray, const unsigned int id) {
    float ray_length = rsqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

    ray.x *= ray_length;
    ray.y *= ray_length;
    ray.z *= ray_length;

    RGBF result;
    result.r = 0;
    result.g = 0;
    result.b = 0;

    float depth = 0;

    unsigned int hit_id = 0;

    vec3 curr;

    float dist;

    for (int i = 0; i < 1000; i++) {
        curr.x = origin.x + ray.x * depth;
        curr.y = origin.y + ray.y * depth;
        curr.z = origin.z + ray.z * depth;

        dist = FLT_MAX;

        for (int j = 0; j < scene.spheres_length; j++) {
            Sphere sphere = scene.spheres[j];

            if (id == sphere.id) continue;

            float d = sphere_dist(sphere,curr);

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
                normal = sphere_normal(scene.spheres[i], curr);
                break;
            }
        }

        RGBF light_sum;

        light_sum.r = 0.0f;
        light_sum.g = 0.0f;
        light_sum.b = 0.0f;

        curr.x += normal.x * epsilon * 2;
        curr.y += normal.y * epsilon * 2;
        curr.z += normal.z * epsilon * 2;

        for (int i = 0; i < scene.lights_length; i++) {
            RGBF light_color = trace_light(scene.lights[i], scene, curr);
            light_sum.r += light_color.r;
            light_sum.g += light_color.g;
            light_sum.b += light_color.b;
        }

        result.r *= light_sum.r;
        result.g *= light_sum.g;
        result.b *= light_sum.b;
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

        vec3 ray;

        ray.x = -scene.camera.fov + step * x + offset_x - scene.camera.pos.x;
        ray.y = -vfov + step * y + offset_y - scene.camera.pos.y;
        ray.z = -1 - scene.camera.pos.z;

        RGBF result = trace_specular(scene, scene.camera.pos, ray, 0);

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
            pixel.g = (int)(pixel_float.g * 255);
            pixel.b = (int)(pixel_float.b * 255);


            image[i + instance->width * j] = pixel;
        }
    }
}

extern "C" void free_raytracing(raytrace_instance* instance) {
    cudaFree(instance->frame_buffer_gpu);

    free(instance->frame_buffer);

    free(instance);
}
