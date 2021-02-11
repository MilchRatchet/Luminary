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
int device_reflection_depth;

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
float cuboid_dist(Cuboid cuboid, const vec3 pos) {
    vec3 dist_vector;

    dist_vector.x = fabsf(cuboid.pos.x - pos.x) - cuboid.size.x;
    dist_vector.y = fabsf(cuboid.pos.y - pos.y) - cuboid.size.y;
    dist_vector.z = fabsf(cuboid.pos.z - pos.z) - cuboid.size.z;
    dist_vector.x = (dist_vector.x + fabsf(dist_vector.x)) * 0.5f;
    dist_vector.y = (dist_vector.y + fabsf(dist_vector.y)) * 0.5f;
    dist_vector.z = (dist_vector.z + fabsf(dist_vector.z)) * 0.5f;

    return cuboid.sign * sqrtf(dist_vector.x * dist_vector.x + dist_vector.y * dist_vector.y + dist_vector.z * dist_vector.z);
}

__device__
vec3 cuboid_normal(Cuboid cuboid, const vec3 pos) {
    vec3 normal;
    vec3 abs_norm;

    normal.x = pos.x - cuboid.pos.x;
    normal.y = pos.y - cuboid.pos.y;
    normal.z = pos.z - cuboid.pos.z;

    abs_norm.x = epsilon + fabsf(normal.x) - cuboid.size.x;
    abs_norm.y = epsilon + fabsf(normal.y) - cuboid.size.y;
    abs_norm.z = epsilon + fabsf(normal.z) - cuboid.size.z;

    abs_norm.x = (abs_norm.x + fabsf(abs_norm.x)) * 0.5f;
    abs_norm.y = (abs_norm.y + fabsf(abs_norm.y)) * 0.5f;
    abs_norm.z = (abs_norm.z + fabsf(abs_norm.z)) * 0.5f;

    normal.x = copysignf(abs_norm.x, normal.x);
    normal.y = copysignf(abs_norm.y, normal.y);
    normal.z = copysignf(abs_norm.z, normal.z);

    float length = rsqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) * cuboid.sign;

    normal.x *= length;
    normal.y *= length;
    normal.z *= length;

    return normal;
}


__device__
RGBF trace_light(Light light, Scene scene, const vec3 origin, const vec3 ray, float goal_dist) {
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

        for (int j = 0; j < scene.cuboids_length; j++) {
            Cuboid cuboid = scene.cuboids[j];

            float d = cuboid_dist(cuboid,curr);

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
RGBF trace_specular(Scene scene, const vec3 origin, vec3 ray, const unsigned int id, const unsigned int reflection_number) {
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

        for (int j = 0; j < scene.cuboids_length; j++) {
            Cuboid cuboid = scene.cuboids[j];

            if (id == cuboid.id) continue;

            float d = cuboid_dist(cuboid,curr);

            if (d < dist) {
                dist = d;
                hit_id = cuboid.id;
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
        float smoothness;

        for (int i = 0; i < scene.spheres_length; i++) {
            if (hit_id == scene.spheres[i].id) {
                result = scene.spheres[i].color;
                normal = sphere_normal(scene.spheres[i], curr);
                smoothness = scene.spheres[i].smoothness;
                break;
            }
        }
        for (int i = 0; i < scene.cuboids_length; i++) {
            if (hit_id == scene.cuboids[i].id) {
                result = scene.cuboids[i].color;
                normal = cuboid_normal(scene.cuboids[i], curr);
                smoothness = scene.cuboids[i].smoothness;
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

        vec3 specular_ray;

        float projected_length = ray.x * normal.x + ray.y * normal.y + ray.z * normal.z;

        specular_ray.x = ray.x - 2 * projected_length * normal.x;
        specular_ray.y = ray.y - 2 * projected_length * normal.y;
        specular_ray.z = ray.z - 2 * projected_length * normal.z;

        float specular_ray_length = rsqrtf(specular_ray.x * specular_ray.x + specular_ray.y * specular_ray.y + specular_ray.z * specular_ray.z);

        specular_ray.x *= specular_ray_length;
        specular_ray.y *= specular_ray_length;
        specular_ray.z *= specular_ray_length;

        for (int i = 0; i < scene.lights_length; i++) {
            vec3 light_ray;

            light_ray.x = scene.lights[i].pos.x - curr.x;
            light_ray.y = scene.lights[i].pos.y - curr.y;
            light_ray.z = scene.lights[i].pos.z - curr.z;

            float goal_dist = rsqrtf(light_ray.x * light_ray.x + light_ray.y * light_ray.y + light_ray.z * light_ray.z);

            light_ray.x *= goal_dist;
            light_ray.y *= goal_dist;
            light_ray.z *= goal_dist;

            RGBF light_color = trace_light(scene.lights[i], scene, curr, light_ray, 1.0f/goal_dist);

            float angle = 1.0f + light_ray.x * specular_ray.x + light_ray.y * specular_ray.y + light_ray.z * specular_ray.z;

            angle += powf(angle,smoothness);

            light_sum.r += light_color.r * angle;
            light_sum.g += light_color.g * angle;
            light_sum.b += light_color.b * angle;
        }

        result.r *= light_sum.r;
        result.g *= light_sum.g;
        result.b *= light_sum.b;


        if (reflection_number < device_reflection_depth) {
            RGBF specular_color = trace_specular(scene, curr, specular_ray, hit_id, reflection_number + 1);

            result.r += (fabsf(specular_color.r) + specular_color.r) * 0.5f * smoothness;
            result.g += (fabsf(specular_color.g) + specular_color.g) * 0.5f * smoothness;
            result.b += (fabsf(specular_color.b) + specular_color.b) * 0.5f * smoothness;
        }
    }

    return result;
}



__global__
void trace_rays(RGBF* frame, Scene scene, const unsigned int width, const unsigned int height, const unsigned int reflection_depth) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int amount = width * height;

    device_reflection_depth = reflection_depth;

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

        float ray_length = rsqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

        ray.x *= ray_length;
        ray.y *= ray_length;
        ray.z *= ray_length;

        RGBF result = trace_specular(scene, scene.camera.pos, ray, 0, 0);

        frame[id] = result;

        id += blockDim.x * gridDim.x;
    }
}


extern "C" raytrace_instance* init_raytracing(const unsigned int width, const unsigned int height, const unsigned int reflection_depth) {
    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)malloc(sizeof(RGBF) * width * height);

    cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height);

    instance->reflection_depth = reflection_depth;

    return instance;
}

extern "C" void trace_scene(Scene scene, raytrace_instance* instance) {
    Scene scene_gpu = scene;

    cudaMalloc((void**) &(scene_gpu.spheres), sizeof(Sphere) * scene_gpu.spheres_length);
    cudaMalloc((void**) &(scene_gpu.cuboids), sizeof(Cuboid) * scene_gpu.cuboids_length);
    cudaMalloc((void**) &(scene_gpu.lights), sizeof(Light) * scene_gpu.lights_length);

    cudaMemcpy(scene_gpu.spheres, scene.spheres, sizeof(Sphere) * scene.spheres_length, cudaMemcpyHostToDevice);
    cudaMemcpy(scene_gpu.cuboids, scene.cuboids, sizeof(Cuboid) * scene.cuboids_length, cudaMemcpyHostToDevice);
    cudaMemcpy(scene_gpu.lights, scene.lights, sizeof(Light) * scene.lights_length, cudaMemcpyHostToDevice);

    trace_rays<<<blocks_per_grid,threads_per_block>>>(instance->frame_buffer_gpu,scene_gpu,instance->width,instance->height, instance->reflection_depth);

    cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost);

    cudaFree(scene_gpu.spheres);
    cudaFree(scene_gpu.cuboids);
    cudaFree(scene_gpu.lights);
}

extern "C" void frame_buffer_to_image(Camera camera, raytrace_instance* instance, RGB8* image) {
    for (int j = 0; j < instance->height; j++) {
        for (int i = 0; i < instance->width; i++) {
            RGB8 pixel;
            RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

            pixel.r = (pixel_float.r > 1.0f) ? 255 : ((pixel_float.r < 0.0f) ? 0 : (int)(pixel_float.r * 255));
            pixel.g = (pixel_float.g > 1.0f) ? 255 : ((pixel_float.g < 0.0f) ? 0 : (int)(pixel_float.g * 255));
            pixel.b = (pixel_float.b > 1.0f) ? 255 : ((pixel_float.b < 0.0f) ? 0 : (int)(pixel_float.b * 255));


            image[i + instance->width * j] = pixel;
        }
    }
}

extern "C" void free_raytracing(raytrace_instance* instance) {
    cudaFree(instance->frame_buffer_gpu);

    free(instance->frame_buffer);

    free(instance);
}
