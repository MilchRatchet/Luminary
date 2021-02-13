#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

const int threads_per_block = 256;
const int blocks_per_grid = 960;

__device__
const float epsilon = 0.0001f;

__device__
int device_reflection_depth;

__device__
Scene device_scene;

__device__
int device_diffuse_samples;

__device__
RGBF* device_frame;

__device__
unsigned int device_width;

__device__
unsigned int device_height;

__device__
unsigned int device_amount;

__device__
float device_step;

__device__
float device_vfov;

__device__
float device_offset_x;

__device__
float device_offset_y;

__device__
Quaternion device_camera_rotation;



__device__
float sphere_dist(Sphere sphere, const vec3 pos) {
    return sphere.sign * (__fsqrt_rn((sphere.pos.x-pos.x)*(sphere.pos.x-pos.x) + (sphere.pos.y-pos.y)*(sphere.pos.y-pos.y) + (sphere.pos.z-pos.z)*(sphere.pos.z-pos.z)) - sphere.radius);
}

__device__
vec3 sphere_normal(Sphere sphere, const vec3 pos) {
    vec3 result;
    result.x = pos.x - sphere.pos.x;
    result.y = pos.y - sphere.pos.y;
    result.z = pos.z - sphere.pos.z;

    float length = __frsqrt_rn(result.x * result.x + result.y * result.y + result.z * result.z) * sphere.sign;

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

    return cuboid.sign * __fsqrt_rn(dist_vector.x * dist_vector.x + dist_vector.y * dist_vector.y + dist_vector.z * dist_vector.z);
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

    float length = __frsqrt_rn(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) * cuboid.sign;

    normal.x *= length;
    normal.y *= length;
    normal.z *= length;

    return normal;
}

__device__
RGBF trace_ray_iterative(vec3 origin, vec3 ray, curandStateXORWOW_t* random) {
    RGBF result;
    result.r = 0.0f;
    result.g = 0.0f;
    result.b = 0.0f;

    float weight = 1.0f;
    RGBF record;
    record.r = 1.0f;
    record.g = 1.0f;
    record.b = 1.0f;

    RGBF sky;
    sky.r = 10.0f;
    sky.g = 10.0f;
    sky.b = 14.0f;

    for (int reflection_number = 0; reflection_number < device_reflection_depth; reflection_number++) {
        float depth = 0.0f;

        int hit_id = 0;

        vec3 curr;

        float dist = 0.0f;

        for (int i = 0; i < 1000; i++) {
            curr.x = origin.x + ray.x * depth;
            curr.y = origin.y + ray.y * depth;
            curr.z = origin.z + ray.z * depth;

            dist = FLT_MAX;

            for (int j = 0; j < device_scene.spheres_length; j++) {
                Sphere sphere = device_scene.spheres[j];

                float d = sphere_dist(sphere,curr);

                if (d < dist) {
                    dist = d;
                    hit_id = sphere.id;
                }
            }

            for (int j = 0; j < device_scene.cuboids_length; j++) {
                Cuboid cuboid = device_scene.cuboids[j];

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

            if (depth > device_scene.far_clip_distance) {
                hit_id = 0;
                break;
            }
        }

        if (hit_id == 0) {
            result.r += sky.r * weight * record.r;
            result.g += sky.g * weight * record.g;
            result.b += sky.b * weight * record.b;

            return result;
        }

        RGBF intermediate_result;
        vec3 normal;
        float smoothness;
        RGBF emission;
        float intensity;

        for (int i = 0; i < device_scene.spheres_length; i++) {
            if (hit_id == device_scene.spheres[i].id) {
                intermediate_result = device_scene.spheres[i].color;
                normal = sphere_normal(device_scene.spheres[i], curr);
                smoothness = device_scene.spheres[i].smoothness;
                emission = device_scene.spheres[i].emission;
                intensity = device_scene.spheres[i].intensity;
                break;
            }
        }
        for (int i = 0; i < device_scene.cuboids_length; i++) {
            if (hit_id == device_scene.cuboids[i].id) {
                intermediate_result = device_scene.cuboids[i].color;
                normal = cuboid_normal(device_scene.cuboids[i], curr);
                smoothness = device_scene.cuboids[i].smoothness;
                emission = device_scene.cuboids[i].emission;
                intensity = device_scene.cuboids[i].intensity;
                break;
            }
        }

        result.r += emission.r * intensity * weight * record.r;
        result.g += emission.g * intensity * weight * record.g;
        result.b += emission.b * intensity * weight * record.b;

        curr.x += normal.x * epsilon * 2.0f;
        curr.y += normal.y * epsilon * 2.0f;
        curr.z += normal.z * epsilon * 2.0f;

        origin.x = curr.x;
        origin.y = curr.y;
        origin.z = curr.z;

        float alpha = 1.57079632679f - 3.1415926535f * curand_uniform(random);
        float beta = 1.57079632679f - 3.1415926535f * curand_uniform(random);
        float gamma = 1.57079632679f - 3.1415926535f * curand_uniform(random);

        ray.x =
        __cosf(alpha)*__cosf(beta)*normal.x+
            (__cosf(alpha)*__sinf(beta)*__sinf(gamma)-__sinf(alpha)*__cosf(gamma))*normal.y+
            (__cosf(alpha)*__sinf(beta)*__cosf(gamma)+__sinf(alpha)*__sinf(gamma))*normal.z;

        ray.y =
        __sinf(alpha)*__cosf(beta)*normal.x+
            (__sinf(alpha)*__sinf(beta)*__sinf(gamma)+__cosf(alpha)*__cosf(gamma))*normal.y+
            (__sinf(alpha)*__sinf(beta)*__cosf(gamma)-__cosf(alpha)*__sinf(gamma))*normal.z;

        ray.z =
            -__sinf(beta)*normal.x+
            __cosf(beta)*__sinf(gamma)*normal.y+
            __cosf(beta)*__cosf(gamma)*normal.z;

        float angle = normal.x * ray.x + normal.y * ray.y + normal.z * ray.z;

        record.r *= intermediate_result.r;
        record.g *= intermediate_result.g;
        record.b *= intermediate_result.b;

        weight *= 0.5f * 0.31830988618f * 0.31830988618f * angle;
    }

    return result;
}

__device__
vec3 rotate_vector_by_quaternion(vec3 v, Quaternion q) {
    vec3 result;

    vec3 u;
    u.x = q.x;
    u.y = q.y;
    u.z = q.z;

    float s = q.w;

    float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
    float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

    vec3 cross;

    cross.x = u.y*v.z - u.z*v.y;
    cross.y = u.z*v.x - u.x*v.z;
    cross.z = u.x*v.y - u.y*v.x;

    result.x = 2.0f * dot_uv * u.x + ((s*s)-dot_uu) * v.x + 2.0f * s * cross.x;
    result.y = 2.0f * dot_uv * u.y + ((s*s)-dot_uu) * v.y + 2.0f * s * cross.y;
    result.z = 2.0f * dot_uv * u.z + ((s*s)-dot_uu) * v.z + 2.0f * s * cross.z;

    return result;
}

__global__
void trace_rays() {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    while (id < device_amount) {
        int x = id % device_width;
        int y = (id - x) / device_width;

        curandStateXORWOW_t random;

        curand_init(id, 0, 0, &random);

        vec3 ray;
        vec3 default_ray;

        RGBF result;

        result.r = 0.0f;
        result.g = 0.0f;
        result.b = 0.0f;

        for (int i = 0; i < device_diffuse_samples; i++) {
            default_ray.x = -device_scene.camera.fov + device_step * x + device_offset_x * curand_uniform(&random) * 2.0f;
            default_ray.y = -device_vfov + device_step * y + device_offset_y * curand_uniform(&random) * 2.0f;
            default_ray.z = -1.0f;

            ray = rotate_vector_by_quaternion(default_ray, device_camera_rotation);

            float ray_length = __frsqrt_rn(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

            ray.x *= ray_length;
            ray.y *= ray_length;
            ray.z *= ray_length;

            RGBF color = trace_ray_iterative(device_scene.camera.pos, ray, &random);

            result.r += color.r;
            result.g += color.g;
            result.b += color.b;
        }

        float weight = 1.0f/(float)device_diffuse_samples;

        result.r *= weight;
        result.g *= weight;
        result.b *= weight;

        device_frame[id] = result;

        id += blockDim.x * gridDim.x;
    }
}

__global__
void set_up_raytracing_device() {
    device_amount = device_width * device_height;

    device_step = 2.0f * (device_scene.camera.fov / device_width);

    device_vfov = device_step * device_height / 2.0f;

    device_offset_x = (device_step / 2.0f);
    device_offset_y = (device_step / 2.0f);

    float alpha = device_scene.camera.rotation.x;
    float beta = device_scene.camera.rotation.y;
    float gamma = device_scene.camera.rotation.z;

    double cy = __cosf(gamma * 0.5);
    double sy = __sinf(gamma * 0.5);
    double cp = __cosf(beta * 0.5);
    double sp = __sinf(beta * 0.5);
    double cr = __cosf(alpha * 0.5);
    double sr = __sinf(alpha * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    device_camera_rotation = q;
}


extern "C" raytrace_instance* init_raytracing(const unsigned int width, const unsigned int height, const int reflection_depth, const int diffuse_samples) {
    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)malloc(sizeof(RGBF) * width * height);

    cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height);

    instance->reflection_depth = reflection_depth;
    instance->diffuse_samples = diffuse_samples;

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

    cudaMemcpyToSymbol(device_frame, &(instance->frame_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_scene, &scene_gpu, sizeof(Scene), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_reflection_depth, &(instance->reflection_depth), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_diffuse_samples, &(instance->diffuse_samples), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_width, &(instance->width), sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device_height, &(instance->height), sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    set_up_raytracing_device<<<1,1>>>();

    cudaDeviceSynchronize();

    trace_rays<<<blocks_per_grid,threads_per_block>>>();

    cudaDeviceSynchronize();

    cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost);

    puts(cudaGetErrorString(cudaGetLastError()));

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
