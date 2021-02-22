#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include "mesh.h"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>

const int threads_per_block = 256;
const int blocks_per_grid = 960;

__device__
const float epsilon = 0.001f;

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
curandStateXORWOW_t device_random;

__device__
vec3 cross_product(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.y*b.z - a.z*b.y;
    result.y = a.z*b.x - a.x*b.z;
    result.z = a.x*b.y - a.y*b.x;

    return result;
}

__device__
float dot_product(const vec3 a, const vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
vec3 vec_diff(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;

    return result;
}

__device__
vec3 get_coordinates_in_triangle(const Triangle triangle, const vec3 point) {
    const vec3 e1 = vec_diff(triangle.v2, triangle.v1);
    const vec3 e2 = vec_diff(triangle.v3, triangle.v1);
    const vec3 diff = vec_diff(point, triangle.v1);
    const float d00 = dot_product(e1,e1);
    const float d01 = dot_product(e1,e2);
    const float d11 = dot_product(e2,e2);
    const float d20 = dot_product(diff,e1);
    const float d21 = dot_product(diff,e2);
    const float denom = 1.0f / (d00 * d11 - d01 * d01);
    vec3 result;
    result.x = (d11 * d20 - d01 * d21) * denom;
    result.y = (d00 * d21 - d01 * d20) * denom;
    return result;
}

/*
 * Normals should really be spherically lerped but in most cases
 * the angle between the vertex normals is small enough, so that
 * the performance advantage is probably worth it.
 */
__device__
vec3 lerp_normals(const Triangle triangle, const float lambda, const float mu) {
    const vec3 e1 = vec_diff(triangle.vn2, triangle.vn1);
    const vec3 e2 = vec_diff(triangle.vn3, triangle.vn1);

    vec3 result;

    result.x = triangle.vn1.x + lambda * e1.x + mu * e2.x;
    result.y = triangle.vn1.y + lambda * e1.y + mu * e2.y;
    result.z = triangle.vn1.z + lambda * e1.z + mu * e2.z;

    const float length = 1.0f / sqrtf(dot_product(result, result));

    result.x *= length;
    result.y *= length;
    result.z *= length;

    return result;
}

/*
 * Based on:
 * A. Majercik, C. Crassin, P. Shirley, M. McGuire,
 * "A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering",
 * Journal of Computer Graphics Techniques, 7(3), pp. 66-82, 2018
 *
 * This implementation is probably not quite sufficient
 *
 * Assumes that origin is not inside a box
 */
/*__device__
float cuboid_intersect(Cuboid cuboid, vec3 origin, vec3 ray) {
    origin.x -= cuboid.pos.x;
    origin.y -= cuboid.pos.y;
    origin.z -= cuboid.pos.z;

    vec3 d;

    vec3 sign;

    sign.x = copysignf(1.0f, -ray.x);
    sign.y = copysignf(1.0f, -ray.y);
    sign.z = copysignf(1.0f, -ray.z);

    d.x = cuboid.size.x * sign.x - origin.x;
    d.y = cuboid.size.y * sign.y - origin.y;
    d.z = cuboid.size.z * sign.z - origin.z;

    d.x /= ray.x;
    d.y /= ray.y;
    d.z /= ray.z;

    const bool test_x = (d.x >= 0.0f) && (fabsf(origin.y + ray.y * d.x) < cuboid.size.y) && (fabsf(origin.z + ray.z * d.x) < cuboid.size.z);
    const bool test_y = (d.y >= 0.0f) && (fabsf(origin.x + ray.x * d.y) < cuboid.size.x) && (fabsf(origin.z + ray.z * d.y) < cuboid.size.z);
    const bool test_z = (d.z >= 0.0f) && (fabsf(origin.x + ray.x * d.z) < cuboid.size.x) && (fabsf(origin.y + ray.y * d.z) < cuboid.size.y);

    vec3 sgn;

    sgn.x = 0.0f;
    sgn.y = 0.0f;
    sgn.z = 0.0f;

    if (test_x) {
        sgn.x = sign.x;
    }
    else if (test_y) {
        sgn.y = sign.y;
    }
    else if (test_z) {
        sgn.z = sign.z;
    }

    if (sgn.x != 0.0f) {
        return d.x;
    }
    else if (sgn.y != 0.0f) {
        return d.y;
    }
    else if (sgn.z != 0.0f) {
        return d.z;
    }
    else {
        return FLT_MAX;
    }
}*/

/*
 * Based on
 *
 * Möller, Tomas, Trumbore, Ben,
 * "Fast, Minimum Storage Ray-Triangle Intersection",
 * Journal of Graphics Tools, 2, pp. 21-28, 1997.
 *
 * This suggests storing vertex1, edge1 and edge2 in triangle instead of the three vertices
 */
__device__
float triangle_intersection(Triangle triangle, vec3 origin, vec3 ray) {
    const vec3 e1 = vec_diff(triangle.v2, triangle.v1);
    const vec3 e2 = vec_diff(triangle.v3, triangle.v1);

    const vec3 h = cross_product(ray, e2);
    const float a = dot_product(e1, h);

    if (a > -epsilon && a < epsilon) return FLT_MAX;

    const float f = 1.0f / a;
    const vec3 s = vec_diff(origin, triangle.v1);
    const float u = f * dot_product(s, h);

    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    const vec3 q = cross_product(s, e1);
    const float v = f * dot_product(ray, q);

    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    const float t = f * dot_product(e2, q);

    if (t > epsilon) {
        return t;
    } else {
        return FLT_MAX;
    }
}

/*
 * Based on
 * J. Frisvad,
 * "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization",
 * Journal of Graphics Tools, 16(3), pp. 151-159, 2012
 */
__device__
vec3 sample_ray_from_angles_and_vector(const float theta, const float phi, const vec3 basis) {
    vec3 u1, u2;
    if (basis.z < -0.9999999f) {
        u1.x = 0.0f;
        u1.y = -1.0f;
        u1.z = 0.0f;
        u2.x = -1.0f;
        u2.y = 0.0f;
        u2.z = 0.0f;
    }
    else
    {
        const float a = 1.0f/(1.0f+basis.z);
        const float b = -basis.x*basis.y*a;
        u1.x = 1.0f - basis.x*basis.x*a;
        u1.y = b;
        u1.z = -basis.x;
        u2.x = b;
        u2.y = 1.0f-basis.y*basis.y*a;
        u2.z = -basis.y;
    }


    const float c1 = sinf(theta) * cosf(phi);
    const float c2 = sinf(theta) * sinf(phi);
    const float c3 = cosf(theta);

    vec3 result;

    result.x = c1 * u1.x + c2 * u2.x + c3 * basis.x;
    result.y = c1 * u1.y + c2 * u2.y + c3 * basis.y;
    result.z = c1 * u1.z + c2 * u2.z + c3 * basis.z;

    return result;
}

/*
 * Uses some self modified Blinn-Phong BRDF.
 */
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
    sky.r = 1.0f;
    sky.g = 1.0f;
    sky.b = 1.0f;

    for (int reflection_number = 0; reflection_number < device_reflection_depth; reflection_number++) {
        float depth = 1000.0f;

        unsigned int hit_id = 0xffffffff;

        vec3 curr;

        for (unsigned int i = 0; i < device_scene.triangles_length; i++) {
            const float d = triangle_intersection(device_scene.triangles[i], origin, ray);

            if (d < depth) {
                depth = d;
                hit_id = i;
            }
        }

        if (hit_id == 0xffffffff) {
            result.r += sky.r * weight * record.r;
            result.g += sky.g * weight * record.g;
            result.b += sky.b * weight * record.b;

            return result;
        }

        curr.x = origin.x + ray.x * depth;
        curr.y = origin.y + ray.y * depth;
        curr.z = origin.z + ray.z * depth;

        Triangle hit_triangle = device_scene.triangles[hit_id];

        RGBF albedo;
        albedo.r = 0.5f;
        albedo.g = 0.5f;
        albedo.b = 0.5f;
        vec3 normal;
        normal.x = 1.0f;
        normal.y = 1.0f;
        normal.z = 1.0f;
        const float smoothness = 0.0f;
        RGBF emission;
        emission.r = 0.0f;
        emission.g = 0.0f;
        emission.b = 0.0f;
        const float intensity = 0.0f;
        const float metallic = 1.0f;

        normal = get_coordinates_in_triangle(hit_triangle, curr);

        const float lambda = normal.x;
        const float mu = normal.y;

        normal = lerp_normals(hit_triangle, lambda, mu);

        result.r += emission.r * intensity * weight * record.r;
        result.g += emission.g * intensity * weight * record.g;
        result.b += emission.b * intensity * weight * record.b;

        curr.x += normal.x * epsilon * 2.0f;
        curr.y += normal.y * epsilon * 2.0f;
        curr.z += normal.z * epsilon * 2.0f;

        origin.x = curr.x;
        origin.y = curr.y;
        origin.z = curr.z;

        vec3 specular_ray;
        const float projected_length = ray.x * normal.x + ray.y * normal.y + ray.z * normal.z;

        specular_ray.x = ray.x - 2.0f * projected_length * normal.x;
        specular_ray.y = ray.y - 2.0f * projected_length * normal.y;
        specular_ray.z = ray.z - 2.0f * projected_length * normal.z;

        const float specular_ray_length = rsqrtf(specular_ray.x * specular_ray.x + specular_ray.y * specular_ray.y + specular_ray.z * specular_ray.z);

        specular_ray.x *= specular_ray_length;
        specular_ray.y *= specular_ray_length;
        specular_ray.z *= specular_ray_length;

        const float specular = curand_uniform(&device_random);

        if (specular < smoothness) {
            record.r *= (albedo.r * metallic + 1.0f - metallic);
            record.g *= (albedo.g * metallic + 1.0f - metallic);
            record.b *= (albedo.b * metallic + 1.0f - metallic);

            const float n = 100.0f * smoothness / (1.0f + epsilon - smoothness);

            const float alpha = acosf(__powf(curand_uniform(random),(1.0f/(1.0f+n))));
            const float gamma = 2.0f * 3.1415926535f * curand_uniform(random);

            ray = sample_ray_from_angles_and_vector(alpha, gamma, specular_ray);

            weight *= 2.0f * 3.1415926535f;
        }
        else
        {
            record.r *= albedo.r;
            record.g *= albedo.g;
            record.b *= albedo.b;

            const float alpha = acosf(__fsqrt_rn(curand_uniform(random)));
            const float gamma = 2.0f * 3.1415926535f * curand_uniform(random);

            ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

            const float angle = normal.x * ray.x + normal.y * ray.y + normal.z * ray.z;

            weight *= 3.1415926535f / (angle + epsilon);
        }

        const float angle = specular_ray.x * ray.x + specular_ray.y * ray.y + specular_ray.z * ray.z;

        weight *= ((1.0f - smoothness) * 0.31830988618f) + (smoothness * 0.5f * 0.31830988618f);
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
            default_ray.y = device_vfov - device_step * y - device_offset_y * curand_uniform(&random) * 2.0f;
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

    curand_init(0,0,0,&device_random);
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

    cudaMalloc((void**) &(scene_gpu.triangles), sizeof(Triangle) * scene_gpu.triangles_length);
    puts(cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(scene_gpu.triangles, scene.triangles, sizeof(Triangle) * scene.triangles_length, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));

    cudaMemcpyToSymbol(device_frame, &(instance->frame_buffer_gpu), sizeof(RGBF*), 0, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));
    cudaMemcpyToSymbol(device_scene, &scene_gpu, sizeof(Scene), 0, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));
    cudaMemcpyToSymbol(device_reflection_depth, &(instance->reflection_depth), sizeof(int), 0, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));
    cudaMemcpyToSymbol(device_diffuse_samples, &(instance->diffuse_samples), sizeof(int), 0, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));
    cudaMemcpyToSymbol(device_width, &(instance->width), sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    puts(cudaGetErrorString(cudaGetLastError()));
    cudaMemcpyToSymbol(device_height, &(instance->height), sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    puts(cudaGetErrorString(cudaGetLastError()));

    set_up_raytracing_device<<<1,1>>>();

    cudaDeviceSynchronize();

    puts(cudaGetErrorString(cudaGetLastError()));

    trace_rays<<<blocks_per_grid,threads_per_block>>>();

    cudaDeviceSynchronize();

    puts(cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost);

    puts(cudaGetErrorString(cudaGetLastError()));

    cudaFree(scene_gpu.triangles);
}

extern "C" void frame_buffer_to_image(Camera camera, raytrace_instance* instance, RGB8* image) {
    for (int j = 0; j < instance->height; j++) {
        for (int i = 0; i < instance->width; i++) {
            RGB8 pixel;
            RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

            pixel.r = (uint8_t)(min(255.9f, pixel_float.r * 255.9f));
            pixel.g = (uint8_t)(min(255.9f, pixel_float.g * 255.9f));
            pixel.b = (uint8_t)(min(255.9f, pixel_float.b * 255.9f));


            image[i + instance->width * j] = pixel;
        }
    }
}

extern "C" void free_raytracing(raytrace_instance* instance) {
    cudaFree(instance->frame_buffer_gpu);

    free(instance->frame_buffer);

    free(instance);
}
