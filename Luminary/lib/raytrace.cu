#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include "mesh.h"
#include "SDL/SDL.h"
#include "cuda/utils.cuh"
#include "cuda/math.cuh"
#include "cuda/sky.cuh"
#include "cuda/brdf.cuh"
#include "cuda/bvh.cuh"
#include "cuda/directives.cuh"
#include "cuda/random.cuh"
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <immintrin.h>

const static int threads_per_block = 128;
const static int blocks_per_grid = 512;

struct Sample {
    vec3 origin;
    vec3 ray;
    RGBF record;
    RGBF result;
    short2 state; //x = depth, y = 1st bit (finished?) 2nd bit (albedo buffer written?)
    ushort2 index;
    int random_index;
} typedef Sample;

//---------------------------------
// Path Tracing
//---------------------------------

__device__
float get_light_angle(Light light, vec3 pos) {
    const float d = get_length(vec_diff(pos, light.pos)) + eps;
    return fminf(PI/2.0f,asinf(light.radius / d));
}

__device__
Sample trace_ray_iterative(const Sample input_sample) {
    int albedo_buffer_written = input_sample.state.y & 0b10;
    const ushort2 index = input_sample.index;
    int random_index = input_sample.random_index;

    RGBF result = input_sample.result;
    RGBF record = input_sample.record;

    vec3 ray = input_sample.ray;
    vec3 origin = input_sample.origin;

    short2 state = input_sample.state;
    const int starting_threads = __popc(__activemask());

    for (; state.x < device_reflection_depth; state.x++) {
        traversal_result traversal = traverse_bvh(origin, ray, device_scene.nodes, device_scene.traversal_triangles);

        if (traversal.hit_id == 0xffffffff) {
            RGBF sky = get_sky_color(ray);

            if (device_denoiser && !albedo_buffer_written) {
                RGBF sum = device_albedo_buffer[index.x + index.y * device_width];
                sum.r += sky.r;
                sum.g += sky.g;
                sum.b += sky.b;
                device_albedo_buffer[index.x + index.y * device_width] = sum;

                albedo_buffer_written++;
                state.y |= 0b10;
            }

            result.r += sky.r * record.r;
            result.g += sky.g * record.g;
            result.b += sky.b * record.b;

            state.y |= 0b1;

            break;
        }

        vec3 curr;
        curr.x = origin.x + ray.x * traversal.depth;
        curr.y = origin.y + ray.y * traversal.depth;
        curr.z = origin.z + ray.z * traversal.depth;

        const float4* hit_address = (float4*)(device_scene.triangles + traversal.hit_id);

        const float4 t1 = __ldg(hit_address);
        const float4 t2 = __ldg(hit_address + 1);
        const float4 t3 = __ldg(hit_address + 2);
        const float4 t4 = __ldg(hit_address + 3);
        const float4 t5 = __ldg(hit_address + 4);
        const float4 t6 = __ldg(hit_address + 5);
        const float4 t7 = __ldg(hit_address + 6);

        vec3 vertex;
        vertex.x = t1.x;
        vertex.y = t1.y;
        vertex.z = t1.z;

        vec3 edge1;
        edge1.x = t1.w;
        edge1.y = t2.x;
        edge1.z = t2.y;

        vec3 edge2;
        edge2.x = t2.z;
        edge2.y = t2.w;
        edge2.z = t3.x;

        vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, curr);

        const float lambda = normal.x;
        const float mu = normal.y;

        vec3 vertex_normal;
        vertex_normal.x = t3.y;
        vertex_normal.y = t3.z;
        vertex_normal.z = t3.w;

        vec3 edge1_normal;
        edge1_normal.x = t4.x;
        edge1_normal.y = t4.y;
        edge1_normal.z = t4.z;

        vec3 edge2_normal;
        edge2_normal.x = t4.w;
        edge2_normal.y = t5.x;
        edge2_normal.z = t5.y;

        normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu);

        UV vertex_texture;
        vertex_texture.u = t5.z;
        vertex_texture.v = t5.w;

        UV edge1_texture;
        edge1_texture.u = t6.x;
        edge1_texture.v = t6.y;

        UV edge2_texture;
        edge2_texture.u = t6.z;
        edge2_texture.v = t6.w;

        const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

        vec3 face_normal;
        face_normal.x = t7.x;
        face_normal.y = t7.y;
        face_normal.z = t7.z;

        const int texture_object = __float_as_int(t7.w);

        const ushort4 maps = __ldg((ushort4*)(device_texture_assignments + texture_object));

        float roughness;
        float metallic;
        float intensity;

        if (maps.z) {
            const float4 material_f = tex2D<float4>(device_material_atlas[maps.z], tex_coords.u, 1.0f - tex_coords.v);

            roughness = (1.0f - material_f.x) * (1.0f - material_f.x);
            metallic = material_f.y;
            intensity = material_f.z * 255.0f;
        } else {
            roughness = 0.81f;
            metallic = 0.0f;
            intensity = 1.0f;
        }

        if (maps.y) {
            #ifdef LIGHTS_AT_NIGHT_ONLY
            if (device_sun.y < NIGHT_THRESHOLD) {
            #endif
            const float4 illuminance_f = tex2D<float4>(device_illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

            RGBF emission;
            emission.r = illuminance_f.x;
            emission.g = illuminance_f.y;
            emission.b = illuminance_f.z;

            result.r += emission.r * intensity * record.r;
            result.g += emission.g * intensity * record.g;
            result.b += emission.b * intensity * record.b;

            #ifdef FIRST_LIGHT_ONLY
            const double max_result = fmaxf(result.r, fmaxf(result.g, result.b));
            if (max_result > eps) {
                state.y |= 0b1;
                break;
            }
            #endif

            #ifdef LIGHTS_AT_NIGHT_ONLY
            }
            #endif
        }

        RGBAF albedo;

        if (maps.x) {
            const float4 albedo_f = tex2D<float4>(device_albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
            albedo.r = albedo_f.x;
            albedo.g = albedo_f.y;
            albedo.b = albedo_f.z;
            albedo.a = albedo_f.w;
        } else {
            albedo.r = 0.9f;
            albedo.g = 0.9f;
            albedo.b = 0.9f;
            albedo.a = 1.0f;
        }

        if (sample_blue_noise(index.x, index.y, random_index, 40) > albedo.a) {
            origin.x = curr.x + 2.0f * eps * ray.x;
            origin.y = curr.y + 2.0f * eps * ray.y;
            origin.z = curr.z + 2.0f * eps * ray.z;
        } else {
            if (device_denoiser && !albedo_buffer_written) {
                RGBF sum = device_albedo_buffer[index.x + index.y * device_width];
                sum.r += albedo.r;
                sum.g += albedo.g;
                sum.b += albedo.b;
                device_albedo_buffer[index.x + index.y * device_width] = sum;

                albedo_buffer_written++;
                state.y |= 0b10;
            }

            const float specular_probability = lerp(0.5f, 1.0f - eps, metallic);

            if (dot_product(normal, face_normal) < 0.0f) {
                face_normal = scale_vector(face_normal, -1.0f);
            }

            const vec3 V = scale_vector(ray, -1.0f);

            if (dot_product(face_normal, V) < 0.0f) {
                normal = scale_vector(normal, -1.0f);
                face_normal = scale_vector(face_normal, -1.0f);
            }

            origin.x = curr.x + face_normal.x * (eps * 8.0f);
            origin.y = curr.y + face_normal.y * (eps * 8.0f);
            origin.z = curr.z + face_normal.z * (eps * 8.0f);

            const float light_sample = sample_blue_noise(index.x, index.y, random_index, 50);

            float light_angle;
            vec3 light_source;
            #ifdef LIGHTS_AT_NIGHT_ONLY
            const int light_count = (device_sun.y < NIGHT_THRESHOLD) ? device_scene.lights_length - 1 : 1;
            #else
            const int light_count = device_scene.lights_length;
            #endif

            const float light_sample_probability = 1.0f - 1.0f/(light_count + 1);


            if (light_sample < light_sample_probability) {
                #ifdef LIGHTS_AT_NIGHT_ONLY
                    const uint32_t light = (device_sun.y < NIGHT_THRESHOLD && light_count > 0) ? 1 + (uint32_t)(sample_blue_noise(index.x, index.y, random_index, 51) * light_count) : 0;
                #else
                    const uint32_t light = (uint32_t)(sample_blue_noise(index.x, index.y, random_index, 51) * light_count);
                #endif

                const float4 light_data = __ldg((float4*)(device_scene.lights + light));
                vec3 light_pos;
                light_pos.x = light_data.x;
                light_pos.y = light_data.y;
                light_pos.z = light_data.z;
                light_pos = vec_diff(light_pos, origin);
                light_source = normalize_vector(light_pos);
                const float d = get_length(light_pos) + eps;
                light_angle = fminf(PI/2.0f,asinf(light_data.w / d)) * 2.0f / PI;
            }

            if (sample_blue_noise(index.x, index.y, random_index, 10) < specular_probability) {
                const float alpha = roughness * roughness;

                const float beta = sample_blue_noise(index.x, index.y, random_index, 2);
                const float gamma = 2.0f * 3.1415926535f * sample_blue_noise(index.x, index.y, random_index, 3);

                const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);

                float weight = 1.0f;

                const vec3 V_local = rotate_vector_by_quaternion(V, rotation_to_z);
                vec3 H_local;

                if (alpha < eps) {
                    H_local.x = 0.0f;
                    H_local.y = 0.0f;
                    H_local.z = 1.0f;
                } else {
                    const vec3 S_local = rotate_vector_by_quaternion(
                        normalize_vector(sample_ray_from_angles_and_vector(beta * light_angle, gamma, light_source)),
                        rotation_to_z);

                    if (light_sample < light_sample_probability && S_local.z > 0.0f) {
                        H_local.x = S_local.x + V_local.x;
                        H_local.y = S_local.y + V_local.y;
                        H_local.z = S_local.z + V_local.z;

                        H_local = normalize_vector(H_local);

                        weight = (1.0f/light_sample_probability) * light_angle * light_count;
                    } else {
                        H_local = sample_GGX_VNDF(V_local, alpha, sample_blue_noise(index.x, index.y, random_index, 4), sample_blue_noise(index.x, index.y, random_index, 5));

                        if (S_local.z > 0.0f) weight = (1.0f/(1.0f-light_sample_probability));
                    }
                }

                const vec3 ray_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

                const float HdotR = fmaxf(eps, fminf(1.0f, dot_product(H_local, ray_local)));
                const float NdotR = fmaxf(eps, fminf(1.0f, ray_local.z));
                const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));

                ray = normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));

                vec3 specular_f0;
                specular_f0.x = lerp(0.04f, albedo.r, metallic);
                specular_f0.y = lerp(0.04f, albedo.g, metallic);
                specular_f0.z = lerp(0.04f, albedo.b, metallic);

                const vec3 F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotR);

                const float milchs_energy_recovery = lerp(1.0f, 1.51f + 1.51f * NdotV, roughness);

                weight *= milchs_energy_recovery * Smith_G2_over_G1_height_correlated(alpha * alpha, NdotR, NdotV) / specular_probability;

                record.r *= F.x * weight;
                record.g *= F.y * weight;
                record.b *= F.z * weight;
            }
            else
            {
                float weight = 1.0f;

                const float alpha = acosf(sqrtf(sample_blue_noise(index.x, index.y, random_index, 2)));
                const float gamma = 2.0f * PI * sample_blue_noise(index.x, index.y, random_index, 3);

                ray = normalize_vector(sample_ray_from_angles_and_vector(alpha * light_angle, gamma, light_source));
                const float light_feasible = dot_product(ray, normal);

                if (light_sample < light_sample_probability && light_feasible >= 0.0f) {
                    weight = (1.0f/light_sample_probability) * light_angle * light_count;
                } else {
                    ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

                    if (light_feasible >=0.0f) weight = (1.0f/(1.0f-light_sample_probability));
                }

                vec3 H;
                H.x = V.x + ray.x;
                H.y = V.y + ray.y;
                H.z = V.z + ray.z;
                H = normalize_vector(H);

                const float half_angle = fmaxf(eps, fminf(dot_product(H,ray),1.0f));
                const float energyFactor = lerp(1.0f, 1.0f/1.51f, roughness);

                const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

                const float angle = fmaxf(eps, fminf(dot_product(normal, ray),1.0f));
                const float previous_angle = fmaxf(eps, fminf(dot_product(V, normal),1.0f));

                const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
                const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

                weight *= FDL * FDV * energyFactor * (1.0f - metallic) / (1.0f - specular_probability);

                record.r *= albedo.r * weight;
                record.g *= albedo.g * weight;
                record.b *= albedo.b * weight;
            }
        }

        #ifdef WEIGHT_BASED_EXIT
        const double max_record = fmaxf(record.r, fmaxf(record.g, record.b));
        if (max_record < CUTOFF ||
        (max_record < PROBABILISTIC_CUTOFF && sample_blue_noise(index.x, index.y, random_index, 20) > (max_record - CUTOFF)/(CUTOFF-PROBABILISTIC_CUTOFF)))
        {
            state.y |= 0b1;
            break;
        }
        #endif

        #ifdef LOW_QUALITY_LONG_BOUNCES
        if (state.x >= MIN_BOUNCES && sample_blue_noise(index.x, index.y, random_index, 21) < 1.0f/device_reflection_depth) {
            state.y |= 0b1;
            break;
        }
        #endif

        if (__popc(__activemask()) < 0.1f *  starting_threads) break;

        random_index++;
    }

    if (state.x >= device_reflection_depth - 1) state.y |= 0b1;

    Sample output_sample;
    output_sample.origin = origin;
    output_sample.ray = ray;
    output_sample.result = result;
    output_sample.record = record;
    output_sample.state = state;
    output_sample.index = index;
    output_sample.random_index = random_index + 1;

    return output_sample;
}

__global__
void trace_rays(volatile uint32_t* progress, const int temporal_frames) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    Sample sample;
    sample.state.y = 0b11;
    int i = 0;

    RGBF pixel;
    pixel.r = 0.0f;
    pixel.g = 0.0f;
    pixel.b = 0.0f;

    while (id < device_amount) {
        if (sample.state.y & 0b1) {
            vec3 ray;
            vec3 default_ray;

            const int x = (id / 1) % device_width;
            const int y = (id % 1) + 1 * (id / (1 * device_width));

            const int random_index = temporal_frames + ((id * 3) << 10);

            default_ray.x = device_scene.camera.focal_length * (-device_scene.camera.fov + device_step * x + device_offset_x * sample_blue_noise(x,y,random_index,8) * 2.0f);
            default_ray.y = device_scene.camera.focal_length * (device_vfov - device_step * y - device_offset_y * sample_blue_noise(x,y,random_index,9) * 2.0f);
            default_ray.z = -device_scene.camera.focal_length;

            const float alpha = sample_blue_noise(x,y,random_index,0) * 2.0f * PI;
            const float beta  = sample_blue_noise(x,y,random_index,1) * device_scene.camera.aperture_size;

            vec3 point_on_aperture;
            point_on_aperture.x = cosf(alpha) * beta;
            point_on_aperture.y = sinf(alpha) * beta;
            point_on_aperture.z = 0.0f;

            default_ray = vec_diff(default_ray, point_on_aperture);
            ray = normalize_vector(rotate_vector_by_quaternion(default_ray, device_camera_rotation));
            point_on_aperture = rotate_vector_by_quaternion(point_on_aperture, device_camera_rotation);

            sample.ray = ray;
            sample.origin.x = device_scene.camera.pos.x + point_on_aperture.x;
            sample.origin.y = device_scene.camera.pos.y + point_on_aperture.y;
            sample.origin.z = device_scene.camera.pos.z + point_on_aperture.z;
            sample.index.x = x;
            sample.index.y = y;
            sample.random_index = random_index;
            sample.state.x = 0;
            sample.state.y = 0;
            sample.record.r = 1.0f;
            sample.record.g = 1.0f;
            sample.record.b = 1.0f;
            sample.result.r = 0.0f;
            sample.result.g = 0.0f;
            sample.result.b = 0.0f;
        }

        sample = trace_ray_iterative(sample);

        if (sample.state.y & 0b1) {
            if (isnan(sample.result.r) || isinf(sample.result.r)) {
                sample.result.r = 1.0f;
            }
            if (isnan(sample.result.g) || isinf(sample.result.g)) {
                sample.result.g = 0.0f;
            }
            if (isnan(sample.result.b) || isinf(sample.result.b)) {
                sample.result.b = 0.0f;
            }

            pixel.r += sample.result.r;
            pixel.g += sample.result.g;
            pixel.b += sample.result.b;

            i++;
        }

        if (i == device_diffuse_samples) {
            const float weight = 1.0f/(float)device_diffuse_samples;

            pixel.r *= weight;
            pixel.g *= weight;
            pixel.b *= weight;

            const int buffer_index = sample.index.x + sample.index.y * device_width;

            if (device_denoiser) {
                RGBF sum = device_albedo_buffer[buffer_index];
                sum.r *= weight;
                sum.g *= weight;
                sum.b *= weight;
                device_albedo_buffer[buffer_index] = sum;
            }

            if (temporal_frames) {
                RGBF temporal_pixel = device_frame[buffer_index];
                pixel.r = (pixel.r + temporal_pixel.r * temporal_frames) / (temporal_frames + 1);
                pixel.g = (pixel.g + temporal_pixel.g * temporal_frames) / (temporal_frames + 1);
                pixel.b = (pixel.b + temporal_pixel.b * temporal_frames) / (temporal_frames + 1);
            }

            device_frame[buffer_index] = pixel;

            id += blockDim.x * gridDim.x;
            i = 0;

            pixel.r = 0.0f;
            pixel.g = 0.0f;
            pixel.b = 0.0f;

            atomicAdd((uint32_t*)progress, 1);
            __threadfence_system();
        }
    }
}

static void update_sun(const Scene scene) {
    vec3 sun;
    sun.x = sinf(scene.azimuth) * cosf(scene.altitude);
    sun.y = sinf(scene.altitude);
    sun.z = cosf(scene.azimuth) * cosf(scene.altitude);
    sun = normalize_vector(sun);

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

    instance->scene_gpu = scene;

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

extern "C" void trace_scene(Scene scene, raytrace_instance* instance, const int progress, const int temporal_frames) {
    volatile uint32_t *progress_gpu, *progress_cpu;

    cudaEvent_t kernelFinished;
    gpuErrchk(cudaEventCreate(&kernelFinished));
    gpuErrchk(cudaHostAlloc((void**)&progress_cpu, sizeof(uint32_t), cudaHostAllocMapped));
    gpuErrchk(cudaHostGetDevicePointer((uint32_t**)&progress_gpu, (uint32_t*)progress_cpu, 0));
    *progress_cpu = 0;

    gpuErrchk(cudaMemcpyToSymbol(device_scene, &(instance->scene_gpu), sizeof(Scene), 0, cudaMemcpyHostToDevice));

    update_sun(instance->scene_gpu);
    update_camera_pos(instance->scene_gpu, instance->width, instance->height);

    clock_t t = clock();

    trace_rays<<<blocks_per_grid,threads_per_block>>>(progress_gpu, temporal_frames);

    if (progress == 1) {
        gpuErrchk(cudaEventRecord(kernelFinished));

        uint32_t progress = 0;
        const uint32_t total_pixels = instance->width * instance->height;
        const float ratio = 1.0f/((float)instance->width * (float)instance->height);
        while (cudaEventQuery(kernelFinished) != cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::microseconds(100000));
            gpuErrchk(cudaPeekAtLastError());
            uint32_t new_progress = *progress_cpu;
            if (new_progress > progress) {
                progress = new_progress;
            }
            clock_t curr_time = clock();
            const double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
            printf("\r                                                                                                          \rProgress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs - Performance: %.1f Mrays/s",
                (float)progress * ratio * 100, time_elapsed,
                (total_pixels - progress) * (time_elapsed/progress),
                0.000001 * instance->diffuse_samples * instance->reflection_depth * progress / time_elapsed);
        }

        printf("\r                                                                                                              \r");
    }

    gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void free_inputs(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->scene_gpu.texture_assignments));
    gpuErrchk(cudaFree(instance->scene_gpu.triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.traversal_triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.nodes));
    gpuErrchk(cudaFree(instance->scene_gpu.lights));
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

extern "C" void initiliaze_realtime(raytrace_instance* instance) {
    gpuErrchk(cudaMalloc((void**) &(instance->buffer_8bit_gpu), sizeof(RGB8) * instance->width * instance->height));
    gpuErrchk(cudaMemcpyToSymbol(device_frame_8bit, &(instance->buffer_8bit_gpu), sizeof(RGB8*), 0, cudaMemcpyHostToDevice));
}

extern "C" void free_realtime(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->buffer_8bit_gpu));
}

extern "C" void copy_framebuffer_to_8bit(RGB8* buffer, RGBF* source, raytrace_instance* instance) {
    convert_RGBF_to_RGB8<<<blocks_per_grid,threads_per_block>>>(source);
    gpuErrchk(cudaMemcpy(buffer, instance->buffer_8bit_gpu, sizeof(RGB8) * instance->width * instance->height, cudaMemcpyDeviceToHost));
}

struct realtime_denoise {
    OptixDeviceContext ctx;
    OptixDenoiser denoiser;
    OptixDenoiserOptions opt;
    OptixDenoiserSizes denoiserReturnSizes;
    CUdeviceptr denoiserState;
    CUdeviceptr denoiserScratch;
    OptixImage2D inputLayer[2];
    OptixImage2D outputLayer;
    CUdeviceptr hdr_intensity;
    CUdeviceptr avg_color;
} typedef realtime_denoise;

extern "C" void* initialize_optix_denoise_for_realtime(raytrace_instance* instance) {
    OPTIX_CHECK(optixInit());

    realtime_denoise* denoise_setup = (realtime_denoise*)malloc(sizeof(realtime_denoise));

    OPTIX_CHECK(optixDeviceContextCreate((CUcontext)0,(OptixDeviceContextOptions*)0, &denoise_setup->ctx));

    denoise_setup->opt.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

    OPTIX_CHECK(optixDenoiserCreate(denoise_setup->ctx, &denoise_setup->opt, &denoise_setup->denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoise_setup->denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoise_setup->denoiser, instance->width, instance->height, &denoise_setup->denoiserReturnSizes));

    gpuErrchk(cudaMalloc((void**) &denoise_setup->denoiserState, denoise_setup->denoiserReturnSizes.stateSizeInBytes));

    const size_t scratchSize = (denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes) ?
                                denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes :
                                denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes;

    gpuErrchk(cudaMalloc((void**) &denoise_setup->denoiserScratch, scratchSize));


    OPTIX_CHECK(optixDenoiserSetup(denoise_setup->denoiser, 0,
      instance->width, instance->height,
      denoise_setup->denoiserState,
      denoise_setup->denoiserReturnSizes.stateSizeInBytes,
      denoise_setup->denoiserScratch,
      scratchSize));

    denoise_setup->inputLayer[0].data = (CUdeviceptr)instance->frame_buffer_gpu;
    denoise_setup->inputLayer[0].width = instance->width;
    denoise_setup->inputLayer[0].height = instance->height;
    denoise_setup->inputLayer[0].rowStrideInBytes = instance->width * sizeof(RGBF);
    denoise_setup->inputLayer[0].pixelStrideInBytes = sizeof(RGBF);
    denoise_setup->inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT3;

    denoise_setup->inputLayer[1].data = (CUdeviceptr)instance->albedo_buffer_gpu;
    denoise_setup->inputLayer[1].width = instance->width;
    denoise_setup->inputLayer[1].height = instance->height;
    denoise_setup->inputLayer[1].rowStrideInBytes = instance->width * sizeof(RGBF);
    denoise_setup->inputLayer[1].pixelStrideInBytes = sizeof(RGBF);
    denoise_setup->inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT3;

    RGBF* output;
    gpuErrchk(cudaMalloc((void**) &output, sizeof(RGBF) * instance->width * instance->height));

    denoise_setup->outputLayer.data = (CUdeviceptr)output;
    denoise_setup->outputLayer.width = instance->width;
    denoise_setup->outputLayer.height = instance->height;
    denoise_setup->outputLayer.rowStrideInBytes = instance->width * sizeof(RGBF);
    denoise_setup->outputLayer.pixelStrideInBytes = sizeof(RGBF);
    denoise_setup->outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    gpuErrchk(cudaMalloc((void**) &denoise_setup->hdr_intensity, sizeof(float)));

    gpuErrchk(cudaMalloc((void**) &denoise_setup->avg_color, sizeof(float) * 3));

    return denoise_setup;
}

extern "C" RGBF* denoise_with_optix_realtime(void* input) {
    realtime_denoise* denoise_setup = (realtime_denoise*) input;

    const size_t scratchSize = (denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes) ?
                                denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes :
                                denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes;

    OPTIX_CHECK(optixDenoiserComputeIntensity(denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], denoise_setup->hdr_intensity, denoise_setup->denoiserScratch, scratchSize));

    OPTIX_CHECK(optixDenoiserComputeAverageColor(denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], denoise_setup->avg_color, denoise_setup->denoiserScratch, scratchSize));

    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 0;
    denoiserParams.hdrIntensity = denoise_setup->hdr_intensity;
    denoiserParams.blendFactor = 0.0f;
    denoiserParams.hdrAverageColor = denoise_setup->avg_color;

    OPTIX_CHECK(optixDenoiserInvoke(denoise_setup->denoiser,
        0,
        &denoiserParams,
        denoise_setup->denoiserState,
        denoise_setup->denoiserReturnSizes.stateSizeInBytes,
        &denoise_setup->inputLayer[0],
        2,
        0,
        0,
        &denoise_setup->outputLayer,
        denoise_setup->denoiserScratch,
        scratchSize));

    return (RGBF*)denoise_setup->outputLayer.data;
}

extern "C" void free_realtime_denoise(void* input) {
    realtime_denoise denoise_setup = *(realtime_denoise*) input;

    OPTIX_CHECK(optixDeviceContextDestroy(denoise_setup.ctx));
    OPTIX_CHECK(optixDenoiserDestroy(denoise_setup.denoiser));

    gpuErrchk(cudaFree((void*)denoise_setup.outputLayer.data));
    gpuErrchk(cudaFree((void*)denoise_setup.hdr_intensity));
    gpuErrchk(cudaFree((void*)denoise_setup.avg_color));
    gpuErrchk(cudaFree((void*)denoise_setup.denoiserState));
    gpuErrchk(cudaFree((void*)denoise_setup.denoiserScratch));
}
