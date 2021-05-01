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
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
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

//---------------------------------
// Path Tracing
//---------------------------------

__device__
float get_light_angle(Light light, vec3 pos) {
    const float d = get_length(vec_diff(pos, light.pos)) + eps;
    return atanf(light.radius / d);
}

__device__
RGBF trace_ray_iterative(vec3 origin, vec3 ray, curandStateXORWOW_t* random, int buffer_index) {
    int albedo_buffer_written = 0;

    RGBF result;
    result.r = 0.0f;
    result.g = 0.0f;
    result.b = 0.0f;

    float weight = 1.0f;
    RGBF record;
    record.r = 1.0f;
    record.g = 1.0f;
    record.b = 1.0f;

    for (int reflection_number = 0; reflection_number < device_reflection_depth; reflection_number++) {
        vec3 curr;

        traversal_result traversal = traverse_bvh(origin, ray, device_scene.nodes, device_scene.triangles);

        if (traversal.hit_id == 0xffffffff) {
            RGBF sky = get_sky_color(ray);

            if (device_denoiser && !albedo_buffer_written) {
                RGBF sum = device_albedo_buffer[buffer_index];
                sum.r += sky.r;
                sum.g += sky.g;
                sum.b += sky.b;
                device_albedo_buffer[buffer_index] = sum;

                albedo_buffer_written++;
            }

            result.r += sky.r * weight * record.r;
            result.g += sky.g * weight * record.g;
            result.b += sky.b * weight * record.b;

            return result;
        }

        curr.x = origin.x + ray.x * traversal.depth;
        curr.y = origin.y + ray.y * traversal.depth;
        curr.z = origin.z + ray.z * traversal.depth;

        Triangle hit_triangle = device_scene.triangles[traversal.hit_id];

        vec3 normal = get_coordinates_in_triangle(hit_triangle, curr);

        const float lambda = normal.x;
        const float mu = normal.y;

        normal = lerp_normals(hit_triangle, lambda, mu);

        UV tex_coords = lerp_uv(hit_triangle, lambda, mu);

        RGBAF albedo;
        RGBF emission;

        float4 albedo_f = tex2D<float4>(device_albedo_atlas[device_texture_assignments[hit_triangle.object_maps].albedo_map], tex_coords.u, 1.0f - tex_coords.v);
        float4 illumiance_f = tex2D<float4>(device_illuminance_atlas[device_texture_assignments[hit_triangle.object_maps].illuminance_map], tex_coords.u, 1.0f - tex_coords.v);
        float4 material_f = tex2D<float4>(device_material_atlas[device_texture_assignments[hit_triangle.object_maps].material_map], tex_coords.u, 1.0f - tex_coords.v);

        albedo.r = albedo_f.x;
        albedo.g = albedo_f.y;
        albedo.b = albedo_f.z;
        albedo.a = albedo_f.w;

        emission.r = illumiance_f.x;
        emission.g = illumiance_f.y;
        emission.b = illumiance_f.z;

        const float roughness = (1.0f - material_f.x) * (1.0f - material_f.x);
        const float metallic = material_f.y;
        const float intensity = material_f.z * 255.0f;

        result.r += emission.r * intensity * weight * record.r;
        result.g += emission.g * intensity * weight * record.g;
        result.b += emission.b * intensity * weight * record.b;

        const float transparent_pass = curand_uniform(random);

        if (transparent_pass > albedo.a) {
            origin.x = curr.x + 2.0f * eps * ray.x;
            origin.y = curr.y + 2.0f * eps * ray.y;
            origin.z = curr.z + 2.0f * eps * ray.z;

            continue;
        }

        if (device_denoiser && !albedo_buffer_written) {
            RGBF sum = device_albedo_buffer[buffer_index];
            sum.r += albedo.r;
            sum.g += albedo.g;
            sum.b += albedo.b;
            device_albedo_buffer[buffer_index] = sum;

            albedo_buffer_written++;
        }

        const float specular = curand_uniform(random);
        const float specular_probability = lerp(0.5f, 1.0f - eps, metallic);

        vec3 face_normal = normalize_vector(cross_product(hit_triangle.edge1,hit_triangle.edge2));

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

        const float light_sample = curand_uniform(random);
        float light_angle;
        vec3 light_source;

        if (light_sample < 0.5f) {
            const uint32_t light = (uint32_t)(curand_uniform(random) * device_scene.lights_length);
            light_source = normalize_vector(vec_diff(device_scene.lights[light].pos, origin));
            light_angle = get_light_angle(device_scene.lights[light], origin) * 2.0f / PI;
        }

        if (specular < specular_probability) {
            const float alpha = roughness * roughness;

            const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);

            const vec3 V_local = rotate_vector_by_quaternion(V, rotation_to_z);

            vec3 H_local;

            const float beta = acosf(curand_uniform(random));
            const float gamma = 2.0f * 3.1415926535f * curand_uniform(random);

            const vec3 S_local = rotate_vector_by_quaternion(
                normalize_vector(sample_ray_from_angles_and_vector(beta * light_angle, gamma, light_source)),
                rotation_to_z);

            if (light_sample < 0.5f && S_local.z >= 0.0f) {
                H_local.x = S_local.x + V_local.x;
                H_local.y = S_local.y + V_local.y;
                H_local.z = S_local.z + V_local.z;

                H_local = normalize_vector(H_local);

                weight *= 2.0f * light_angle * device_scene.lights_length;
            } else {
                if (alpha == 0.0f) {
                    H_local.x = 0.0f;
                    H_local.y = 0.0f;
                    H_local.z = 1.0f;
                } else {
                    H_local = sample_GGX_VNDF(V_local, alpha, curand_uniform(random), curand_uniform(random));
                }

                weight *= ((S_local.z >= 0.0f) ? 2.0f : 1.0f);
            }

            const vec3 ray_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

            const float HdotR = fmaxf(eps, fminf(1.0f, dot_product(H_local, ray_local)));
            const float NdotR = fmaxf(eps, fminf(1.0f, ray_local.z));
            const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));
            const float NdotH = fmaxf(eps, fminf(1.0f, H_local.z));

            vec3 specular_f0;
            specular_f0.x = lerp(0.04f, albedo.r, metallic);
            specular_f0.y = lerp(0.04f, albedo.g, metallic);
            specular_f0.z = lerp(0.04f, albedo.b, metallic);

            const vec3 F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotR);

            weight *= 2.0f * Smith_G2_over_G1_height_correlated(alpha, alpha * alpha, NdotR, NdotV) / specular_probability;

            record.r *= F.x;
            record.g *= F.y;
            record.b *= F.z;

            ray = normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));
        }
        else
        {
            record.r *= albedo.r;
            record.g *= albedo.g;
            record.b *= albedo.b;

            const float alpha = acosf(curand_uniform(random));
            const float gamma = 2.0f * 3.1415926535f * curand_uniform(random);

            ray = normalize_vector(sample_ray_from_angles_and_vector(alpha * light_angle, gamma, light_source));

            const float light_feasible = dot_product(ray, normal);

            if (light_sample < 0.5f && light_feasible >= 0.0f) {
                weight *= 2.0f * light_angle * device_scene.lights_length;
            } else {
                ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

                weight *= ((light_feasible >= 0.0f) ? 2.0f : 1.0f);
            }

            const float angle = fmaxf(eps, fminf(dot_product(normal, ray),1.0f));
            const float previous_angle = fmaxf(eps, fminf(dot_product(V, normal),1.0f));

            vec3 H;
            H.x = V.x + ray.x;
            H.y = V.y + ray.y;
            H.z = V.z + ray.z;
            H = normalize_vector(H);

            const float half_angle = fmaxf(eps, fminf(dot_product(H,ray),1.0f));
            const float energyFactor = lerp(1.0f, 1.0f/1.51f, roughness);

            const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

            const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
            const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

            weight *= 2.0f * FDL * FDV * energyFactor * (1.0f - metallic) / (1.0f - specular_probability);
        }
    }

    return result;
}

__global__
void trace_rays(volatile uint32_t* progress, int offset_x, int offset_y, int size_x, int size_y, int limit) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    while (id < device_amount && id < limit) {
        int x = offset_x + id % size_x;
        int y = offset_y + id / size_x;

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

            RGBF color = trace_ray_iterative(device_scene.camera.pos, ray, &random, x + y * device_width);

            if (isnan(color.r) || isinf(color.r)) {
                color.r = 0.0f;
            }
            if (isnan(color.g) || isinf(color.g)) {
                color.g = 0.0f;
            }
            if (isnan(color.b) || isinf(color.b)) {
                color.b = 0.0f;
            }

            result.r += color.r;
            result.g += color.g;
            result.b += color.b;
        }

        float weight = 1.0f/(float)device_diffuse_samples;

        result.r *= weight;
        result.g *= weight;
        result.b *= weight;

        if (device_denoiser) {
            RGBF sum = device_albedo_buffer[x + y * device_width];
            sum.r *= weight;
            sum.g *= weight;
            sum.b *= weight;
            device_albedo_buffer[x + y * device_width] = sum;
        }

        device_frame[x + y * device_width] = result;

        id += blockDim.x * gridDim.x;

        atomicAdd((uint32_t*)progress, 1);
        __threadfence_system();
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

    device_camera_space = inverse_quaternion(q);

    curand_init(0,0,0,&device_random);

    device_sun.x = sinf(device_scene.azimuth) * cosf(device_scene.altitude);
    device_sun.y = sinf(device_scene.altitude);
    device_sun.z = cosf(device_scene.azimuth) * cosf(device_scene.altitude);
    device_sun = normalize_vector(device_sun);

    device_scene.lights[0].pos = scale_vector(device_sun, 149630000000.0f);
}


extern "C" raytrace_instance* init_raytracing(
    const unsigned int width, const unsigned int height, const int reflection_depth,
    const int diffuse_samples, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas,
    int illuminance_atlas_length, void* material_atlas, int material_atlas_length, Scene scene, int denoiser) {
    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)_mm_malloc(sizeof(RGBF) * width * height, 32);

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
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.nodes), sizeof(Node) * instance->scene_gpu.nodes_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.lights), sizeof(Light) * instance->scene_gpu.lights_length));

    gpuErrchk(cudaMemcpy(instance->scene_gpu.texture_assignments, scene.texture_assignments, sizeof(texture_assignment) * scene.materials_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.triangles, scene.triangles, sizeof(Triangle) * scene.triangles_length, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(instance->scene_gpu.nodes, scene.nodes, sizeof(Node) * scene.nodes_length, cudaMemcpyHostToDevice));
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

static float linearRGB_to_SRGB(const float value) {
    if (value <= 0.0031308f) {
      return 12.92f * value;
    }
    else {
      return 1.055f * powf(value, 0.416666666667f) - 0.055f;
    }
  }

extern "C" void trace_scene(Scene scene, raytrace_instance* instance, const int progress) {
    volatile uint32_t *progress_gpu, *progress_cpu;

    cudaEvent_t kernelFinished;
    gpuErrchk(cudaEventCreate(&kernelFinished));
    gpuErrchk(cudaHostAlloc((void**)&progress_cpu, sizeof(uint32_t), cudaHostAllocMapped));
    gpuErrchk(cudaHostGetDevicePointer((uint32_t**)&progress_gpu, (uint32_t*)progress_cpu, 0));
    *progress_cpu = 0;

    gpuErrchk(cudaMemcpyToSymbol(device_scene, &(instance->scene_gpu), sizeof(Scene), 0, cudaMemcpyHostToDevice));

    set_up_raytracing_device<<<1,1>>>();

    gpuErrchk(cudaDeviceSynchronize());

    clock_t t = clock();



    if (progress == 2) {
        const unsigned int max_block_width = 128;
        const unsigned int max_block_height = 128;


        SDL_Init(SDL_INIT_VIDEO);
        SDL_Window* window = SDL_CreateWindow(
          "Luminary", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, instance->width, instance->height,
          SDL_WINDOW_SHOWN);

        SDL_Surface* window_surface = SDL_GetWindowSurface(window);

        Uint32 rmask, gmask, bmask, amask;

      #if SDL_BYTEORDER == SDL_BIG_ENDIAN
        rmask = 0xff000000;
        gmask = 0x00ff0000;
        bmask = 0x0000ff00;
        amask = 0x000000ff;
      #else
        rmask = 0x000000ff;
        gmask = 0x0000ff00;
        bmask = 0x00ff0000;
        amask = 0xff000000;
      #endif

        SDL_Surface* surface =
          SDL_CreateRGBSurface(0, instance->width, instance->height, 24, rmask, gmask, bmask, amask);

        RGB8* buffer = (RGB8*) surface->pixels;

        int exit = 0;

        char* title = (char*) malloc(4096);

        unsigned int offset_x = 0;
        unsigned int offset_y = 0;
        unsigned int block_width = 0;
        unsigned int block_height = 0;

        unsigned int block_counter = 0;
        const unsigned int blocks_per_row = (instance->width / max_block_width) + 1;
        const unsigned int total_blocks = blocks_per_row * ((instance->height / max_block_height) + 1);

        while (!exit && block_counter < total_blocks) {
          SDL_Event event;

          for (unsigned int i = 0; i < block_height; i++) {
            gpuErrchk(cudaMemcpy(
                instance->frame_buffer + offset_x + (offset_y + i) * instance->width,
                instance->frame_buffer_gpu + offset_x + (offset_y + i) * instance->width,
                sizeof(RGBF) * block_width,
                cudaMemcpyDeviceToHost));
          }

          const unsigned int block_x = block_counter % blocks_per_row;
          const unsigned int block_y = block_counter / blocks_per_row;
          const unsigned int new_offset_x = block_x * max_block_width;
          const unsigned int new_offset_y = block_y * max_block_height;

          const unsigned int new_block_width = (new_offset_x + max_block_width > instance->width) ? instance->width - new_offset_x : max_block_width;
          const unsigned int new_block_height = (new_offset_y + max_block_height > instance->height) ? instance->height - new_offset_y : max_block_height;

          trace_rays<<<blocks_per_grid,threads_per_block>>>(progress_gpu, new_offset_x , new_offset_y, new_block_width, new_block_height, new_block_width * new_block_height);

          for (unsigned int j = new_offset_y; j < new_offset_y + new_block_height; j++) {
            for (unsigned int i = new_offset_x; i < new_offset_x + new_block_width; i++) {
              RGB8 pixel;
              pixel.r = 255;
              pixel.g = 255;
              pixel.b = 255;

              buffer[i + instance->width * j] = pixel;
            }
          }

          for (unsigned int j = offset_y; j < offset_y + block_height; j++) {
              for (unsigned int i = offset_x; i < offset_x + block_width; i++) {
                RGB8 pixel;
                RGBF pixel_float = instance->frame_buffer[i + instance->width * j];

                RGBF color;
                color.r = min(255.9f, linearRGB_to_SRGB(pixel_float.r) * 255.9f);
                color.g = min(255.9f, linearRGB_to_SRGB(pixel_float.g) * 255.9f);
                color.b = min(255.9f, linearRGB_to_SRGB(pixel_float.b) * 255.9f);

                pixel.r = (uint8_t) color.r;
                pixel.g = (uint8_t) color.g;
                pixel.b = (uint8_t) color.b;

                buffer[i + instance->width * j] = pixel;
              }
          }

          SDL_BlitSurface(surface, 0, window_surface, 0);

          clock_t curr_time = clock();
          const double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
          sprintf(title, "Luminary - Progress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs", (((float)block_counter) / total_blocks) * 100.0f, time_elapsed, (total_blocks - block_counter) * (time_elapsed/block_counter));

          SDL_SetWindowTitle(window, title);
          SDL_UpdateWindowSurface(window);

          while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
              exit = 1;
            }
          }

          offset_x = new_offset_x;
          offset_y = new_offset_y;
          block_width = new_block_width;
          block_height = new_block_height;

          block_counter++;
        }

        free(title);

        SDL_DestroyWindow(window);
        SDL_Quit();
    } else if (progress == 1) {
        trace_rays<<<blocks_per_grid,threads_per_block>>>(progress_gpu, 0, 0, instance->width, instance->height, instance->width * instance->height);

        gpuErrchk(cudaEventRecord(kernelFinished));

        uint32_t progress = 0;
        const uint32_t total_pixels = instance->width * instance->height;
        const float ratio = 1.0f/((float)instance->width * (float)instance->height);
        while (cudaEventQuery(kernelFinished) != cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::microseconds(100000));
            uint32_t new_progress = *progress_cpu;
            if (new_progress > progress) {
                progress = new_progress;
            }
            clock_t curr_time = clock();
            const double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
            printf("\r                                                                                         \rProgress: %2.1f%% - Time Elapsed: %.1fs - Time Remaining: %.1fs",(float)progress * ratio * 100, time_elapsed, (total_pixels - progress) * (time_elapsed/progress));
        }

        printf("\r                                                                             \r");
    }



    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(instance->frame_buffer, instance->frame_buffer_gpu, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost));
}

extern "C" void free_raytracing(raytrace_instance* instance) {
    gpuErrchk(cudaFree(instance->frame_buffer_gpu));

    gpuErrchk(cudaFree(instance->scene_gpu.texture_assignments));
    gpuErrchk(cudaFree(instance->scene_gpu.triangles));
    gpuErrchk(cudaFree(instance->scene_gpu.nodes));
    gpuErrchk(cudaFree(instance->scene_gpu.lights));

    _mm_free(instance->frame_buffer);

    if (instance->denoiser) {
        gpuErrchk(cudaFree(instance->albedo_buffer_gpu));
    }

    free(instance);
}
