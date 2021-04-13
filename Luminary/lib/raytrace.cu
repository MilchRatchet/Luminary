#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include "mesh.h"
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int threads_per_block = 128;
const int blocks_per_grid = 512;

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
cudaTextureObject_t* device_albedo_atlas;

__device__
cudaTextureObject_t* device_illuminance_atlas;

__device__
cudaTextureObject_t* device_material_atlas;

__device__
texture_assignment* device_texture_assignments;

__device__
vec3 device_sun;

//---------------------------------
// Vector Utilities
//---------------------------------

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
float lerp(const float a, const float b, const float t) {
    return a + t * (b - a);
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
vec3 reflect_vector(const vec3 v, const vec3 n) {
    vec3 result;

    const float dot = dot_product(v, n);

    result.x = v.x - 2.0f * dot * n.x;
    result.y = v.y - 2.0f * dot * n.y;
    result.z = v.z - 2.0f * dot * n.z;

    return result;
}

__device__
vec3 scale_vector(vec3 vector, const float scale) {
    vector.x *= scale;
    vector.y *= scale;
    vector.z *= scale;

    return vector;
}

__device__
float get_length(const vec3 vector) {
    return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

__device__
vec3 normalize_vector(vec3 vector) {
    const float inv_length = 1.0f / get_length(vector);

    return scale_vector(vector, inv_length);
}

__device__
vec3 Fresnel_Schlick(const vec3 f0, const float f90, const float NdotS) {
    vec3 result;

    const float t = powf(1.0f - NdotS, 5.0f);

    result.x = lerp(f0.x, f90, t);
    result.y = lerp(f0.y, f90, t);
    result.z = lerp(f0.z, f90, t);

    return result;
}

__device__
float luminance(const vec3 v) {
    vec3 w;
    w.x = 0.2126f;
    w.y = 0.7152f;
    w.z = 0.0722f;

    return dot_product(v,w);
}

__device__
float shadowed_F90(const vec3 f0) {
    const float t = 1.0f / 0.04f;
    return fminf(1.0f, t * luminance(f0));
}

__device__
vec3 get_coordinates_in_triangle(const Triangle triangle, const vec3 point) {
    const vec3 diff = vec_diff(point, triangle.vertex);
    const float d00 = dot_product(triangle.edge1,triangle.edge1);
    const float d01 = dot_product(triangle.edge1,triangle.edge2);
    const float d11 = dot_product(triangle.edge2,triangle.edge2);
    const float d20 = dot_product(diff,triangle.edge1);
    const float d21 = dot_product(diff,triangle.edge2);
    const float denom = 1.0f / (d00 * d11 - d01 * d01);
    vec3 result;
    result.x = (d11 * d20 - d01 * d21) * denom;
    result.y = (d00 * d21 - d01 * d20) * denom;
    return result;
}

__device__
vec3 lerp_normals(const Triangle triangle, const float lambda, const float mu) {
    vec3 result;

    result.x = triangle.vertex_normal.x + lambda * triangle.edge1_normal.x + mu * triangle.edge2_normal.x;
    result.y = triangle.vertex_normal.y + lambda * triangle.edge1_normal.y + mu * triangle.edge2_normal.y;
    result.z = triangle.vertex_normal.z + lambda * triangle.edge1_normal.z + mu * triangle.edge2_normal.z;

    return normalize_vector(result);
}

__device__
UV lerp_uv(const Triangle triangle, const float lambda, const float mu) {
    UV result;

    result.u = triangle.vertex_texture.u + lambda * triangle.edge1_texture.u + mu * triangle.edge2_texture.u;
    result.v = triangle.vertex_texture.v + lambda * triangle.edge1_texture.v + mu * triangle.edge2_texture.v;

    return result;
}

__device__
float ray_box_intersect(const vec3 low, const vec3 high, vec3 origin, const vec3 ray) {
    origin.x -= (high.x + low.x) * 0.5f;
    origin.y -= (high.y + low.y) * 0.5f;
    origin.z -= (high.z + low.z) * 0.5f;

    const float size_x = (high.x - low.x) * 0.5f;
    const float size_y = (high.y - low.y) * 0.5f;
    const float size_z = (high.z - low.z) * 0.5f;

    if ((fabsf(origin.x) < size_x) && (fabsf(origin.y) < size_y) && (fabsf(origin.z) < size_z)) return 0.0f;

    vec3 d;

    vec3 sign;

    sign.x = copysignf(1.0f, -ray.x);
    sign.y = copysignf(1.0f, -ray.y);
    sign.z = copysignf(1.0f, -ray.z);

    d.x = size_x * sign.x - origin.x;
    d.y = size_y * sign.y - origin.y;
    d.z = size_z * sign.z - origin.z;

    d.x /= ray.x;
    d.y /= ray.y;
    d.z /= ray.z;

    const bool test_x = (d.x >= 0.0f) && (fabsf(origin.y + ray.y * d.x) < size_y) && (fabsf(origin.z + ray.z * d.x) < size_z);
    const bool test_y = (d.y >= 0.0f) && (fabsf(origin.x + ray.x * d.y) < size_x) && (fabsf(origin.z + ray.z * d.y) < size_z);
    const bool test_z = (d.z >= 0.0f) && (fabsf(origin.x + ray.x * d.z) < size_x) && (fabsf(origin.y + ray.y * d.z) < size_y);

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
}

__device__
float triangle_intersection(const Triangle triangle, const vec3 origin, const vec3 ray) {
    const vec3 h = cross_product(ray, triangle.edge2);
    const float a = dot_product(triangle.edge1, h);

    if (a > -0.00000001 && a < 0.00000001) return FLT_MAX;

    const float f = 1.0f / a;
    const vec3 s = vec_diff(origin, triangle.vertex);
    const float u = f * dot_product(s, h);

    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    const vec3 q = cross_product(s, triangle.edge1);
    const float v = f * dot_product(ray, q);

    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    const float t = f * dot_product(triangle.edge2, q);

    if (t > -epsilon) {
        return t;
    } else {
        return FLT_MAX;
    }
}

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

__device__
int trailing_zeros(const unsigned int n) {
    int mask = 1;
    for (int i = 0; i < 32; i++, mask <<= 1)
        if ((n & mask) != 0)
            return i;

    return 32;
}

__device__
vec3 decompress_vector(const compressed_vec3 vector, const vec3 p, const float ex, const float ey, const float ez) {
    vec3 result;

    result.x = p.x + ex * (float)vector.x;
    result.y = p.y + ey * (float)vector.y;
    result.z = p.z + ez * (float)vector.z;

    return result;
}

//---------------------------------
// Quaternion Utilities
//---------------------------------

__device__
Quaternion inverse_quaternion(const Quaternion q) {
    Quaternion result;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;
    result.w = q.w;
    return result;
}

__device__
Quaternion get_rotation_to_z_canonical(const vec3 v) {
    Quaternion res;
    if (v.z < -1.0f + epsilon) {
        res.x = 1.0f;
        res.y = 0.0f;
        res.z = 0.0f;
        res.w = 0.0f;
        return res;
    }
    res.x = v.y;
    res.y = -v.x;
    res.z = 0.0f;
    res.w = 1.0f + v.z;

    const float norm = 1.0f / sqrtf(res.x * res.x + res.y * res.y + res.w * res.w);

    res.x *= norm;
    res.y *= norm;
    res.w *= norm;

    return res;
}

__device__
vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
    vec3 result;

    vec3 u;
    u.x = q.x;
    u.y = q.y;
    u.z = q.z;

    const float s = q.w;

    const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
    const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

    vec3 cross;

    cross.x = u.y*v.z - u.z*v.y;
    cross.y = u.z*v.x - u.x*v.z;
    cross.z = u.x*v.y - u.y*v.x;

    result.x = 2.0f * dot_uv * u.x + ((s*s)-dot_uu) * v.x + 2.0f * s * cross.x;
    result.y = 2.0f * dot_uv * u.y + ((s*s)-dot_uu) * v.y + 2.0f * s * cross.y;
    result.z = 2.0f * dot_uv * u.z + ((s*s)-dot_uu) * v.z + 2.0f * s * cross.z;

    return result;
}

//---------------------------------
// Procedural Sky
//---------------------------------

__device__
float get_length_to_border(const vec3 origin, vec3 ray, const float atmosphere_end) {
    if (ray.y < 0.0f) ray = scale_vector(ray, -1.0f);
    const float a = dot_product(origin,ray);
    return -a + sqrtf(a * a - dot_product(origin,origin) + atmosphere_end * atmosphere_end);
}

__device__
float density_at_height(const float height, const float density_falloff) {
    return expf(-height * density_falloff);
}

__device__
float height_at_point(const vec3 point) {
    const float earth_radius = 6371.0f;
    return (get_length(point) - earth_radius);
}


__device__
float get_optical_depth(const vec3 origin, const vec3 ray, const float length) {
    if (length == 0.0f) return 0.0f;

    const int steps = 8;
    const float step_size = length / steps;
    float depth = 0.0f;
    vec3 point = origin;

    point.x += step_size * ray.x * 0.125f;
    point.y += step_size * ray.y * 0.125f;
    point.z += step_size * ray.z * 0.125f;

    for (int i = 0; i < steps; i++) {
        depth += density_at_height(height_at_point(point),0.125f) * step_size;

        point.x += step_size * ray.x;
        point.y += step_size * ray.y;
        point.z += step_size * ray.z;
    }

    return depth;
}

__device__
RGBF get_sky_color(const vec3 ray) {
    RGBF result;
    result.r = 0.0f;
    result.g = 0.0f;
    result.b = 0.0f;

    if (ray.y < 0.0f) {
        return result;
    }

    const float angular_diameter = 0.009f;

    const float overall_density = 1.0f;

    RGBF scatter;
    scatter.r = 5.8f * 0.001f * overall_density;
    scatter.g = 13.558f * 0.001f * overall_density;
    scatter.b = 33.1f * 0.001f * overall_density;

    const float mie_scatter = 3.996f * 0.001f * overall_density;

    RGBF ozone_absorbtion;
    ozone_absorbtion.r = 0.65f * 0.001f * overall_density;
    ozone_absorbtion.g = 1.881f * 0.001f * overall_density;
    ozone_absorbtion.b = 0.085f * 0.001f * overall_density;

    const float sun_dist = 150000000.0f;

    RGBF sun_color;

    const float sun_intensity = 6.0f;

    sun_color.r = 1.0f * sun_intensity;
    sun_color.g = 0.9f * sun_intensity;
    sun_color.b = 0.8f * sun_intensity;

    const vec3 sun_normalized = device_sun;
    const vec3 sun = scale_vector(sun_normalized, sun_dist);

    const float earth_radius = 6371.0f;
    const float atmosphere_height = 100.0f;

    vec3 origin;
    origin.x = 0.0f;
    origin.y = earth_radius + 0.0f;
    origin.z = 0.0f;

    const vec3 origin_default = origin;

    const float limit = get_length_to_border(origin, ray, earth_radius + atmosphere_height);
    const int steps = 8;
    const float step_size = limit/steps;
    float reach = 0.0f;

    reach += step_size * 0.125f;

    origin.x += step_size * ray.x * 0.125f;
    origin.y += step_size * ray.y * 0.125f;
    origin.z += step_size * ray.z * 0.125f;

    for (int i = 0; i < steps; i++) {
        const vec3 ray_scatter = normalize_vector(vec_diff(sun, origin));

        const float optical_depth = get_optical_depth(origin_default, ray, reach) + get_optical_depth(origin, ray_scatter, get_length_to_border(origin, ray_scatter, earth_radius + atmosphere_height));

        const float height = height_at_point(origin);

        const float local_density = density_at_height(height, 0.125f);
        const float mie_density = density_at_height(height, 0.83333f);
        //The tent function is disabled atm, first argument 0.0f to activate
        const float ozone_density = fmaxf(1.0f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);

        RGBF transmittance;
        transmittance.r = expf(-optical_depth * (scatter.r + ozone_density * ozone_absorbtion.r + 1.11f * mie_scatter));
        transmittance.g = expf(-optical_depth * (scatter.g + ozone_density * ozone_absorbtion.g + 1.11f * mie_scatter));
        transmittance.b = expf(-optical_depth * (scatter.b + ozone_density * ozone_absorbtion.b + 1.11f * mie_scatter));

        float cos_angle = dot_product(ray, ray_scatter);

        cos_angle = cosf(fmaxf(0.0f,acosf(cos_angle) - angular_diameter));

        const float rayleigh = 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);

        const float g = 0.8f;
        const float mie = 1.5f * (1.0f + cos_angle * cos_angle) * (1.0f - g * g) / (4.0f * 3.1415926535f * (2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * cos_angle, 1.5f));

        result.r += sun_color.r * transmittance.r * (local_density * scatter.r * rayleigh + mie_density * mie_scatter * mie) * step_size;
        result.g += sun_color.g * transmittance.g * (local_density * scatter.g * rayleigh + mie_density * mie_scatter * mie) * step_size;
        result.b += sun_color.b * transmittance.b * (local_density * scatter.b * rayleigh + mie_density * mie_scatter * mie) * step_size;

        reach += step_size;

        origin.x += step_size * ray.x;
        origin.y += step_size * ray.y;
        origin.z += step_size * ray.z;
    }
    const vec3 ray_sun = normalize_vector(vec_diff(sun, origin_default));

    float cos_angle = dot_product(ray, ray_sun);
    cos_angle = cosf(fmaxf(0.0f,acosf(cos_angle) - angular_diameter));

    if (cos_angle >= 0.99999f) {
        const float optical_depth = get_optical_depth(origin_default, ray, limit);

        const float height = height_at_point(origin_default);

        const float ozone_density = fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);

        RGBF transmittance;
        transmittance.r = expf(-optical_depth * (scatter.r + ozone_density * ozone_absorbtion.r + 1.11f * mie_scatter));
        transmittance.g = expf(-optical_depth * (scatter.g + ozone_density * ozone_absorbtion.g + 1.11f * mie_scatter));
        transmittance.b = expf(-optical_depth * (scatter.b + ozone_density * ozone_absorbtion.b + 1.11f * mie_scatter));

        result.r += sun_color.r * transmittance.r * cos_angle * device_scene.sun_strength;
        result.g += sun_color.g * transmittance.g * cos_angle * device_scene.sun_strength;
        result.b += sun_color.b * transmittance.b * cos_angle * device_scene.sun_strength;
    }

    return result;
}

//---------------------------------
// Microfacet BRDF
//---------------------------------

__device__
float Smith_G1_GGX(const float alpha2, const float NdotS2) {
    return 2.0f / (sqrtf(((alpha2 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__
float Smith_G2_over_G1_height_correlated(const float alpha, const float alpha2, const float NdotL, const float NdotV) {
    const float G1V = Smith_G1_GGX(alpha2, NdotV * NdotV);
    const float G1L = Smith_G1_GGX(alpha2, NdotL * NdotL);
    return G1L / (G1V + G1L - G1V * G1L);
}

__device__
vec3 sample_GGX_VNDF(const vec3 v, const float alpha, const float random1, const float random2) {
    vec3 v_hemi;

    v_hemi.x = alpha * v.x;
    v_hemi.y = alpha * v.y;
    v_hemi.z = v.z;

    const float length_squared = v_hemi.x * v_hemi.x + v_hemi.y * v_hemi.y;
    vec3 T1;

    if (length_squared == 0.0f) {
        T1.x = 1.0f;
        T1.y = 0.0f;
        T1.z = 0.0f;
    } else {
        const float length = 1.0f / sqrtf(length_squared);
        T1.x = -v_hemi.y * length;
        T1.y = v_hemi.x * length;
        T1.z = 0.0f;
    }

    const vec3 T2 = cross_product(v_hemi, T1);

    const float r = sqrtf(random1);
    const float phi = 2.0f * PI * random2;
    const float t1 = r * cosf(phi);
    const float s = 0.5f * (1.0f + v_hemi.z);
    const float t2 = lerp(sqrtf(1.0f - t1 * t1), r * sinf(phi), s);

    vec3 normal_hemi;

    const float scalar = sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
    normal_hemi.x = alpha * (t1 * T1.x + t2 * T2.x + scalar * v_hemi.x);
    normal_hemi.y = alpha * (t1 * T1.y + t2 * T2.y + scalar * v_hemi.y);
    normal_hemi.z = fmaxf(0.0f, t1 * T1.z + t2 * T2.z + scalar * v_hemi.z);

    return normalize_vector(normal_hemi);
}

//---------------------------------
// Path Tracing
//---------------------------------

__device__
float get_light_angle(Light light, vec3 pos) {
    const float d = get_length(vec_diff(pos, light.pos)) + epsilon;
    return atanf(light.radius / d);
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

    for (int reflection_number = 0; reflection_number < device_reflection_depth; reflection_number++) {
        float depth = device_scene.far_clip_distance;

        unsigned int hit_id = 0xffffffff;

        vec3 curr;

        int node_address = 0;
        int node_key = 1;
        int bit_trail = 0;
        int mrpn_address = -1;

        while (node_address != -1) {
            while (device_scene.nodes[node_address].triangles_address == -1) {
                Node node = device_scene.nodes[node_address];

                const float decompression_x = __powf(2.0f, (float)node.ex);
                const float decompression_y = __powf(2.0f, (float)node.ey);
                const float decompression_z = __powf(2.0f, (float)node.ez);

                const vec3 left_high = decompress_vector(node.left_high, node.p, decompression_x, decompression_y, decompression_z);
                const vec3 left_low = decompress_vector(node.left_low, node.p, decompression_x, decompression_y, decompression_z);
                const vec3 right_high = decompress_vector(node.right_high, node.p, decompression_x, decompression_y, decompression_z);
                const vec3 right_low = decompress_vector(node.right_low, node.p, decompression_x, decompression_y, decompression_z);

                const float L = ray_box_intersect(left_low, left_high, origin, ray);
                const float R = ray_box_intersect(right_low, right_high, origin, ray);
                const int R_is_closest = R < L;

                if (L < depth || R < depth) {

                    node_key = node_key << 1;
                    bit_trail = bit_trail << 1;

                    if (L >= depth || R_is_closest) {
                        node_address = 2 * node_address + 2;
                        node_key = node_key ^ 0b1;
                    }
                    else {
                        node_address = 2 * node_address + 1;
                    }

                    if (L < depth && R < depth) {
                        bit_trail = bit_trail ^ 0b1;
                        if (R_is_closest) {
                            mrpn_address = node_address - 1;
                        }
                        else {
                            mrpn_address = node_address + 1;
                        }
                    }
                } else {
                    break;
                }
            }

            if (device_scene.nodes[node_address].triangles_address != -1) {
                Node node = device_scene.nodes[node_address];

                for (unsigned int i = 0; i < node.triangle_count; i++) {
                    const float d = triangle_intersection(device_scene.triangles[node.triangles_address + i], origin, ray);

                    if (d < depth) {
                        depth = d;
                        hit_id = node.triangles_address + i;
                    }
                }
            }

            if (bit_trail == 0) {
                node_address = -1;
            }
            else {
                int num_levels = trailing_zeros(bit_trail);
                bit_trail = (bit_trail >> num_levels) ^ 0b1;
                node_key = (node_key >> num_levels) ^ 0b1;
                if (mrpn_address != -1) {
                    node_address = mrpn_address;
                    mrpn_address = -1;
                }
                else {
                    node_address = node_key - 1;
                }
            }
        }

        if (hit_id == 0xffffffff) {
            RGBF sky = get_sky_color(ray);

            result.r += sky.r * weight * record.r;
            result.g += sky.g * weight * record.g;
            result.b += sky.b * weight * record.b;

            return result;
        }

        curr.x = origin.x + ray.x * depth;
        curr.y = origin.y + ray.y * depth;
        curr.z = origin.z + ray.z * depth;

        Triangle hit_triangle = device_scene.triangles[hit_id];

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
            origin.x = curr.x + 2.0f * epsilon * ray.x;
            origin.y = curr.y + 2.0f * epsilon * ray.y;
            origin.z = curr.z + 2.0f * epsilon * ray.z;

            continue;
        }

        const float specular = curand_uniform(random);
        const float specular_probability = lerp(0.5f, 1.0f - epsilon, metallic);

        const vec3 V = scale_vector(ray, -1.0f);

        if (dot_product(normal, V) <= 0.0f) {
            normal = scale_vector(normal, -1.0f);
        }

        curr.x += normal.x * epsilon * 2.0f;
        curr.y += normal.y * epsilon * 2.0f;
        curr.z += normal.z * epsilon * 2.0f;

        origin.x = curr.x;
        origin.y = curr.y;
        origin.z = curr.z;

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

            const float HdotR = fmaxf(epsilon, fminf(1.0f, dot_product(H_local, ray_local)));
            const float NdotR = fmaxf(epsilon, fminf(1.0f, ray_local.z));
            const float NdotV = fmaxf(epsilon, fminf(1.0f, V_local.z));
            const float NdotH = fmaxf(epsilon, fminf(1.0f, H_local.z));

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

            const float angle = fmaxf(epsilon, fminf(dot_product(normal, ray),1.0f));
            const float previous_angle = fmaxf(epsilon, fminf(dot_product(V, normal),1.0f));

            vec3 H;
            H.x = V.x + ray.x;
            H.y = V.y + ray.y;
            H.z = V.z + ray.z;
            H = normalize_vector(H);

            const float half_angle = fmaxf(epsilon, fminf(dot_product(H,ray),1.0f));
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
void trace_rays(volatile uint32_t* progress) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    while (id < device_amount) {
        int x = id % device_width;
        int y = id / device_width;

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

        device_frame[id] = result;

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

    curand_init(0,0,0,&device_random);

    device_sun.x = sinf(device_scene.azimuth)*cosf(device_scene.altitude);
    device_sun.y = sinf(device_scene.altitude);
    device_sun.z = cosf(device_scene.azimuth)*cosf(device_scene.altitude);
    device_sun = normalize_vector(device_sun);

    device_scene.lights[0].pos = scale_vector(device_sun, 149630000000.0f);
}


extern "C" raytrace_instance* init_raytracing(
    const unsigned int width, const unsigned int height, const int reflection_depth,
    const int diffuse_samples, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas,
    int illuminance_atlas_length, void* material_atlas, int material_atlas_length, Scene scene) {
    raytrace_instance* instance = (raytrace_instance*)malloc(sizeof(raytrace_instance));

    instance->width = width;
    instance->height = height;
    instance->frame_buffer = (RGBF*)_mm_malloc(sizeof(RGBF) * width * height, 32);

    cudaMalloc((void**) &(instance->frame_buffer_gpu), sizeof(RGBF) * width * height);

    instance->reflection_depth = reflection_depth;
    instance->diffuse_samples = diffuse_samples;

    instance->albedo_atlas = albedo_atlas;
    instance->illuminance_atlas = illuminance_atlas;
    instance->material_atlas = material_atlas;

    instance->albedo_atlas_length = albedo_atlas_length;
    instance->illuminance_atlas_length = illuminance_atlas_length;
    instance->material_atlas_length = material_atlas_length;

    instance->scene_gpu = scene;

    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.texture_assignments), sizeof(texture_assignment) * scene.meshes_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.triangles), sizeof(Triangle) * instance->scene_gpu.triangles_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.nodes), sizeof(Node) * instance->scene_gpu.nodes_length));
    gpuErrchk(cudaMalloc((void**) &(instance->scene_gpu.lights), sizeof(Light) * instance->scene_gpu.lights_length));

    gpuErrchk(cudaMemcpy(instance->scene_gpu.texture_assignments, scene.texture_assignments, sizeof(texture_assignment) * scene.meshes_length, cudaMemcpyHostToDevice));
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

    trace_rays<<<blocks_per_grid,threads_per_block>>>(progress_gpu);

    if (progress) {
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
            double time_elapsed = (((double)curr_time - t)/CLOCKS_PER_SEC);
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

    free(instance);
}
