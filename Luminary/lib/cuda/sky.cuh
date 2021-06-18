#ifndef CU_SKY_H
#define CU_SKY_H

#include "utils.cuh"
#include "math.cuh"
#include <cuda_runtime_api.h>

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

    point = add_vector(point, scale_vector(ray, 0.125f * step_size));

    for (int i = 0; i < steps; i++) {
        depth += density_at_height(height_at_point(point),0.125f) * step_size;

        point = add_vector(point, scale_vector(ray, step_size));
    }

    return depth;
}

__device__
RGBF get_sky_color(const vec3 ray) {
    RGBF result = get_color(0.0f, 0.0f, 0.0f);;

    if (ray.y < 0.0f) {
        return result;
    }

    const float angular_diameter = 0.009f;

    const float overall_density = device_scene.sky.base_density;

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
    sun_color.g = 1.0f * sun_intensity;
    sun_color.b = 1.0f * sun_intensity;

    const vec3 sun_normalized = device_sun;
    const vec3 sun = scale_vector(sun_normalized, sun_dist);

    const float earth_radius = 6371.0f;
    const float atmosphere_height = 100.0f;

    vec3 origin = get_vector(0.0f, earth_radius, 0.0f);

    const vec3 origin_default = origin;

    const float limit = get_length_to_border(origin, ray, earth_radius + atmosphere_height);
    const int steps = 8;
    const float step_size = limit/steps;
    float reach = 0.0f;

    reach += step_size * 0.125f;

    origin = add_vector(origin, scale_vector(ray, 0.125f * step_size));

    for (int i = 0; i < steps; i++) {
        const vec3 ray_scatter = normalize_vector(sub_vector(sun, origin));

        const float optical_depth = get_optical_depth(origin_default, ray, reach) + get_optical_depth(origin, ray_scatter, get_length_to_border(origin, ray_scatter, earth_radius + atmosphere_height));

        const float height = height_at_point(origin);

        const float local_density = density_at_height(height, device_scene.sky.rayleigh_falloff);
        const float mie_density = density_at_height(height, device_scene.sky.mie_falloff);
        //The tent function is disabled atm, first argument 0.0f to activate
        const float ozone_density = fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);

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

        origin = add_vector(origin, scale_vector(ray, step_size));
    }
    const vec3 ray_sun = normalize_vector(sub_vector(sun, origin_default));

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

#endif /* CU_SKY_H */
