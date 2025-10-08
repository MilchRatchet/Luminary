#ifndef CU_LUMINARY_OPTIX_COMPILE_DEFINES_H
#define CU_LUMINARY_OPTIX_COMPILE_DEFINES_H

#define OPTIX_COMPILATION_DEFINITION_FOUND false

#ifdef optix_kernel_raytrace
#define OPTIX_KERNEL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_geometry_geo
#define OPTIX_KERNEL
#define OPTIX_ENABLE_GEOMETRY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_geometry_sky
#define OPTIX_KERNEL
#define OPTIX_ENABLE_SKY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_volume_geo
#define OPTIX_KERNEL
#define OPTIX_ENABLE_GEOMETRY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_volume_sky
#define OPTIX_KERNEL
#define OPTIX_ENABLE_SKY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_particles_geo
#define OPTIX_KERNEL
#define OPTIX_ENABLE_GEOMETRY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shading_particles_sky
#define OPTIX_KERNEL
#define OPTIX_ENABLE_SKY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_shadow
#define OPTIX_KERNEL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

#ifdef optix_kernel_dummy_intellisense
#define OPTIX_KERNEL
#define OPTIX_ENABLE_GEOMETRY_DL
#define OPTIX_ENABLE_SKY_DL
#undef OPTIX_COMPILATION_DEFINITION_FOUND
#define OPTIX_COMPILATION_DEFINITION_FOUND true
#endif

static_assert(OPTIX_COMPILATION_DEFINITION_FOUND, "Unknown OptiX translation unit");

#endif /* CU_LUMINARY_OPTIX_COMPILE_DEFINES_H */
