#ifndef SKY_BAKE_H
#define SKY_BAKE_H

#include "structs.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void sky_bake_generate_LUTs(RaytraceInstance* instance);
void sky_bake_hdri_generate_LUT(RaytraceInstance* instance);
void sky_bake_hdri_set_pos_to_cam(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* SKY_BAKE_H */
