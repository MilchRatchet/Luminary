#include <stdint.h>
#include "scene.h"

#if __cplusplus
extern "C" {
#endif
uint8_t* scene_to_frame(Scene scene, const unsigned int width, const unsigned int height);
#if __cplusplus
}
#endif
