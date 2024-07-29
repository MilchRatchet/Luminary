#ifndef LIGHT_H
#define LIGHT_H

#include "utils.h"

void lights_process(Scene* scene, int dmm_active);
void lights_build_light_tree(Scene* scene);

#endif /* LIGHT_H */
