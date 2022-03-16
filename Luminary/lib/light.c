#include "light.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "error.h"
#include "utils.h"

void process_lights(Scene* scene) {
  bench_tic();

  Scene data = *scene;

  unsigned int lights_ids_length = 16;
  uint32_t* lights_ids           = (uint32_t*) malloc(sizeof(uint32_t) * lights_ids_length);
  unsigned int light_count       = 2;

  lights_ids[0] = LIGHT_ID_SUN;
  lights_ids[1] = LIGHT_ID_TOY;

  for (unsigned int i = 0; i < data.triangles_length; i++) {
    const Triangle triangle = data.triangles[i];

    if (data.texture_assignments[triangle.object_maps].illuminance_map != 0) {
      lights_ids[light_count++] = i;
      if (light_count == lights_ids_length) {
        lights_ids_length *= 2;
        lights_ids = safe_realloc(lights_ids, sizeof(uint32_t) * lights_ids_length);
      }
    }
  }

  lights_ids = safe_realloc(lights_ids, sizeof(uint32_t) * light_count);

  data.lights_ids        = lights_ids;
  data.lights_ids_length = light_count;

  *scene = data;

  bench_toc("Processing Lights");
}
