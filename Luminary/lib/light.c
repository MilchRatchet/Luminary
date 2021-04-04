#include "utils.h"
#include "light.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static float vector_distance(const vec3 a, const vec3 b) {
  const float x = a.x - b.x;
  const float y = a.y - b.y;
  const float z = a.z - b.z;

  return sqrtf(x * x + y * y + z * z);
}

void process_lights(Scene* scene) {
  Scene data = *scene;

  unsigned int lights_length = 16;
  Light* lights              = (Light*) malloc(sizeof(Light) * lights_length);
  unsigned int light_count   = 0;

  for (unsigned int i = 0; i < data.triangles_length; i++) {
    Triangle triangle = data.triangles[i];

    if (data.texture_assignments[triangle.object_maps].illuminance_map != 0) {
      light_count++;
      if (light_count == lights_length) {
        lights_length *= 2;
        lights = realloc(lights, sizeof(Light) * lights_length);
      }

      vec3 vertex2;
      vertex2.x = triangle.vertex.x + triangle.edge1.x;
      vertex2.y = triangle.vertex.y + triangle.edge1.y;
      vertex2.z = triangle.vertex.z + triangle.edge1.z;

      vec3 vertex3;
      vertex3.x = triangle.vertex.x + triangle.edge2.x;
      vertex3.y = triangle.vertex.y + triangle.edge2.y;
      vertex3.z = triangle.vertex.z + triangle.edge2.z;

      float l1 = sqrtf(
        triangle.vertex.x * triangle.vertex.x + triangle.vertex.y * triangle.vertex.y
        + triangle.vertex.z * triangle.vertex.z);

      float l2 = sqrtf(vertex2.x * vertex2.x + vertex2.y * vertex2.y + vertex2.z * vertex2.z);

      float l3 = sqrtf(vertex3.x * vertex3.x + vertex3.y * vertex3.y + vertex3.z * vertex3.z);

      l1 *= l1;
      l2 *= l2;
      l3 *= l3;

      const float w1 = l1 * (l2 + l3 - l1);
      const float w2 = l2 * (l3 + l1 - l2);
      const float w3 = l3 * (l1 + l2 - l3);

      const float inv = 1.0f / (w1 + w2 + w3);

      vec3 middle;
      middle.x = inv * (w1 * triangle.vertex.x + w2 * vertex2.x + w3 * vertex3.x);
      middle.y = inv * (w1 * triangle.vertex.y + w2 * vertex2.y + w3 * vertex3.y);
      middle.z = inv * (w1 * triangle.vertex.z + w2 * vertex2.z + w3 * vertex3.z);

      lights[light_count - 1].pos = middle;

      l1 = vertex2.x - middle.x;
      l2 = vertex2.y - middle.y;
      l3 = vertex2.z - middle.z;

      lights[light_count - 1].radius = sqrtf(l1 * l1 + l2 * l2 + l3 * l3);
    }
  }

  lights = realloc(lights, sizeof(Light) * light_count);

  unsigned int light_groups_length = 16;
  Light* light_groups              = (Light*) malloc(sizeof(Light) * lights_length);
  unsigned int light_group_count   = 1;

  vec3 sun;
  sun.x = sinf(data.azimuth) * cosf(data.altitude) * 149630000000.0f;
  sun.y = sinf(data.altitude) * 149630000000.0f;
  sun.z = cosf(data.azimuth) * cosf(data.altitude) * 149630000000.0f;

  light_groups[0].pos    = sun;
  light_groups[0].radius = 696340000.0f;

  while (light_count != 0) {
    if (light_group_count == light_groups_length) {
      light_groups_length *= 2;
      light_groups = realloc(light_groups, sizeof(Light) * light_groups_length);
    }

    printf("\r                                  \r%d Lights left to process.", light_count);

    Light light_group;

    light_group.pos    = lights[0].pos;
    light_group.radius = lights[0].radius;

    vec3 sum_pos                 = light_group.pos;
    unsigned int lights_in_group = 1;

    for (unsigned int i = 1; i < light_count; i++) {
      Light light = lights[i];

      if (light.radius < 0.0f)
        continue;

      const float dist = vector_distance(light.pos, light_group.pos) - light_group.radius;

      if (dist <= light.radius) {
        sum_pos.x += light.pos.x;
        sum_pos.y += light.pos.y;
        sum_pos.z += light.pos.z;

        lights_in_group++;

        vec3 new_pos;
        new_pos.x = sum_pos.x / lights_in_group;
        new_pos.y = sum_pos.y / lights_in_group;
        new_pos.z = sum_pos.z / lights_in_group;

        const float min_radius_increase = vector_distance(new_pos, light_group.pos);
        light_group.radius += (min_radius_increase < dist) ? dist : min_radius_increase;

        light_group.pos  = new_pos;
        lights[i].radius = -1.0f;

        i = 1;
      }
    }

    unsigned int new_light_count = 0;

    for (unsigned int i = 1; i < light_count; i++) {
      Light light = lights[i];

      if (light.radius >= 0.0f) {
        lights[new_light_count++] = light;
      }
    }

    light_count = new_light_count;

    light_groups[light_group_count] = light_group;
    light_group_count++;
  }

  printf("\r                                      \r");

  free(lights);

  light_groups = realloc(light_groups, sizeof(Light) * light_group_count);

  data.lights        = light_groups;
  data.lights_length = light_group_count;

  *scene = data;
}
