#include "lib/scene.h"
#include <stdlib.h>

Scene example_scene1() {
  Camera camera = {
    .pos      = {.x = 0.0f, .y = -3.0f, .z = 0.0f},
    .rotation = {.x = 0.3f, .y = -0.7f, .z = 0.0f},
    .fov      = 1.0f};

  Sphere* spheres = (Sphere*) malloc(sizeof(Sphere) * 12);

  spheres[0].id         = 1;
  spheres[0].pos.x      = 0.0f;
  spheres[0].pos.y      = 0.0f;
  spheres[0].pos.z      = -4.0f;
  spheres[0].radius     = 1.0f;
  spheres[0].sign       = 1.0f;
  spheres[0].color.r    = 0.8f;
  spheres[0].color.g    = 0.05f;
  spheres[0].color.b    = 0.05f;
  spheres[0].emission.r = 0.0f;
  spheres[0].emission.g = 0.0f;
  spheres[0].emission.b = 0.0f;
  spheres[0].intensity  = 0.0f;
  spheres[0].smoothness = 0.5f;

  spheres[1].id         = 2;
  spheres[1].pos.x      = 5.0f;
  spheres[1].pos.y      = 0.25f;
  spheres[1].pos.z      = -4.0f;
  spheres[1].radius     = 0.75f;
  spheres[1].sign       = 1.0f;
  spheres[1].color.r    = 0.5f;
  spheres[1].color.g    = 0.5f;
  spheres[1].color.b    = 0.8f;
  spheres[1].emission.r = 0.0f;
  spheres[1].emission.g = 0.0f;
  spheres[1].emission.b = 0.0f;
  spheres[1].intensity  = 0.0f;
  spheres[1].smoothness = 0.5f;

  spheres[2].id         = 3;
  spheres[2].pos.x      = 8.0f;
  spheres[2].pos.y      = 0.0f;
  spheres[2].pos.z      = -6.0f;
  spheres[2].radius     = 1.0f;
  spheres[2].sign       = 1.0f;
  spheres[2].color.r    = 0.9f;
  spheres[2].color.g    = 0.9f;
  spheres[2].color.b    = 0.9f;
  spheres[2].emission.r = 0.0f;
  spheres[2].emission.g = 0.0f;
  spheres[2].emission.b = 0.0f;
  spheres[2].intensity  = 0.0f;
  spheres[2].smoothness = 0.5f;

  spheres[3].id         = 4;
  spheres[3].pos.x      = 12.0f;
  spheres[3].pos.y      = 0.0f;
  spheres[3].pos.z      = -5.0f;
  spheres[3].radius     = 1.0f;
  spheres[3].sign       = 1.0f;
  spheres[3].color.r    = 0.8f;
  spheres[3].color.g    = 0.5f;
  spheres[3].color.b    = 0.1f;
  spheres[3].emission.r = 0.0f;
  spheres[3].emission.g = 0.0f;
  spheres[3].emission.b = 0.0f;
  spheres[3].intensity  = 0.0f;
  spheres[3].smoothness = 0.5f;

  spheres[4].id         = 5;
  spheres[4].pos.x      = 2.0f;
  spheres[4].pos.y      = 0.25f;
  spheres[4].pos.z      = -11.0f;
  spheres[4].radius     = 0.75f;
  spheres[4].sign       = 1.0f;
  spheres[4].color.r    = 0.1f;
  spheres[4].color.g    = 0.1f;
  spheres[4].color.b    = 0.6f;
  spheres[4].emission.r = 0.0f;
  spheres[4].emission.g = 0.0f;
  spheres[4].emission.b = 0.0f;
  spheres[4].intensity  = 0.0f;
  spheres[4].smoothness = 0.5f;

  spheres[5].id         = 501;
  spheres[5].pos.x      = 12.0f;
  spheres[5].pos.y      = 0.5f;
  spheres[5].pos.z      = -12.0f;
  spheres[5].radius     = 0.5f;
  spheres[5].sign       = 1.0f;
  spheres[5].color.r    = 0.0f;
  spheres[5].color.g    = 0.0f;
  spheres[5].color.b    = 0.0f;
  spheres[5].emission.r = 1.0f;
  spheres[5].emission.g = 1.0f;
  spheres[5].emission.b = 1.0f;
  spheres[5].intensity  = 1000.0f;
  spheres[5].smoothness = 0.0f;

  Cuboid* cuboids = (Cuboid*) malloc(sizeof(Cuboid) * 4);

  cuboids[0].id         = 101;
  cuboids[0].pos.x      = 0.0f;
  cuboids[0].pos.y      = 2.0f;
  cuboids[0].pos.z      = 0.0f;
  cuboids[0].size.x     = 20.0f;
  cuboids[0].size.y     = 1.0f;
  cuboids[0].size.z     = 20.0f;
  cuboids[0].sign       = 1.0f;
  cuboids[0].color.r    = 0.4f;
  cuboids[0].color.g    = 0.4f;
  cuboids[0].color.b    = 0.4f;
  cuboids[0].emission.r = 0.0f;
  cuboids[0].emission.g = 0.0f;
  cuboids[0].emission.b = 0.0f;
  cuboids[0].intensity  = 0.0f;
  cuboids[0].smoothness = 0.05f;

  cuboids[1].id         = 102;
  cuboids[1].pos.x      = 14.0f;
  cuboids[1].pos.y      = -2.0f;
  cuboids[1].pos.z      = 0.0f;
  cuboids[1].size.x     = 1.0f;
  cuboids[1].size.y     = 4.0f;
  cuboids[1].size.z     = 20.0f;
  cuboids[1].sign       = 1.0f;
  cuboids[1].color.r    = 0.2f;
  cuboids[1].color.g    = 0.2f;
  cuboids[1].color.b    = 0.2f;
  cuboids[1].emission.r = 0.0f;
  cuboids[1].emission.g = 0.0f;
  cuboids[1].emission.b = 0.0f;
  cuboids[1].intensity  = 0.0f;
  cuboids[1].smoothness = 0.05f;

  Light* lights = (Light*) malloc(sizeof(Light));

  lights[0].id    = 201;
  lights[0].pos.x = 12.0f;
  lights[0].pos.y = -5.5f;
  lights[0].pos.z = -12.0f;

  lights[0].color.r = 1.0f;
  lights[0].color.g = 1.0f;
  lights[0].color.b = 1.0f;

  Scene scene = {
    .camera            = camera,
    .far_clip_distance = 1000,
    .spheres           = spheres,
    .spheres_length    = 6,
    .cuboids           = cuboids,
    .cuboids_length    = 2,
    .lights            = lights,
    .lights_length     = 1};

  return scene;
}
