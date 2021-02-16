#ifndef SCENE_H
#define SCENE_H

#include "primitives.h"

struct Camera {
  vec3 pos;
  vec3 rotation;
  float fov;  // As the ratio of grid width / 2 to grid distance from camera
} typedef Camera;

struct Scene {
  Camera camera;
  unsigned int far_clip_distance;
  Sphere* spheres;
  unsigned int spheres_length;
  Cuboid* cuboids;
  unsigned int cuboids_length;
} typedef Scene;

#endif /* SCENE_H */
