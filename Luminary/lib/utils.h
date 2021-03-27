#ifndef UTILS_H
#define UTILS_H

#include "primitives.h"
#include "mesh.h"
#include "bvh.h"
#include "texture.h"

struct Camera {
  vec3 pos;
  vec3 rotation;
  float fov;
} typedef Camera;

struct Scene {
  Camera camera;
  unsigned int far_clip_distance;
  Triangle* triangles;
  unsigned int triangles_length;
  Node* nodes;
  unsigned int nodes_length;
  int meshes_length;
  texture_assignment* texture_assignments;
  float azimuth;
  float altitude;
  float sun_strength;
} typedef Scene;

struct raytrace_instance {
  unsigned int width;
  unsigned int height;
  RGBF* frame_buffer;
  RGBF* frame_buffer_gpu;
  void* albedo_atlas;
  int albedo_atlas_length;
  void* illuminance_atlas;
  int illuminance_atlas_length;
  void* material_atlas;
  int material_atlas_length;
  int reflection_depth;
  int diffuse_samples;
} typedef raytrace_instance;

#endif /* UTILS_H */
