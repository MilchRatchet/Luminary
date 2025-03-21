#ifndef MANDARIN_DUCK_RENDER_REGION_H
#define MANDARIN_DUCK_RENDER_REGION_H

#include "mouse_state.h"
#include "utils.h"

struct Display typedef Display;
struct UIRenderer typedef UIRenderer;

enum RenderRegionVertex {
  RENDER_REGION_VERTEX_TOP_LEFT,
  RENDER_REGION_VERTEX_TOP_RIGHT,
  RENDER_REGION_VERTEX_BOTTOM_LEFT,
  RENDER_REGION_VERTEX_BOTTOM_RIGHT,
  RENDER_REGION_VERTEX_COUNT
} typedef RenderRegionVertex;

enum RenderRegionState {
  RENDER_REGION_STATE_DEFAULT,
  RENDER_REGION_STATE_MOVE,
  RENDER_REGION_STATE_VERTEX_MOVE,
  RENDER_REGION_STATE_COUNT
} typedef RenderRegionState;

struct RenderRegion {
  RenderRegionState state;
  RenderRegionVertex selected_vertex;
  float x;
  float y;
  float width;
  float height;
} typedef RenderRegion;

void render_region_create(RenderRegion** region);
void render_region_handle_inputs(RenderRegion* region, Display* display, LuminaryHost* host, MouseState* mouse_state);
void render_region_remove_focus(RenderRegion* region, LuminaryHost* host);
void render_region_render(RenderRegion* region, Display* display, UIRenderer* renderer);
void render_region_destroy(RenderRegion** region);

#endif /* MANDARIN_DUCK_RENDER_REGION_H */
