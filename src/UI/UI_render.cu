#include "UI.h"
#include "UI_render.h"
#include "utils.cuh"

__constant__ UIRenderingContext ui;

LUMINARY_KERNEL void ui_render_kernel() {
}

LUMINARY_KERNEL void ui_blit_kernel(XRGB8* dst, int width, int height, int ld) {
  unsigned int id = THREAD_ID;

  const int amount = ui.width * ui.height;

  while (id < amount) {
    const int y = id / ui.width;
    const int x = id - y * ui.width;

    const int dst_x = x + ui.offset_x;
    const int dst_y = height - 1 - (y + ui.offset_y);

    XRGB8 pixel;
    pixel.ignore = 0;
    pixel.r      = 0;
    pixel.g      = 0;
    pixel.b      = 0xFF;

    dst[dst_x + dst_y * ld] = pixel;

    id += blockDim.x * gridDim.x;
  }
}

void ui_render_update_context(UI* _ui) {
  if (!_ui->rendering_context_dirty)
    return;

  UIRenderingContext ctx;

  ctx.width    = UI_WIDTH;
  ctx.height   = UI_HEIGHT;
  ctx.offset_x = _ui->x;
  ctx.offset_y = _ui->y;

  gpuErrchk(cudaMemcpyToSymbol(ui, &ctx, sizeof(UIRenderingContext), 0, cudaMemcpyHostToDevice));
}

void ui_render_internal(UI* ui, void* dst, int width, int height, int ld) {
  ui_blit_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((XRGB8*) dst, width, height, ld);
}
