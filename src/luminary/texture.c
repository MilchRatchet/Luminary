#include "texture.h"

#include <string.h>

#include "internal_error.h"
#include "utils.h"

LuminaryResult texture_create(
  Texture** _tex, uint32_t width, uint32_t height, uint32_t depth, uint32_t pitch, void* data, TextureDataType type,
  uint32_t num_components) {
  __CHECK_NULL_ARGUMENT(_tex);

  Texture* tex;
  __FAILURE_HANDLE(host_malloc(&tex, sizeof(Texture)));

  tex->width            = width;
  tex->height           = height;
  tex->depth            = depth;
  tex->pitch            = pitch;
  tex->data             = data;
  tex->dim              = (depth > 1) ? Tex3D : Tex2D;
  tex->type             = type;
  tex->wrap_mode_S      = TexModeWrap;
  tex->wrap_mode_T      = TexModeWrap;
  tex->wrap_mode_R      = TexModeWrap;
  tex->filter           = TexFilterLinear;
  tex->read_mode        = TexReadModeNormalized;
  tex->mipmap           = TexMipmapNone;
  tex->mipmap_max_level = 0;
  tex->gamma            = 1.0f;
  tex->num_components   = num_components;

  *_tex = tex;

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_destroy(Texture** tex) {
  __CHECK_NULL_ARGUMENT(tex);
  __CHECK_NULL_ARGUMENT(*tex);

  if ((*tex)->data) {
    __FAILURE_HANDLE(host_free(&(*tex)->data));
  }

  __FAILURE_HANDLE(host_free(tex));

  return LUMINARY_SUCCESS;
}
