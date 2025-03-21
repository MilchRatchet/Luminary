#include "texture.h"

#include <string.h>

#include "internal_error.h"
#include "utils.h"

static uint32_t _texture_compute_pitch(const uint32_t width, TextureDataType type, uint32_t num_components) {
  switch (type) {
    case TexDataFP32:
      return 4 * num_components * width;
    case TexDataUINT8:
      return 1 * num_components * width;
    case TexDataUINT16:
      return 2 * num_components * width;
    default:
      return 0;
  }
}

LuminaryResult texture_create(
  Texture** _tex, uint32_t width, uint32_t height, uint32_t depth, void* data, TextureDataType type, uint32_t num_components) {
  __CHECK_NULL_ARGUMENT(_tex);

  Texture* tex;
  __FAILURE_HANDLE(host_malloc(&tex, sizeof(Texture)));

  tex->width            = width;
  tex->height           = height;
  tex->depth            = depth;
  tex->pitch            = _texture_compute_pitch(width, type, num_components);
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
