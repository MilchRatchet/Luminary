#include "texture.h"

#include "internal_error.h"
#include "utils.h"

LuminaryResult texture_create(
  Texture** _tex, uint32_t width, uint32_t height, uint32_t depth, uint32_t pitch, void* data, TextureDataType type,
  uint32_t num_components, TextureStorageLocation storage) {
  if (!_tex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Texture is NULL.");
  }

  Texture* tex;
  __FAILURE_HANDLE(host_malloc(&tex, sizeof(Texture)));

  tex->width            = width;
  tex->height           = height;
  tex->depth            = depth;
  tex->pitch            = pitch;
  tex->data             = data;
  tex->dim              = (depth > 1) ? Tex3D : Tex2D;
  tex->storage          = storage;
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
  if (!tex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Texture ptr is NULL.");
  }

  if (!(*tex)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture is NULL.");
  }

  if ((*tex)->data) {
    __FAILURE_HANDLE(host_free(&(*tex)->data));
  }

  __FAILURE_HANDLE(host_free(tex));

  return LUMINARY_SUCCESS;
}
