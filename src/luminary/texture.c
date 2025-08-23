#include "texture.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "host/png.h"
#include "image.h"
#include "internal_error.h"
#include "queue_worker.h"
#include "spinlock.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////
// Async loading
////////////////////////////////////////////////////////////////////

struct TextureAsyncLoadingWork {
  Texture* texture;
  bool success;
  char* path;
  SpinLockObject lock;  // Locked during loading, used for other threads to wait on the
} typedef TextureAsyncLoadingWork;

static LuminaryResult _texture_load_async_queue_work(void* context, TextureAsyncLoadingWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  // Context-less
  (void) context;

  LuminaryResult result = image_load(work->texture, work->path);

  work->success = result == LUMINARY_SUCCESS;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _texture_load_async_clear_work(void* context, TextureAsyncLoadingWork* work) {
  __CHECK_NULL_ARGUMENT(work);

  // Context-less
  (void) context;

  Texture* texture = work->texture;

  texture->status = (work->success) ? TEXTURE_STATUS_NONE : TEXTURE_STATUS_INVALID;

  atomic_thread_fence(memory_order_release);

  __FAILURE_HANDLE(host_free(&work->path));

  spinlock_unlock(&work->lock);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// API
////////////////////////////////////////////////////////////////////

static uint32_t _texture_compute_pitch(const uint32_t width, TextureDataType type, uint32_t num_components) {
  switch (type) {
    case TEXTURE_DATA_TYPE_FP32:
      return 4 * num_components * width;
    case TEXTURE_DATA_TYPE_U8:
      return 1 * num_components * width;
    case TEXTURE_DATA_TYPE_U16:
      return 2 * num_components * width;
    default:
      return 0;
  }
}

LuminaryResult texture_create(Texture** texture) {
  __CHECK_NULL_ARGUMENT(texture);

  __FAILURE_HANDLE(host_malloc(texture, sizeof(Texture)));
  memset(*texture, 0, sizeof(Texture));

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_fill(
  Texture* tex, uint32_t width, uint32_t height, uint32_t depth, void* data, TextureDataType type, uint32_t num_components) {
  __CHECK_NULL_ARGUMENT(tex);

  tex->status           = TEXTURE_STATUS_NONE;
  tex->width            = width;
  tex->height           = height;
  tex->depth            = depth;
  tex->pitch            = _texture_compute_pitch(width, type, num_components);
  tex->data             = data;
  tex->dim              = (depth > 1) ? TEXTURE_DIMENSION_TYPE_3D : TEXTURE_DIMENSION_TYPE_2D;
  tex->type             = type;
  tex->wrap_mode_S      = TEXTURE_WRAPPING_MODE_WRAP;
  tex->wrap_mode_T      = TEXTURE_WRAPPING_MODE_WRAP;
  tex->wrap_mode_R      = TEXTURE_WRAPPING_MODE_WRAP;
  tex->filter           = TEXTURE_FILTER_MODE_LINEAR;
  tex->read_mode        = TEXTURE_READ_MODE_NORMALIZED;
  tex->mipmap           = TEXTURE_MIPMAP_MODE_NONE;
  tex->mipmap_max_level = 0;
  tex->gamma            = 1.0f;
  tex->num_components   = num_components;

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_invalidate(Texture* texture) {
  __CHECK_NULL_ARGUMENT(texture);

  // Save the work buffer so we can reinstate it after clearing the texture
  TextureAsyncLoadingWork* work = texture->async_work_data;

  memset(texture, 0, sizeof(Texture));

  texture->status          = TEXTURE_STATUS_INVALID;
  texture->async_work_data = work;

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_is_valid(const Texture* texture, bool* is_valid) {
  __CHECK_NULL_ARGUMENT(texture);

  *is_valid = texture->status != TEXTURE_STATUS_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_load_async(Texture* texture, Queue* queue, const char* path) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(path);

  TextureAsyncLoadingWork* work = texture->async_work_data;

  if (work == (TextureAsyncLoadingWork*) 0) {
    __FAILURE_HANDLE(host_malloc(&work, sizeof(TextureAsyncLoadingWork)));
    memset(work, 0, sizeof(TextureAsyncLoadingWork));

    work->texture = texture;
    work->lock    = false;

    texture->async_work_data = work;
  }

  spinlock_lock(&work->lock);

  const size_t path_len = strlen(path);
  __FAILURE_HANDLE(host_malloc(&work->path, path_len + 1));
  memcpy(work->path, path, path_len);
  work->path[path_len] = '\0';

  texture->status = TEXTURE_STATUS_ASYNC_LOADING;

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Load Texture";
  entry.function   = (QueueEntryFunction) _texture_load_async_queue_work;
  entry.clear_func = (QueueEntryFunction) _texture_load_async_clear_work;
  entry.args       = (void*) work;

  __FAILURE_HANDLE(queue_push(queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_await(const Texture* texture) {
  __CHECK_NULL_ARGUMENT(texture);

  atomic_thread_fence(memory_order_acquire);

  if (texture->status != TEXTURE_STATUS_ASYNC_LOADING)
    return LUMINARY_SUCCESS;

  if (texture->async_work_data == (TextureAsyncLoadingWork*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture is async loading but there is no async work data.");
  }

  TextureAsyncLoadingWork* work = texture->async_work_data;

  spinlock_lock(&work->lock);

  if (texture->status == TEXTURE_STATUS_ASYNC_LOADING) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture is marked as async loading but loading thread has release the mutex.");
  }

  spinlock_unlock(&work->lock);

  return LUMINARY_SUCCESS;
}

LuminaryResult texture_destroy(Texture** tex) {
  __CHECK_NULL_ARGUMENT(tex);
  __CHECK_NULL_ARGUMENT(*tex);

  __FAILURE_HANDLE(texture_await(*tex));

  if ((*tex)->async_work_data != (TextureAsyncLoadingWork*) 0) {
    __FAILURE_HANDLE(host_free(&(*tex)->async_work_data));
  }

  if ((*tex)->data) {
    __FAILURE_HANDLE(host_free(&(*tex)->data));
  }

  __FAILURE_HANDLE(host_free(tex));

  return LUMINARY_SUCCESS;
}
