#ifndef LUMINARY_QUEUE_H
#define LUMINARY_QUEUE_H

#include <luminary/api_utils.h>

struct LuminaryQueue;
typedef struct LuminaryQueue LuminaryQueue;

#define queue_create(array, size_of_element, num_elements) \
  _queue_create(queue, size_of_element, num_elements, (const char*) #array, (const char*) __func__, __LINE__)
#define queue_push(array, object) _queue_push(queue, (void*) object, (const char*) #array, (const char*) __func__, __LINE__)
#define queue_pop(array, size) _queue_pop(queue, (void*) object, (const char*) #array, (const char*) __func__, __LINE__)
#define queue_destroy(array) _queue_destroy(queue, (const char*) #array, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult _queue_create(LuminaryQueue** queue, size_t size_of_element, size_t num_elements);
LUMINARY_API LuminaryResult _queue_push(LuminaryQueue* queue, void* object);
LUMINARY_API LuminaryResult _queue_pop(LuminaryQueue* queue, void* object);
LUMINARY_API LuminaryResult _queue_destroy(LuminaryQueue** queue);

#endif /* LUMINARY_QUEUE_H */
