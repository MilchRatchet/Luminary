#ifndef LUMINARY_API_ARRAY_H
#define LUMINARY_API_ARRAY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

#define ARRAY

#define array_create(array, size_of_element, num_elements) \
  _array_create((void**) array, size_of_element, num_elements, (const char*) #array, (const char*) __func__, __LINE__)
#define array_push(array, object) _array_push((void**) array, (void*) object, (const char*) #array, (const char*) __func__, __LINE__)
#define array_resize(array, size) _array_resize((void**) array, size, (const char*) #array, (const char*) __func__, __LINE__)
#define array_destroy(array) _array_destroy((void**) array, (const char*) #array, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult _array_create(void** array, size_t size_of_element, size_t num_elements);
LUMINARY_API LuminaryResult _array_push(void** array, void* object);
LUMINARY_API LuminaryResult _array_resize(void** array, size_t size);
LUMINARY_API LuminaryResult _array_destroy(void** array);

#endif /* LUMINARY_API_ARRAY_H */
