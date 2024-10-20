#include <string.h>

#include "internal_error.h"
#include "utils.h"

struct ArrayHeader {
  uint64_t magic;
  uint64_t size_of_element;
  uint32_t num_elements;
  uint32_t allocated_num_elements;
  uint64_t padding[5];
} typedef ArrayHeader;
LUM_STATIC_SIZE_ASSERT(ArrayHeader, 64u);

// LUMARRAY
#define ARRAY_HEADER_MAGIC (0x59415252414D554Cull)
#define ARRAY_HEADER_FREED_MAGIC (420ull)

LuminaryResult _array_create(
  void** _array, size_t size_of_element, uint32_t num_elements, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_array);

  void* array;
  __FAILURE_HANDLE(_host_malloc((void**) &array, size_of_element * num_elements + sizeof(ArrayHeader), buf_name, func, line));

  ArrayHeader* header = array;

  header->magic                  = ARRAY_HEADER_MAGIC;
  header->size_of_element        = size_of_element;
  header->allocated_num_elements = num_elements;
  header->num_elements           = 0;

  *_array = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _array_resize(void** array, size_t num_elements, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(array);
  __CHECK_NULL_ARGUMENT(*array);

  ArrayHeader* header = ((ArrayHeader*) (*array)) - 1;

  if (header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  __FAILURE_HANDLE(_host_realloc((void**) &header, header->size_of_element * num_elements + sizeof(ArrayHeader), buf_name, func, line));

  header->allocated_num_elements = num_elements;
  header->num_elements           = (header->num_elements < num_elements) ? header->num_elements : num_elements;

  *array = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _array_destroy(void** array, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(array);
  __CHECK_NULL_ARGUMENT(*array);

  ArrayHeader* header = ((ArrayHeader*) (*array)) - 1;

  if (header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  header->magic = ARRAY_HEADER_FREED_MAGIC;

  __FAILURE_HANDLE(_host_free((void**) &header, buf_name, func, line));

  *array = (void*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult _array_push(void** array, void* object, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(array);
  __CHECK_NULL_ARGUMENT(object);
  __CHECK_NULL_ARGUMENT(*array);

  ArrayHeader* header = ((ArrayHeader*) (*array)) - 1;

  if (header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  if (header->num_elements == 0xFFFFFFFF) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Array exceeded maximum number of elements.");
  }

  if (header->num_elements == header->allocated_num_elements) {
    __FAILURE_HANDLE(_array_resize(array, header->allocated_num_elements * 2, buf_name, func, line));
    header = ((ArrayHeader*) (*array)) - 1;
  }

  uint8_t* dst_ptr = ((uint8_t*) (header + 1)) + (header->num_elements * header->size_of_element);

  memcpy(dst_ptr, object, header->size_of_element);

  header->num_elements++;

  return LUMINARY_SUCCESS;
}

LuminaryResult _array_copy(void** dst, const void* src, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(src);
  __CHECK_NULL_ARGUMENT(dst);

  __FAILURE_HANDLE(array_clear(*dst));
  __FAILURE_HANDLE(_array_append(dst, src, buf_name, func, line));

  return LUMINARY_SUCCESS;
}

LuminaryResult _array_append(void** dst, const void* src, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(src);
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(*dst);

  const ArrayHeader* src_header = ((const ArrayHeader*) src) - 1;

  ArrayHeader* dst_header = ((ArrayHeader*) (*dst)) - 1;

  if (src_header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  if (dst_header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  if (dst_header->size_of_element != src_header->size_of_element) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Array elements are of different size.");
  }

  if (dst_header->allocated_num_elements < dst_header->num_elements + src_header->num_elements) {
    __FAILURE_HANDLE(_array_resize(dst, dst_header->num_elements + src_header->num_elements, buf_name, func, line));
    dst_header = ((ArrayHeader*) (*dst)) - 1;
  }

  void* memcpy_dst = (void*) (((uint8_t*) (*dst)) + dst_header->size_of_element * dst_header->num_elements);

  memcpy(memcpy_dst, src, src_header->size_of_element * src_header->num_elements);

  dst_header->num_elements += src_header->num_elements;

  return LUMINARY_SUCCESS;
}

LuminaryResult array_clear(void* array) {
  __CHECK_NULL_ARGUMENT(array);

  ArrayHeader* src_header = ((ArrayHeader*) array) - 1;

  if (src_header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  src_header->num_elements = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult array_get_size(const void* array, size_t* size) {
  __CHECK_NULL_ARGUMENT(array);

  const ArrayHeader* header = ((const ArrayHeader*) array) - 1;

  if (header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  *size = header->allocated_num_elements;

  return LUMINARY_SUCCESS;
}

LuminaryResult array_get_num_elements(const void* array, uint32_t* num_elements) {
  __CHECK_NULL_ARGUMENT(array);
  __CHECK_NULL_ARGUMENT(num_elements);

  const ArrayHeader* header = ((const ArrayHeader*) array) - 1;

  if (header->magic != ARRAY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given pointer is not an array.");
  }

  *num_elements = header->num_elements;

  return LUMINARY_SUCCESS;
}
