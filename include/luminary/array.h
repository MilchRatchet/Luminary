/*
  Copyright (C) 2021-2025 Max Jenke

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef LUMINARY_API_ARRAY_H
#define LUMINARY_API_ARRAY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

#define array_create(array, size_of_element, num_elements) \
  _array_create((void**) (array), (size_of_element), (num_elements), (const char*) #array, (const char*) __func__, __LINE__)
#define array_resize(array, size) _array_resize((void**) (array), (size), (const char*) #array, (const char*) __func__, __LINE__)
#define array_push(array, object) _array_push((void**) (array), (void*) (object), (const char*) #array, (const char*) __func__, __LINE__)
#define array_copy(dst, src) _array_copy((void**) (dst), (void**) (src), (const char*) #dst, (const char*) __func__, __LINE__)
#define array_append(dst, src) _array_append((void**) (dst), (const void*) (src), (const char*) #dst, (const char*) __func__, __LINE__)
#define array_set_num_elements(array, num_elements) \
  _array_set_num_elements((void**) (array), (num_elements), (const char*) #array, (const char*) __func__, __LINE__)
#define array_destroy(array) _array_destroy((void**) (array), (const char*) #array, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult
  _array_create(void** array, size_t size_of_element, uint32_t num_elements, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_resize(void** array, size_t size, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_push(void** array, void* object, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_copy(void** dst, const void* src, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_append(void** dst, const void* src, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_destroy(void** array, const char* buf_name, const char* func, uint32_t line);

LUMINARY_API LuminaryResult array_clear(void* array);
LUMINARY_API LuminaryResult array_get_size(const void* array, size_t* size);
LUMINARY_API LuminaryResult array_get_num_elements(const void* array, uint32_t* num_elements);
LUMINARY_API LuminaryResult
  _array_set_num_elements(void** array, uint32_t num_elements, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_API_ARRAY_H */
