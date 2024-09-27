/*
  Copyright (c) 2021-2024, MilchRatchet

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef LUMINARY_API_ARRAY_H
#define LUMINARY_API_ARRAY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

#define array_create(array, size_of_element, num_elements) \
  _array_create((void**) array, size_of_element, num_elements, (const char*) #array, (const char*) __func__, __LINE__)
#define array_resize(array, size) _array_resize((void**) array, size, (const char*) #array, (const char*) __func__, __LINE__)
#define array_push(array, object) _array_push((void**) array, (void*) object, (const char*) #array, (const char*) __func__, __LINE__)
#define array_destroy(array) _array_destroy((void**) array, (const char*) #array, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult
  _array_create(void** array, size_t size_of_element, uint32_t num_elements, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_resize(void** array, size_t size, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_push(void** array, void* object, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _array_destroy(void** array, const char* buf_name, const char* func, uint32_t line);

LUMINARY_API LuminaryResult array_get_size(const void* array, size_t* size);
LUMINARY_API LuminaryResult array_get_num_elements(const void* array, uint32_t* num_elements);

#endif /* LUMINARY_API_ARRAY_H */
