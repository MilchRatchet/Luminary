/*
  Copyright (C) 2021-2024 Max Jenke

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

#ifndef LUMINARY_RINGBUFFER_H
#define LUMINARY_RINGBUFFER_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryRingBuffer;
typedef struct LuminaryRingBuffer LuminaryRingBuffer;

#define ringbuffer_create(buffer, size) _ringbuffer_create(buffer, size, (const char*) #buffer, (const char*) __func__, __LINE__)
#define ringbuffer_destroy(buffer) _ringbuffer_destroy(buffer, (const char*) #buffer, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult
  _ringbuffer_create(LuminaryRingBuffer** buffer, size_t size, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult ringbuffer_allocate_entry(LuminaryRingBuffer* buffer, size_t entry_size, void** entry);
LUMINARY_API LuminaryResult ringbuffer_release_entry(LuminaryRingBuffer* buffer, size_t entry_size);
LUMINARY_API LuminaryResult _ringbuffer_destroy(LuminaryRingBuffer** buffer, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_RINGBUFFER_H */
