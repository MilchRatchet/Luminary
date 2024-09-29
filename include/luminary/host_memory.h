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

#ifndef LUMINARY_API_HOST_MEMORY_H
#define LUMINARY_API_HOST_MEMORY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

#define host_malloc(ptr, size) _host_malloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define host_realloc(ptr, size) _host_realloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define host_free(ptr) _host_free((void**) ptr, (const char*) #ptr, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult _host_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _host_realloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult _host_free(void** ptr, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_API_HOST_MEMORY_H */
