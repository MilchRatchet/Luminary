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

#ifndef LUMINARY_PATH_H
#define LUMINARY_PATH_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryPath;
typedef struct LuminaryPath LuminaryPath;

LUMINARY_API LuminaryResult luminary_path_create(LuminaryPath** path);
LUMINARY_API LuminaryResult luminary_path_set_from_string(LuminaryPath* path, const char* string);

/*
 * Returns an ASCII encoded string corresponding to the given path. If override_path is not null,
 * then if override is an absolute override_path override will be returned, else a string containing the absolute
 * path given by interpreting override_path relative to the given path. The string must not be freed and is only valid until
 * the next path_* function call on this path object.
 * @param path Path instance.
 * @param override_path String containing a path. Optional.
 * @param string The destination the address of the string will be written to.
 *               If the computed path is empty, an empty string will be returned.
 */
LUMINARY_API LuminaryResult luminary_path_apply(LuminaryPath* path, const char* override_path, const char** string);

LUMINARY_API LuminaryResult luminary_path_clear(LuminaryPath* path);
LUMINARY_API LuminaryResult luminary_path_get_is_empty(LuminaryPath* path, bool* is_empty);

LUMINARY_API LuminaryResult luminary_path_destroy(LuminaryPath** path);

#endif /* LUMINARY_PATH_H */
