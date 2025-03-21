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
LUMINARY_API LuminaryResult luminary_path_destroy(LuminaryPath** path);

#endif /* LUMINARY_PATH_H */
