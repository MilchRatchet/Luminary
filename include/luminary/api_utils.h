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

#ifndef LUMINARY_API_UTILS_H
#define LUMINARY_API_UTILS_H

#include <stdbool.h>

#define LUMINARY_API
#define LUMINARY_DEPRECATED

LUMINARY_API struct LuminaryVec3 {
  float x;
  float y;
  float z;
} typedef LuminaryVec3;

LUMINARY_API struct LuminaryRGBF {
  float r;
  float g;
  float b;
} typedef LuminaryRGBF;

LUMINARY_API struct LuminaryRGBAF {
  float r;
  float g;
  float b;
  float a;
} typedef LuminaryRGBAF;

#endif /* LUMINARY_API_UTILS_H */
