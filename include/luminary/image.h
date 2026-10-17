/*
  Copyright (C) 2021-2026 Max Jenke

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

#ifndef LUMINARY_API_IMAGE_H
#define LUMINARY_API_IMAGE_H

#include <luminary/path.h>
#include <luminary/structs.h>

LUMINARY_API enum LuminaryImageFileFormat {
  LUMINARY_IMAGE_FILE_FORMAT_PNG,
  LUMINARY_IMAGE_FILE_FORMAT_JPG,
  LUMINARY_IMAGE_FILE_FORMAT_BMP,
  LUMINARY_IMAGE_FILE_FORMAT_TGA,
  LUMINARY_IMAGE_FILE_FORMAT_COUNT
} typedef LuminaryImageFileFormat;

LUMINARY_API struct LuminaryImageSaveArgs {
  LuminaryImage image;
  LuminaryPath* file_path;
  LuminaryImageFileFormat format;
  uint32_t jpeg_quality; /* Quality from 1 to 100 for JPG */
} typedef LuminaryImageSaveArgs;

LUMINARY_API LuminaryResult luminary_image_save(const LuminaryImageSaveArgs* args);

#endif /* LUMINARY_API_IMAGE_H */
