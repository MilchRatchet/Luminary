#include "internal_error.h"
#include "lum.h"

LuminaryResult lum_parse_file_v5(FILE* file, LumFileContent* content) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(content);

  __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Lum v5 support is not yet implemented.");
}
