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
