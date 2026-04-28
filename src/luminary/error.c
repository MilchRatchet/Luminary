#include "error_registry.h"
#include "utils.h"

void luminary_get_error_details(LuminaryResult result, const LuminaryError** error) {
  error_registry_get_from_result(result, error);
}
