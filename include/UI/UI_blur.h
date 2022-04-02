#ifndef UI_BLUR_H
#define UI_BLUR_H

#include "UI_structs.h"

void blur_background(UI* ui, uint8_t* target, int width);
size_t blur_scratch_needed();

#endif /* UI_BLUR_H */
