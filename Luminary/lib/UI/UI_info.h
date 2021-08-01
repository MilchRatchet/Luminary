#ifndef UI_INFO_H
#define UI_INFO_H

#include "UI_structs.h"

#define PANEL_INFO_TYPE_INT8 0x01
#define PANEL_INFO_TYPE_INT16 0x02
#define PANEL_INFO_TYPE_INT32 0x03
#define PANEL_INFO_TYPE_INT64 0x04
#define PANEL_INFO_TYPE_FP32 0x11
#define PANEL_INFO_TYPE_FP64 0x12

#define PANEL_INFO_STATIC 0x0
#define PANEL_INFO_DYNAMIC 0x1

void render_UIPanel_info(UI* ui, UIPanel* panel);

#endif /* UI_INFO_H */
