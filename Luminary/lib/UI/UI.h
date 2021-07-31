#ifndef UI_H
#define UI_H

#include <stdint.h>
#include "SDL_ttf.h"
#include "UI_panel.h"
#include "UI_structs.h"

#define UI_WIDTH 320
#define UI_HEIGHT 600

#define UI_PANELS_GENERAL_TAB 0x1
#define UI_PANELS_GENERAL_COUNT 1

UI init_UI();
void render_UI(UI* ui);
void blit_UI(UI* ui, uint8_t* target, int width, int height);
void free_UI(UI* ui);

#endif /* UI_H */
