#ifndef UI_TEXT_H
#define UI_TEXT_H

#include "UI.h"
#include "UI_structs.h"
#include "SDL_ttf.h"

void init_text(UI* ui);
SDL_Surface* render_text(UI* ui, const char* text);

#endif /* UI_TEXT_H */
