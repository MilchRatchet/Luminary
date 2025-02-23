#include "lum_tokenizer.h"

#include <stdlib.h>
#include <string.h>

#include "internal_error.h"

LuminaryResult lum_tokenizer_create(LumTokenizer** tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  __FAILURE_HANDLE(host_malloc(tokenizer, sizeof(LumTokenizer)));
  memset(*tokenizer, 0, sizeof(LumTokenizer));

  __FAILURE_HANDLE(array_create(&(*tokenizer)->tokens, sizeof(LumToken), 32));

  return LUMINARY_SUCCESS;
}

static bool _lum_tokenizer_is_identifier(LumToken* token, const char* code, uint32_t* consumed_chars) {
  const char* next_space      = strchr(code, ' ');
  const char* next_newline    = strchr(code, '\n');
  const char* next_semicolon  = strchr(code, ';');
  const char* next_func_begin = strchr(code, '[');
  const char* next_func_end   = strchr(code, ']');

  uint32_t num_chars = UINT32_MAX;

  num_chars = (next_space) ? min(next_space - code, num_chars) : num_chars;
  num_chars = (next_newline) ? min(next_newline - code, num_chars) : num_chars;
  num_chars = (next_semicolon) ? min(next_semicolon - code, num_chars) : num_chars;
  num_chars = (next_func_begin) ? min(next_func_begin - code, num_chars) : num_chars;
  num_chars = (next_func_end) ? min(next_func_end - code, num_chars) : num_chars;

  if (num_chars == UINT32_MAX) {
    num_chars = (uint32_t) strlen(code);
  }

  if (num_chars >= LUM_IDENTIFIER_MAX_LENGTH - 1)
    return false;

  *consumed_chars = num_chars;

  token->type = LUM_TOKEN_TYPE_IDENTIFIER;

  memcpy(token->identifier.name, code, num_chars);
  token->identifier.name[num_chars] = '\0';

  for (uint32_t builtin_type = 0; builtin_type < LUM_BUILTIN_TYPE_COUNT; builtin_type++) {
    if (strcmp(token->identifier.name, lum_builtin_types_strings[builtin_type]) == 0) {
      token->identifier.is_builtin_type = true;
      token->identifier.builtin_type    = (LumBuiltinType) builtin_type;
      break;
    }
  }

  return true;
}

static bool _lum_tokenizer_is_keyword(LumToken* token, const char* code, uint32_t* consumed_chars) {
  LUM_UNUSED(token);
  LUM_UNUSED(code);

  *consumed_chars = 0;

  return false;
}

static bool _lum_tokenizer_is_literal(LumToken* token, const char* code, uint32_t* consumed_chars) {
  const char* next_space      = strchr(code, ' ');
  const char* next_newline    = strchr(code, '\n');
  const char* next_semicolon  = strchr(code, ';');
  const char* next_func_begin = strchr(code, '[');
  const char* next_func_end   = strchr(code, ']');

  uint32_t num_chars = UINT32_MAX;

  num_chars = (next_space) ? min(next_space - code, num_chars) : num_chars;
  num_chars = (next_newline) ? min(next_newline - code, num_chars) : num_chars;
  num_chars = (next_semicolon) ? min(next_semicolon - code, num_chars) : num_chars;
  num_chars = (next_func_begin) ? min(next_func_begin - code, num_chars) : num_chars;
  num_chars = (next_func_end) ? min(next_func_end - code, num_chars) : num_chars;

  if (num_chars == UINT32_MAX) {
    num_chars = (uint32_t) strlen(code);
  }

  if (num_chars >= LUM_IDENTIFIER_MAX_LENGTH - 1)
    return false;

  const bool is_number = code[0] >= '0' && code[0] <= '9';

  if (is_number) {
    char literal_string[LUM_IDENTIFIER_MAX_LENGTH];
    memcpy(literal_string, code, num_chars);
    literal_string[num_chars] = '\0';

    bool has_decimal_seperator = false;
    for (uint32_t index = 0; index < num_chars; index++) {
      if (literal_string[index] == '.') {
        has_decimal_seperator = true;
        break;
      }
    }

    if (has_decimal_seperator) {
      const bool has_float_suffix = literal_string[num_chars - 1] == 'f';

      if (has_float_suffix) {
        token->type              = LUM_TOKEN_TYPE_LITERAL;
        token->literal.type      = LUM_LITERAL_TYPE_FLOAT;
        token->literal.val_float = strtof(literal_string, (char**) 0);
        *consumed_chars          = num_chars;
        return true;
      }
      else {
        token->type               = LUM_TOKEN_TYPE_LITERAL;
        token->literal.type       = LUM_LITERAL_TYPE_DOUBLE;
        token->literal.val_double = strtod(literal_string, (char**) 0);
        *consumed_chars           = num_chars;
        return true;
      }
    }
    else {
      token->type             = LUM_TOKEN_TYPE_LITERAL;
      token->literal.type     = LUM_LITERAL_TYPE_UINT;
      token->literal.val_uint = strtoul(literal_string, (char**) 0, 0);
      *consumed_chars         = num_chars;
      return true;
    }
  }
  else {
    if (num_chars == 4 && code[0] == 't' && code[1] == 'r' && code[2] == 'u' && code[3] == 'e') {
      token->type             = LUM_TOKEN_TYPE_LITERAL;
      token->literal.type     = LUM_LITERAL_TYPE_BOOL;
      token->literal.val_bool = true;
      *consumed_chars         = num_chars;
      return true;
    }

    if (num_chars == 5 && code[0] == 'f' && code[1] == 'a' && code[2] == 'l' && code[3] == 's' && code[4] == 'e') {
      token->type             = LUM_TOKEN_TYPE_LITERAL;
      token->literal.type     = LUM_LITERAL_TYPE_BOOL;
      token->literal.val_bool = false;
      *consumed_chars         = num_chars;
      return true;
    }

    for (uint32_t enum_type = 0; enum_type < LUM_BUILTIN_ENUM_COUNT; enum_type++) {
      const LumBuiltinEnumValuePair* pair = lum_builtin_enums + enum_type;

      const size_t string_length = strlen(pair->string);

      if (string_length != num_chars)
        continue;

      if (memcmp(pair->string, code, num_chars) == 0) {
        token->type             = LUM_TOKEN_TYPE_LITERAL;
        token->literal.type     = LUM_LITERAL_TYPE_ENUM;
        token->literal.val_enum = pair->value;
        *consumed_chars         = num_chars;
        return true;
      }
    }
  }

  return false;
}

static bool _lum_tokenizer_is_operator(LumToken* token, const char* code, uint32_t* consumed_chars) {
  switch (*code) {
    case '=':
      token->type          = LUM_TOKEN_TYPE_OPERATOR;
      token->operator.type = LUM_OPERATOR_TYPE_ASSIGNMENT;
      *consumed_chars      = 1;
      break;
    default:
      break;
  }

  return token->type != LUM_TOKEN_TYPE_INVALID;
}

static bool _lum_tokenizer_is_separator(LumToken* token, const char* code, uint32_t* consumed_chars) {
  switch (*code) {
    case ';':
      token->type           = LUM_TOKEN_TYPE_SEPARATOR;
      token->separator.type = LUM_SEPARATOR_TYPE_EOL;
      *consumed_chars       = 1;
      break;
    case '[':
      token->type           = LUM_TOKEN_TYPE_SEPARATOR;
      token->separator.type = LUM_SEPARATOR_TYPE_FUNC_BEGIN;
      *consumed_chars       = 1;
      break;
    case ']':
      token->type           = LUM_TOKEN_TYPE_SEPARATOR;
      token->separator.type = LUM_SEPARATOR_TYPE_FUNC_END;
      *consumed_chars       = 1;
      break;
    case '.':
      token->type           = LUM_TOKEN_TYPE_SEPARATOR;
      token->separator.type = LUM_SEPARATOR_TYPE_MEMBER;
      *consumed_chars       = 1;
      break;
    default:
      break;
  }

  return token->type != LUM_TOKEN_TYPE_INVALID;
}

LuminaryResult lum_tokenizer_execute(LumTokenizer* tokenizer, const char* code) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(code);

  while (*code != '\0') {
    // Skip whitespace
    if (*code == ' ' || *code == '\n' || *code == '\r') {
      code++;
      continue;
    }

    LumToken token          = (LumToken) {.type = LUM_TOKEN_TYPE_INVALID};
    uint32_t consumed_chars = 0;

    if (_lum_tokenizer_is_operator(&token, code, &consumed_chars)) {
      __FAILURE_HANDLE(array_push(&tokenizer->tokens, &token));
      code += consumed_chars;
      continue;
    }

    if (_lum_tokenizer_is_separator(&token, code, &consumed_chars)) {
      __FAILURE_HANDLE(array_push(&tokenizer->tokens, &token));
      code += consumed_chars;
      continue;
    }

    if (_lum_tokenizer_is_keyword(&token, code, &consumed_chars)) {
      __FAILURE_HANDLE(array_push(&tokenizer->tokens, &token));
      code += consumed_chars;
      continue;
    }

    if (_lum_tokenizer_is_literal(&token, code, &consumed_chars)) {
      __FAILURE_HANDLE(array_push(&tokenizer->tokens, &token));
      code += consumed_chars;
      continue;
    }

    if (_lum_tokenizer_is_identifier(&token, code, &consumed_chars)) {
      __FAILURE_HANDLE(array_push(&tokenizer->tokens, &token));
      code += consumed_chars;
      continue;
    }

    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Encountered invalid token in lum file: %s", code);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_tokenizer_print(LumTokenizer* tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  uint32_t num_tokens;
  __FAILURE_HANDLE(array_get_num_elements(tokenizer->tokens, &num_tokens));

  for (uint32_t token_id = 0; token_id < num_tokens; token_id++) {
    LumToken token = tokenizer->tokens[token_id];

    switch (token.type) {
      case LUM_TOKEN_TYPE_IDENTIFIER:
        info_message(
          "[IDENTIFIER] %s (%u)", token.identifier.name,
          (token.identifier.is_builtin_type) ? (uint32_t) token.identifier.builtin_type : 0xFFFFFFFF);
        break;
      case LUM_TOKEN_TYPE_KEYWORD:
        info_message("[KEYWORD]");
        break;
      case LUM_TOKEN_TYPE_LITERAL:
        switch (token.literal.type) {
          case LUM_LITERAL_TYPE_FLOAT:
            info_message("[LITERAL] Float %ff", token.literal.val_float);
            break;
          case LUM_LITERAL_TYPE_DOUBLE:
            info_message("[LITERAL] Double %f", token.literal.val_double);
            break;
          case LUM_LITERAL_TYPE_UINT:
            info_message("[LITERAL] Uint %u", token.literal.val_uint);
            break;
          case LUM_LITERAL_TYPE_BOOL:
            info_message("[LITERAL] Bool %s", token.literal.val_bool ? "True" : "False");
            break;
          case LUM_LITERAL_TYPE_ENUM:
            info_message("[LITERAL] Enum %u", token.literal.val_enum);
            break;
          default:
            info_message("[LITERAL] Unknown");
            break;
        }
        break;
      case LUM_TOKEN_TYPE_OPERATOR:
        switch (token.operator.type) {
          case LUM_OPERATOR_TYPE_ASSIGNMENT:
            info_message("[OPERATOR] =");
            break;
          default:
            info_message("[OPERATOR] Unknown");
            break;
        }
        break;
      case LUM_TOKEN_TYPE_SEPARATOR:
        switch (token.separator.type) {
          case LUM_SEPARATOR_TYPE_EOL:
            info_message("[SEPARATOR] EOL");
            break;
          case LUM_SEPARATOR_TYPE_FUNC_BEGIN:
            info_message("[SEPARATOR] Func Begin");
            break;
          case LUM_SEPARATOR_TYPE_FUNC_END:
            info_message("[SEPARATOR] Func End");
            break;
          case LUM_SEPARATOR_TYPE_MEMBER:
            info_message("[SEPARATOR] Member");
            break;
          default:
            info_message("[SEPARATOR] Unknown");
            break;
        }
        break;
      default:
        info_message("[UNKNOWN]");
        break;
    }
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_tokenizer_destroy(LumTokenizer** tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  __FAILURE_HANDLE(array_destroy(&(*tokenizer)->tokens));

  __FAILURE_HANDLE(host_free(tokenizer));

  return LUMINARY_SUCCESS;
}
