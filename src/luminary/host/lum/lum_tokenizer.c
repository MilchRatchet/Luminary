#include "lum_tokenizer.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal_error.h"

const LumBuiltinType lum_tokenizer_literal_type_to_builtin[LUM_LITERAL_TYPE_COUNT] = {
  [LUM_LITERAL_TYPE_FLOAT]  = LUM_BUILTIN_TYPE_FLOAT,
  [LUM_LITERAL_TYPE_UINT]   = LUM_BUILTIN_TYPE_UINT,
  [LUM_LITERAL_TYPE_BOOL]   = LUM_BUILTIN_TYPE_BOOL,
  [LUM_LITERAL_TYPE_ENUM]   = LUM_BUILTIN_TYPE_ENUM,
  [LUM_LITERAL_TYPE_STRING] = LUM_BUILTIN_TYPE_STRING};

static const char _lum_separator_chars[LUM_SEPARATOR_COUNT] = {
  [LUM_SEPARATOR_TYPE_STATEMENT_END] = ';',  [LUM_SEPARATOR_TYPE_ACCESS_BEGIN] = '[', [LUM_SEPARATOR_TYPE_ACCESS_END] = ']',
  [LUM_SEPARATOR_TYPE_MEMBER] = '.',         [LUM_SEPARATOR_TYPE_LIST] = ',',         [LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN] = '{',
  [LUM_SEPARATOR_TYPE_INITIALIZER_END] = '}'};

LuminaryResult lum_tokenizer_create(LumTokenizer** tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  __FAILURE_HANDLE(host_malloc(tokenizer, sizeof(LumTokenizer)));
  memset(*tokenizer, 0, sizeof(LumTokenizer));

  (*tokenizer)->allocated_string_mem = 4096;
  __FAILURE_HANDLE(host_malloc(&(*tokenizer)->string_mem, (*tokenizer)->allocated_string_mem));

  return LUMINARY_SUCCESS;
}

static bool _lum_tokenizer_starts_number(const char* code) {
  return (code[0] >= '0' && code[0] <= '9') || (code[0] == '-' && code[1] >= '0' && code[1] <= '9');
}

static uint32_t _lum_identifier_get_num_chars(const char* code, const bool for_literal) {
  const bool starts_number    = for_literal && _lum_tokenizer_starts_number(code);
  const bool is_hex           = starts_number && code[0] == '0' && (code[1] == 'x' || code[1] == 'X');
  bool decimal_separator_seen = false;
  bool exponent_seen          = false;
  uint32_t num_chars          = 0;

  while (code[num_chars] != '\0') {
    const char current_char = code[num_chars];

    if (current_char == ' ' || current_char == '\t' || current_char == '\n' || current_char == '\r' || current_char == '=') {
      break;
    }

    if (current_char == '.' && starts_number && !decimal_separator_seen && !exponent_seen) {
      decimal_separator_seen = true;
      num_chars++;
      continue;
    }

    if (is_hex == false && (current_char == 'e' || current_char == 'E')) {
      exponent_seen = true;
    }

    bool is_separator = false;
    for (uint32_t separator_id = 0; separator_id < LUM_SEPARATOR_COUNT; separator_id++) {
      if (current_char == _lum_separator_chars[separator_id]) {
        is_separator = true;
        break;
      }
    }

    if (is_separator) {
      break;
    }

    num_chars++;
  }

  return num_chars;
}

static LuminaryResult _lum_tokenizer_string_append(LumTokenizer* tokenizer, size_t* write_offset, char value) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(write_offset);

  if (*write_offset >= tokenizer->allocated_string_mem) {
    if (tokenizer->allocated_string_mem > (size_t) -1 / 2) {
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Luminary string literal is too large.");
    }

    tokenizer->allocated_string_mem *= 2;
    __FAILURE_HANDLE(host_realloc(&tokenizer->string_mem, tokenizer->allocated_string_mem));
  }

  tokenizer->string_mem[(*write_offset)++] = value;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_literal_parse_string(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(token);
  __CHECK_NULL_ARGUMENT(consumed_chars);

  const char* code    = tokenizer->code + tokenizer->read_offset;
  uint32_t read_chars = 1;
  size_t write_offset = 0;
  bool closed         = false;

  while (code[read_chars] != '\0') {
    char value = code[read_chars++];

    if (value == '"') {
      closed = true;
      break;
    }

    if (value == '\r' || value == '\n') {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Newlines in Luminary string literals must be escaped.");
    }

    if (value == '\\') {
      value = code[read_chars++];

      switch (value) {
        case '\\':
        case '"':
          break;
        case 'n':
          value = '\n';
          break;
        case 'r':
          value = '\r';
          break;
        case 't':
          value = '\t';
          break;
        case '\0':
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Unterminated escape sequence in Luminary string literal.");
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Invalid escape sequence in Luminary string literal.");
      }
    }

    __FAILURE_HANDLE(_lum_tokenizer_string_append(tokenizer, &write_offset, value));
  }

  if (closed == false) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Unterminated Luminary string literal.");
  }

  const size_t string_length = write_offset;
  __FAILURE_HANDLE(_lum_tokenizer_string_append(tokenizer, &write_offset, '\0'));

  token->literal.val_string = (LumTokenLiteralString) {.data = tokenizer->string_mem, .length = (uint32_t) string_length};
  *consumed_chars           = read_chars;

  return LUMINARY_SUCCESS;
}

static bool _lum_tokenizer_is_identifier(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  const char* code         = tokenizer->code + tokenizer->read_offset;
  const uint32_t num_chars = _lum_identifier_get_num_chars(code, false);

  if (num_chars >= LUM_IDENTIFIER_MAX_LENGTH)
    return false;

  *consumed_chars = num_chars;
  token->type     = LUM_TOKEN_TYPE_IDENTIFIER;
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

static bool _lum_tokenizer_is_keyword(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  LUM_UNUSED(tokenizer);
  LUM_UNUSED(token);

  const char* code = tokenizer->code + tokenizer->read_offset;

  const uint32_t num_chars = _lum_identifier_get_num_chars(code, false);

  if (num_chars >= LUM_IDENTIFIER_MAX_LENGTH - 1)
    return false;

  *consumed_chars = 0;

  return false;
}

static bool _lum_tokenizer_is_literal(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(token);
  __CHECK_NULL_ARGUMENT(consumed_chars);

  const char* code = tokenizer->code + tokenizer->read_offset;
  if (code[0] == '"') {
    token->type         = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type = LUM_LITERAL_TYPE_STRING;

    return _lum_literal_parse_string(tokenizer, token, consumed_chars) == LUMINARY_SUCCESS;
  }

  const uint32_t num_chars = _lum_identifier_get_num_chars(code, true);
  if (num_chars >= LUM_IDENTIFIER_MAX_LENGTH)
    return LUMINARY_SUCCESS;

  if (num_chars == 4 && memcmp(code, "true", 4) == 0) {
    token->type             = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type     = LUM_LITERAL_TYPE_BOOL;
    token->literal.val_bool = true;
    *consumed_chars         = num_chars;
    return true;
  }

  if (num_chars == 5 && memcmp(code, "false", 5) == 0) {
    token->type             = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type     = LUM_LITERAL_TYPE_BOOL;
    token->literal.val_bool = false;
    *consumed_chars         = num_chars;
    return true;
  }

  for (uint32_t enum_type = 0; enum_type < LUM_BUILTIN_ENUM_COUNT; enum_type++) {
    const LumBuiltinEnumValuePair* pair = lum_builtin_enums + enum_type;
    const size_t string_length          = strlen(pair->string);

    if (string_length == num_chars && memcmp(pair->string, code, num_chars) == 0) {
      token->type             = LUM_TOKEN_TYPE_LITERAL;
      token->literal.type     = LUM_LITERAL_TYPE_ENUM;
      token->literal.val_enum = pair->value;
      *consumed_chars         = num_chars;
      return true;
    }
  }

  if (_lum_tokenizer_starts_number(code) == false)
    return false;

  char literal_string[LUM_IDENTIFIER_MAX_LENGTH];
  memcpy(literal_string, code, num_chars);
  literal_string[num_chars] = '\0';

  const bool is_hex = literal_string[0] == '0' && (literal_string[1] == 'x' || literal_string[1] == 'X');
  bool is_float     = literal_string[0] == '-';
  for (uint32_t index = 0; index < num_chars; index++) {
    if (literal_string[index] == '.' || (!is_hex && (literal_string[index] == 'e' || literal_string[index] == 'E'))) {
      is_float = true;
      break;
    }
  }

  if (is_float) {
    errno = 0;
    char* end;
    const float parsed_value = strtof(literal_string, &end);

    if (end == literal_string || *end != '\0' || !isfinite(parsed_value) || (errno == ERANGE && parsed_value == 0.0f))
      return false;

    token->type              = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type      = LUM_LITERAL_TYPE_FLOAT;
    token->literal.val_float = parsed_value;
  }
  else {
    errno = 0;
    char* end;
    const unsigned long parsed_value = strtoul(literal_string, &end, 0);

    if (end == literal_string || *end != '\0' || errno == ERANGE || parsed_value > 0xFFFFFFFFUL)
      return false;

    token->type             = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type     = LUM_LITERAL_TYPE_UINT;
    token->literal.val_uint = (uint32_t) parsed_value;
  }

  *consumed_chars = num_chars;
  return true;
}

static bool _lum_tokenizer_is_operator(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  const char current_char = tokenizer->code[tokenizer->read_offset];

  switch (current_char) {
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

static bool _lum_tokenizer_is_separator(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  const char current_char = tokenizer->code[tokenizer->read_offset];

  for (uint32_t separator_id = 0; separator_id < LUM_SEPARATOR_COUNT; separator_id++) {
    if (current_char == _lum_separator_chars[separator_id]) {
      token->type           = LUM_TOKEN_TYPE_SEPARATOR;
      token->separator.type = (LumSeparatorType) separator_id;
      *consumed_chars       = 1;
      return true;
    }
  }

  return false;
}

LuminaryResult lum_tokenizer_set_code(LumTokenizer* tokenizer, const char* code) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(code);

  tokenizer->code        = code;
  tokenizer->read_offset = 0;
  tokenizer->line        = 3;
  tokenizer->col         = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_tokenizer_parse_next_token(LumTokenizer* tokenizer, LumToken* token) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(token);
  __CHECK_NULL_ARGUMENT(tokenizer->code);

  // Skip special characters.
  while (true) {
    const char current_char = tokenizer->code[tokenizer->read_offset];

    // EOF
    if (current_char == '\0') {
      token->type = LUM_TOKEN_TYPE_EOF;
      return LUMINARY_SUCCESS;
    }

    // Whitespace
    if (current_char == ' ' || current_char == '\t') {
      tokenizer->col++;
      tokenizer->read_offset++;
      continue;
    }

    // Carriage return
    if (current_char == '\r') {
      tokenizer->read_offset++;
      continue;
    }

    // Newline
    if (current_char == '\n') {
      tokenizer->line++;
      tokenizer->col = 0;
      tokenizer->read_offset++;
      continue;
    }

    // Comments
    if (current_char == '#') {
      while (tokenizer->code[tokenizer->read_offset] != '\0' && tokenizer->code[tokenizer->read_offset] != '\n') {
        tokenizer->read_offset++;
        tokenizer->col++;
      }

      if (tokenizer->code[tokenizer->read_offset] == '\n') {
        tokenizer->read_offset++;
        tokenizer->line++;
        tokenizer->col = 0;
      }

      continue;
    }

    break;
  }

  *token = (LumToken) {.type = LUM_TOKEN_TYPE_INVALID, .line = tokenizer->line, .col = tokenizer->col};

  uint32_t consumed_chars = 0;

  if (_lum_tokenizer_is_operator(tokenizer, token, &consumed_chars)) {
    tokenizer->col += consumed_chars;
    tokenizer->read_offset += consumed_chars;
    return LUMINARY_SUCCESS;
  }

  if (_lum_tokenizer_is_separator(tokenizer, token, &consumed_chars)) {
    tokenizer->col += consumed_chars;
    tokenizer->read_offset += consumed_chars;
    return LUMINARY_SUCCESS;
  }

  if (_lum_tokenizer_is_keyword(tokenizer, token, &consumed_chars)) {
    tokenizer->col += consumed_chars;
    tokenizer->read_offset += consumed_chars;
    return LUMINARY_SUCCESS;
  }

  if (_lum_tokenizer_is_literal(tokenizer, token, &consumed_chars)) {
    tokenizer->col += consumed_chars;
    tokenizer->read_offset += consumed_chars;
    return LUMINARY_SUCCESS;
  }

  if (_lum_tokenizer_is_identifier(tokenizer, token, &consumed_chars)) {
    tokenizer->col += consumed_chars;
    tokenizer->read_offset += consumed_chars;
    return LUMINARY_SUCCESS;
  }

  __RETURN_ERROR(
    LUMINARY_ERROR_API_EXCEPTION, "Encountered invalid token at (%u:%u) in lum file:\n%s", tokenizer->line, tokenizer->col,
    tokenizer->code);
}

LuminaryResult lum_tokenizer_print(LumTokenizer* tokenizer, const LumToken token) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  switch (token.type) {
    case LUM_TOKEN_TYPE_IDENTIFIER:
      info_message(
        "[%04u:%04u][IDENTIFIER] %s (%u)", token.line, token.col, token.identifier.name,
        (token.identifier.is_builtin_type) ? (uint32_t) token.identifier.builtin_type : 0xFFFFFFFF);
      break;
    case LUM_TOKEN_TYPE_KEYWORD:
      info_message("[%04u:%04u][KEYWORD]", token.line, token.col);
      break;
    case LUM_TOKEN_TYPE_LITERAL:
      switch (token.literal.type) {
        case LUM_LITERAL_TYPE_FLOAT:
          info_message("[%04u:%04u][LITERAL] Float %ff", token.line, token.col, token.literal.val_float);
          break;
        case LUM_LITERAL_TYPE_UINT:
          info_message("[%04u:%04u][LITERAL] Uint %u", token.line, token.col, token.literal.val_uint);
          break;
        case LUM_LITERAL_TYPE_BOOL:
          info_message("[%04u:%04u][LITERAL] Bool %s", token.line, token.col, token.literal.val_bool ? "True" : "False");
          break;
        case LUM_LITERAL_TYPE_ENUM:
          info_message("[%04u:%04u][LITERAL] Enum %u", token.line, token.col, token.literal.val_enum);
          break;
        case LUM_LITERAL_TYPE_STRING:
          info_message("[%04u:%04u][LITERAL] String \"%s\"", token.line, token.col, token.literal.val_string.data);
          break;
        default:
          info_message("[%04u:%04u][LITERAL] Unknown", token.line, token.col);
          break;
      }
      break;
    case LUM_TOKEN_TYPE_OPERATOR:
      switch (token.operator.type) {
        case LUM_OPERATOR_TYPE_ASSIGNMENT:
          info_message("[%04u:%04u][OPERATOR] =", token.line, token.col);
          break;
        default:
          info_message("[%04u:%04u][OPERATOR] Unknown", token.line, token.col);
          break;
      }
      break;
    case LUM_TOKEN_TYPE_SEPARATOR:
      switch (token.separator.type) {
        case LUM_SEPARATOR_TYPE_STATEMENT_END:
          info_message("[%04u:%04u][SEPARATOR] EOL", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_ACCESS_BEGIN:
          info_message("[%04u:%04u][SEPARATOR] Reference Begin", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_ACCESS_END:
          info_message("[%04u:%04u][SEPARATOR] Reference End", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_MEMBER:
          info_message("[%04u:%04u][SEPARATOR] Member", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_LIST:
          info_message("[%04u:%04u][SEPARATOR] List", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN:
          info_message("[%04u:%04u][SEPARATOR] Initializer Begin", token.line, token.col);
          break;
        case LUM_SEPARATOR_TYPE_INITIALIZER_END:
          info_message("[%04u:%04u][SEPARATOR] Initializer End", token.line, token.col);
          break;
        default:
          info_message("[%04u:%04u][SEPARATOR] Unknown", token.line, token.col);
          break;
      }
      break;
    case LUM_TOKEN_TYPE_EOF:
      info_message("[%04u:%04u][EOF] End of file", token.line, token.col);
      break;
    default:
      info_message("[%04u:%04u][UNKNOWN]", token.line, token.col);
      break;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_tokenizer_destroy(LumTokenizer** tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(*tokenizer);

  __FAILURE_HANDLE(host_free(&(*tokenizer)->string_mem));
  __FAILURE_HANDLE(host_free(tokenizer));

  return LUMINARY_SUCCESS;
}
