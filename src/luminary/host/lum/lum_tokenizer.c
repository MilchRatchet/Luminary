#include "lum_tokenizer.h"

#include <stdlib.h>
#include <string.h>

#include "internal_error.h"

static const char _lum_separator_chars[LUM_SEPARATOR_COUNT] = {
  [LUM_SEPARATOR_TYPE_EOL]               = ';',
  [LUM_SEPARATOR_TYPE_ACCESS_BEGIN]      = '[',
  [LUM_SEPARATOR_TYPE_ACCESS_END]        = ']',
  [LUM_SEPARATOR_TYPE_MEMBER]            = '.',
  [LUM_SEPARATOR_TYPE_LIST]              = ',',
  [LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN] = '{',
  [LUM_SEPARATOR_TYPE_INITIALIZER_END]   = '}'};

LuminaryResult lum_tokenizer_create(LumTokenizer** tokenizer) {
  __CHECK_NULL_ARGUMENT(tokenizer);

  __FAILURE_HANDLE(host_malloc(tokenizer, sizeof(LumTokenizer)));
  memset(*tokenizer, 0, sizeof(LumTokenizer));

  (*tokenizer)->allocated_string_mem = 4096;
  __FAILURE_HANDLE(host_malloc(&(*tokenizer)->string_mem, (*tokenizer)->allocated_string_mem));

  return LUMINARY_SUCCESS;
}

static uint32_t _lum_identifier_get_num_chars(const char* code, const bool for_literal) {
  const char* next_space            = strchr(code, ' ');
  const char* next_newline          = strchr(code, '\n');
  const char* next_charriage_return = strchr(code, '\r');

  uint32_t num_chars = UINT32_MAX;

  num_chars = (next_space) ? min(next_space - code, num_chars) : num_chars;
  num_chars = (next_newline) ? min(next_newline - code, num_chars) : num_chars;
  num_chars = (next_charriage_return) ? min(next_charriage_return - code, num_chars) : num_chars;

  for (uint32_t separator_id = 0; separator_id < LUM_SEPARATOR_COUNT; separator_id++) {
    // Member separator is special as it is allowed to be part of a literal
    if (for_literal && separator_id == LUM_SEPARATOR_TYPE_MEMBER)
      continue;

    const char* next_separator = strchr(code, _lum_separator_chars[separator_id]);

    num_chars = (next_separator) ? min(next_separator - code, num_chars) : num_chars;
  }

  if (num_chars == UINT32_MAX) {
    num_chars = (uint32_t) strlen(code);
  }

  return num_chars;
}

static uint32_t _lum_literal_parse_string(LumTokenizer* tokenizer, LumToken* token) {
  const char* code = tokenizer->code + tokenizer->read_offset;

  // Verify begin of string
  if (code[0] != '\"')
    return 0;

  uint32_t read_chars   = 1;
  uint32_t write_offset = 0;

  while (code[read_chars] != '\0') {
    // Carriage return
    if (code[read_chars] == '\r') {
      read_chars++;
      continue;
    }

    // Newline
    if (code[read_chars] == '\n') {
      read_chars++;
      continue;
    }

    // String End
    if (code[read_chars] == '\"') {
      read_chars++;
      break;
    }

    // Escape Sequence
    if (code[read_chars] == '\\') {
      read_chars++;

      if (write_offset == tokenizer->allocated_string_mem) {
        tokenizer->allocated_string_mem *= 2;
        __FAILURE_HANDLE(host_realloc(&tokenizer->string_mem, tokenizer->allocated_string_mem));
      }

      switch (code[read_chars]) {
        case '0':
          tokenizer->string_mem[write_offset++] = '\0';
          break;
        case 'n':
          tokenizer->string_mem[write_offset++] = '\n';
          break;
        case 't':
          tokenizer->string_mem[write_offset++] = '\t';
          break;
        case 'r':
          tokenizer->string_mem[write_offset++] = '\r';
          break;
        default:
          break;
      }

      read_chars++;
      continue;
    }

    if (write_offset == tokenizer->allocated_string_mem) {
      tokenizer->allocated_string_mem *= 2;
      __FAILURE_HANDLE(host_realloc(&tokenizer->string_mem, tokenizer->allocated_string_mem));
    }

    tokenizer->string_mem[write_offset++] = code[read_chars];

    read_chars++;
  }

  tokenizer->string_mem[write_offset++] = '\0';

  token->literal.val_string = (LumTokenLiteralString) {.data = tokenizer->string_mem, .length = write_offset - 1};

  return read_chars;
}

static bool _lum_tokenizer_is_identifier(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  const char* code = tokenizer->code + tokenizer->read_offset;

  const uint32_t num_chars = _lum_identifier_get_num_chars(code, false);

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

static bool _lum_tokenizer_is_keyword(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  LUM_UNUSED(tokenizer);
  LUM_UNUSED(token);

  *consumed_chars = 0;

  return false;
}

static bool _lum_tokenizer_is_literal(LumTokenizer* tokenizer, LumToken* token, uint32_t* consumed_chars) {
  if (tokenizer->code[tokenizer->read_offset] == '\"') {
    token->type         = LUM_TOKEN_TYPE_LITERAL;
    token->literal.type = LUM_LITERAL_TYPE_STRING;
    *consumed_chars     = _lum_literal_parse_string(tokenizer, token);
    return true;
  }

  const char* code = tokenizer->code + tokenizer->read_offset;

  const uint32_t num_chars = _lum_identifier_get_num_chars(code, true);

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

    return false;
  }

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

  return false;
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

  tokenizer->code        = code;
  tokenizer->read_offset = 0;
  tokenizer->line        = 3;
  tokenizer->col         = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_tokenizer_parse_next_token(LumTokenizer* tokenizer, LumToken* token) {
  __CHECK_NULL_ARGUMENT(tokenizer);
  __CHECK_NULL_ARGUMENT(token);

  // Skip special characters.
  while (true) {
    const char current_char = tokenizer->code[tokenizer->read_offset];

    // EOF
    if (current_char == '\0') {
      token->type = LUM_TOKEN_TYPE_EOF;
      return LUMINARY_SUCCESS;
    }

    // Whitespace
    if (current_char == ' ') {
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
      char comment_char;
      do {
        comment_char = tokenizer->code[tokenizer->read_offset++];
      } while (comment_char != '\0' && comment_char != '\n');

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
        case LUM_LITERAL_TYPE_DOUBLE:
          info_message("[%04u:%04u][LITERAL] Double %f", token.line, token.col, token.literal.val_double);
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
        case LUM_SEPARATOR_TYPE_EOL:
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
