#ifndef LUMINARY_LUM_TOKENIZER_H
#define LUMINARY_LUM_TOKENIZER_H

#include "lum_builtins.h"
#include "utils.h"

#define LUM_IDENTIFIER_MAX_LENGTH 256

////////////////////////////////////////////////////////////////////
// Lum Token Types
////////////////////////////////////////////////////////////////////

enum LumTokenType {
  LUM_TOKEN_TYPE_INVALID,
  LUM_TOKEN_TYPE_IDENTIFIER,
  LUM_TOKEN_TYPE_KEYWORD,
  LUM_TOKEN_TYPE_LITERAL,
  LUM_TOKEN_TYPE_OPERATOR,
  LUM_TOKEN_TYPE_SEPARATOR,
  LUM_TOKEN_TYPE_EOF,
  LUM_TOKEN_TYPE_COUNT
} typedef LumTokenType;

enum LumKeywordType {
  // Version 1
  LUM_KEYWORD_TYPE_COUNT_VERSION_1,

  LUM_KEYWORD_TYPE_COUNT = LUM_KEYWORD_TYPE_COUNT_VERSION_1
} typedef LumKeywordType;

enum LumLiteralType {
  // Version 1
  LUM_LITERAL_TYPE_FLOAT,
  LUM_LITERAL_TYPE_UINT,
  LUM_LITERAL_TYPE_BOOL,
  LUM_LITERAL_TYPE_ENUM,
  LUM_LITERAL_TYPE_STRING,
  LUM_LITERAL_TYPE_COUNT_VERSION_1,

  LUM_LITERAL_TYPE_COUNT = LUM_LITERAL_TYPE_COUNT_VERSION_1
} typedef LumLiteralType;

enum LumOperatorType {
  // Version 1
  LUM_OPERATOR_TYPE_ASSIGNMENT,
  LUM_OPERATOR_TYPE_COUNT_VERSION_1,

  LUM_OPERATOR_TYPE_COUNT = LUM_OPERATOR_TYPE_COUNT_VERSION_1
} typedef LumOperatorType;

enum LumSeparatorType {
  // Version 1
  LUM_SEPARATOR_TYPE_STATEMENT_END,      // ;
  LUM_SEPARATOR_TYPE_ACCESS_BEGIN,       // [
  LUM_SEPARATOR_TYPE_ACCESS_END,         // ]
  LUM_SEPARATOR_TYPE_MEMBER,             // .
  LUM_SEPARATOR_TYPE_LIST,               // ,
  LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN,  // {
  LUM_SEPARATOR_TYPE_INITIALIZER_END,    // }
  LUM_SEPARATOR_COUNT_VERSION_1,

  LUM_SEPARATOR_COUNT = LUM_SEPARATOR_COUNT_VERSION_1
} typedef LumSeparatorType;

////////////////////////////////////////////////////////////////////
// Lum Token Implementations
////////////////////////////////////////////////////////////////////

struct LumTokenIdentifier {
  bool is_builtin_type;
  LumBuiltinType builtin_type;
  char name[LUM_IDENTIFIER_MAX_LENGTH];
} typedef LumTokenIdentifier;

struct LumTokenKeyword {
  LumKeywordType type;
} typedef LumTokenKeyword;

struct LumTokenLiteralString {
  char* data;
  uint32_t length;
} typedef LumTokenLiteralString;

struct LumTokenLiteral {
  LumLiteralType type;
  union {
    float val_float;
    uint32_t val_uint;
    bool val_bool;
    uint32_t val_enum;
    LumTokenLiteralString val_string;
  };
} typedef LumTokenLiteral;

struct LumTokenOperator {
  LumOperatorType type;
} typedef LumTokenOperator;

struct LumTokenSeparator {
  LumSeparatorType type;
} typedef LumTokenSeparator;

////////////////////////////////////////////////////////////////////
// Lum Token
////////////////////////////////////////////////////////////////////

struct LumToken {
  LumTokenType type;
  union {
    LumTokenIdentifier identifier;
    LumTokenKeyword keyword;
    LumTokenLiteral literal;
    LumTokenOperator operator;
    LumTokenSeparator separator;
  };

  uint32_t line;
  uint32_t col;
} typedef LumToken;

struct LumTokenizer {
  const char* code;
  uint32_t read_offset;
  uint32_t line;
  uint32_t col;
  char* string_mem;
  uint32_t allocated_string_mem;
} typedef LumTokenizer;

LuminaryResult lum_tokenizer_create(LumTokenizer** tokenizer);
LuminaryResult lum_tokenizer_set_code(LumTokenizer* tokenizer, const char* code);
LuminaryResult lum_tokenizer_parse_next_token(LumTokenizer* tokenizer, LumToken* token);
LuminaryResult lum_tokenizer_print(LumTokenizer* tokenizer, const LumToken token);
LuminaryResult lum_tokenizer_destroy(LumTokenizer** tokenizer);

extern const LumBuiltinType lum_tokenizer_literal_type_to_builtin[LUM_LITERAL_TYPE_COUNT];

#endif /* LUMINARY_LUM_TOKENIZER_H */
