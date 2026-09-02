#ifndef LEXER_HPP
#define LEXER_HPP

#include <iostream>
#include <string>
#include <cctype>
#include <cstdlib>
#include <cstdio>
enum Token
{
    tok_eof = -1,
    
    // commands
    tok_def = -2,
    tok_extern = -3,

    // primary
    tok_identifier = -4,
    tok_number = -5,

    // control
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,

    // for loop
    tok_for = -9,
    tok_in = -10,

    // operator
    tok_binary = -11,
    tok_unary = -12,

    // var
    tok_var = -13
};

extern std::string IdentifierStr;
extern double NumVal;
extern FILE *SourceFile;

int gettok();
#endif