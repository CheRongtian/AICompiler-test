#include "lexer.hpp"

std::string IdentifierStr;
double NumVal;

FILE *SourceFile = stdin;

int gettok()
{
    static int LastChar = ' ';
    // skip white
    //while(isspace(LastChar)) LastChar = getchar();
    while(isspace(LastChar)) LastChar = fgetc(SourceFile);

    // keyword & identifier
    if(isalpha(LastChar))
    {
        IdentifierStr = LastChar;
        //while(isalnum(LastChar = getchar())) IdentifierStr += LastChar;
        while(isalnum(LastChar = fgetc(SourceFile))) IdentifierStr += LastChar;
        // keyword
        if(IdentifierStr == "def") return tok_def;
        if(IdentifierStr == "extern") return tok_extern;
        if(IdentifierStr == "if") return tok_if;
        if(IdentifierStr == "then") return tok_then;
        if(IdentifierStr == "else") return tok_else;
        if(IdentifierStr == "for") return tok_for;
        if(IdentifierStr == "in") return tok_in;
        if(IdentifierStr == "binary") return tok_binary;
        if(IdentifierStr == "unary") return tok_unary;
        if(IdentifierStr == "var") return tok_var;
        
        // identifier
        return tok_identifier; 
    }

    // number
    if(isdigit(LastChar) || LastChar == '.')
    {
        std::string NumStr;
        do
        {
            NumStr += LastChar;
            //LastChar = getchar(); 
            LastChar = fgetc(SourceFile);
        }while(isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), 0);
        return tok_number;
    }

    // comment
    if(LastChar == '#')
    {
        //do LastChar = getchar();
        do LastChar = fgetc(SourceFile);
        while(LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if(LastChar != EOF) return gettok();
    }

    // end
    if(LastChar == EOF) return tok_eof;

    // other character
    int ThisChar = LastChar;
    //LastChar = getchar();
    LastChar = fgetc(SourceFile);
    return ThisChar;
}
