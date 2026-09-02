#include "AST.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "codegen.hpp"
#include "JITLib.hpp"

extern "C" DLLEXPORT double putchard(double X)
{
    fputc((char)X, stderr);
    return 0;
}

extern "C" DLLEXPORT double printd(double X)
{
    fprintf(stderr, "%f\n", X);
    return 0;
}