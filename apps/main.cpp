#include "lexer.hpp"
#include "AST.hpp"
#include "parser.hpp"
#include "TopLevel.hpp"
#include "codegen.hpp"
#include "JITLib.hpp"

int main(int argc, char **argv)
{
    /*
    Just for lexer part:
    int Tok;
    do
    {
        Tok = gettok();
        std::cout << Tok << std::endl;
    }while(Tok!=tok_eof);
    return 0;
    */
    for(int i=1; i<argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "off")
        {
            EnableOptimization = false;
            fprintf(stderr, "Optimization disabled.\n");
        }
        else
        {
            SourceFile = fopen(arg.c_str(), "r");
            if(!SourceFile)
            {
                fprintf(stderr, "Could not open file: %s\n", arg.c_str());
                return 1;
            }
        }
    }

    /*
    if(argc>1 && std::string(argv[1]) == "off")
    {
        EnableOptimization = false;
        fprintf(stderr, "Optimization disabled.\n");
    }
    */
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    BinopPrecedence['='] = 2;
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 30;
    BinopPrecedence['*'] = 40;

    fprintf(stderr, "ready> ");
    getNextToken();

    TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
    auto G = cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            TheJIT->getDataLayout().getGlobalPrefix()
        )
    );
    TheJIT->getMainJITDylib().addGenerator(std::move(G));
    
    InitializeModuleAndManagers();
   
    MainLoop();
   
    TheModule->print(errs(), nullptr);

    return 0; 
}