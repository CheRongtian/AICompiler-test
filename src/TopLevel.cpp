#include "AST.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "TopLevel.hpp"
#include "codegen.hpp"

void InitializeModuleAndManagers()
{
    TheContext = std::make_unique<LLVMContext>();
    TheModule = std::make_unique<Module>("Kaleidoscope", *TheContext);
    TheModule->setDataLayout(TheJIT->getDataLayout());
    Builder = std::make_unique<IRBuilder<>>(*TheContext);

    TheFPM = std::make_unique<FunctionPassManager>();
    TheLAM = std::make_unique<LoopAnalysisManager>();
    TheFAM = std::make_unique<FunctionAnalysisManager>();
    TheCGAM = std::make_unique<CGSCCAnalysisManager>();
    TheMAM = std::make_unique<ModuleAnalysisManager>();
    ThePIC = std::make_unique<PassInstrumentationCallbacks>();
    TheSI = std::make_unique<StandardInstrumentations>(*TheContext, true);
    TheSI->registerCallbacks(*ThePIC, TheMAM.get());

    TheFPM->addPass(PromotePass()); // Mem2Reg
    TheFPM->addPass(InstCombinePass());
    TheFPM->addPass(ReassociatePass());
    TheFPM->addPass(GVNPass());
    TheFPM->addPass(SimplifyCFGPass());

    PassBuilder PB;
    PB.registerModuleAnalyses(*TheMAM);
    PB.registerFunctionAnalyses(*TheFAM);
    PB.crossRegisterProxies(*TheLAM, *TheFAM, *TheCGAM, *TheMAM);
}

void HandleDefinition()
{
    if(auto FnAST = ParseDefinition()) 
    {
        if(auto *FnIR = FnAST->codegen())
        {
            fprintf(stderr, "Parsed a function defition.");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            ExitOnErr(TheJIT->addModule(ThreadSafeModule(std::move(TheModule), std::move(TheContext))));
            InitializeModuleAndManagers();
        }  
    }
    else getNextToken();
}

void HandleExtern()
{
    if(auto ProtoAST = ParseExtern())
    {
        if(auto *FnIR = ProtoAST->codegen())
        {
            fprintf(stderr, "Read extern.");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
    }
    else getNextToken();
}

void HandleTopLevelExpression()
{
    if(auto FnAST = ParseTopLevelExpr())
    {
        /*
        if(auto *FnIR = FnAST->codegen())
        {
            fprintf(stderr, "Read top-level expression:");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            // remove an anonymous expression
            FnIR->eraseFromParent();
        }
        */
        if(FnAST->codegen())
        {
            auto RT = TheJIT->getMainJITDylib().createResourceTracker();
            auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
            ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
            InitializeModuleAndManagers();

            auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));
            double(*FP)() = ExprSymbol.toPtr<double(*)()>();
            fprintf(stderr, "Evaluated to %f\n", FP());

            ExitOnErr(RT->remove());
        }
    }
    else getNextToken();
}

// top ::= definition | external | expression | ';'
void MainLoop()
{
    while(true)
    {
        fprintf(stderr, "ready> ");
        switch(CurTok)
        {
            case tok_eof: return;
            case ';': getNextToken(); break;
            case tok_def: HandleDefinition(); break;
            case tok_extern: HandleExtern(); break;
            default: HandleTopLevelExpression(); break;
        }
    }    
}

