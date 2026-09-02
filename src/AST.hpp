#ifndef AST_HPP
#define AST_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <cassert>

namespace llvm
{
    class Value;
    class Function;
}

using namespace llvm;

class ExprAST
{
    public:
        virtual ~ExprAST() = default;
        virtual Value *codegen() = 0;
};

class NumberExprAST : public ExprAST
{
    double Val;
    public:
        NumberExprAST(double Val) : Val(Val){}

        Value *codegen() override;
};

class VariableExprAST : public ExprAST
{
    std::string Name;
    public:
        VariableExprAST(const std::string &Name) : Name(Name){}
        const std::string &getName() const
        {
            return Name;
        }
        Value *codegen() override;
};

class UnaryExprAST : public ExprAST
{
    char Opcode;
    std::unique_ptr<ExprAST> Operand;

    public:
        UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand) : 
            Opcode(Opcode), Operand(std::move(Operand)) {}

        Value *codegen() override;
};

class BinaryExprAST : public ExprAST
{
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
    
    public:
        BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS) :
            Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)){}

        Value *codegen() override;
};

class CallExprAST : public ExprAST
{
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

    public:
        CallExprAST(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> Args) :
            Callee(Callee), Args(std::move(Args)){}

        Value *codegen() override;
};

class IfExprAST : public ExprAST
{
    std::unique_ptr<ExprAST> Cond, Then, Else;

    public:
        IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then, std::unique_ptr<ExprAST> Else) :
            Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)){}
        
        Value *codegen() override;
};

class ForExprAST : public ExprAST
{
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;

    public:
        ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start, std::unique_ptr<ExprAST> End,
                   std::unique_ptr<ExprAST> Step, std::unique_ptr<ExprAST> Body) : 
            VarName(VarName), Start(std::move(Start)), End(std::move(End)), Step(std::move(Step)), Body(std::move(Body)){}
        
        Value *codegen() override;
};

class VarExprAST : public ExprAST
{
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::unique_ptr<ExprAST> Body;

    public:
        VarExprAST(std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames, std::unique_ptr<ExprAST> Body) : 
            VarNames(std::move(VarNames)), Body(std::move(Body)) {}
        
        Value *codegen() override;
};

class PrototypeAST
{
    std::string Name;
    std::vector<std::string> Args;
    bool IsOperator;
    unsigned Precedence;

    public:
        PrototypeAST(const std::string &Name, std::vector<std::string> Args,
                     bool IsOperator=false, unsigned Prec=0) :
            Name(Name), Args(std::move(Args)), IsOperator(IsOperator), Precedence(Prec) {}
        std::string getName()
        {
            return Name;
        }
        
        bool isUnaryOp() const 
        {
            return IsOperator && Args.size() == 1;
        }

        bool isBinaryOp() const
        {
            return IsOperator && Args.size() == 2;
        }

        char getOperatorName() const
        {
            assert(isUnaryOp() || isBinaryOp());
            return Name[Name.size()-1];
        }
        unsigned getBinaryPrecedence() const {return Precedence;}

    Function *codegen();
};

class FunctionAST
{
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Body) :
            Proto(std::move(Proto)), Body(std::move(Body)){}
    
    Function *codegen();
};

#endif