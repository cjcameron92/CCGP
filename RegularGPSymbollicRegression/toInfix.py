#Assignment: 1
#Date: 2/16/2024
#Names: Cole Corbett, Cameron Carvalho
#Student ID: 7246325, 7240450
#Emails: cc21gg@brocku.ca, cc21lz@brocku.ca

'''
This file is a utility class for converting a functional expression to infix notation.
'''

import ast

'''
@param expression: The functional expression to be converted to infix notation
@return: The infix notation of the functional expression
'''
def parse_functional_expression(expression):
    # Define a custom visitor for the parser
    class FunctionalVisitor(ast.NodeVisitor): #Class to visit the nodes of the abstract syntax tree
        def generic_visit(self, node):#Raises an error if the node is not supported
            raise SyntaxError(f"Unsupported expression: {ast.dump(node)}")

        def visit_Call(self, node):#Visits the call node
            func_name = node.func.id#Gets the function name
            if func_name not in {'add', 'mul', 'sin', 'cos', 'protectedDiv', 'neg', 'rand101', 'sub', 'if_then_else', 'protectedLog'}:
                raise SyntaxError(f"Unsupported function: {func_name}")#Raises an error if the function is not supported

            args = [self.visit(arg) for arg in node.args]#Visits the arguments of the function
            return (func_name, *args)#returns the function name and its arguments

        def visit_Name(self, node): #Visits the node
            return node.id #returns the name

        def visit_Num(self, node):#Visits the node
            return node.n #returns the number

        def visit_UnaryOp(self, node):#Visits the node
            op = node.op #Gets the operator
            operand = self.visit(node.operand)#Visits the operand
            if isinstance(op, ast.USub):#If the operator is a unary subtraction
                return ('neg', operand)#returns the negative of the operand
            else:#If the operator is not supported
                raise SyntaxError(f"Unsupported unary operation: {ast.dump(node)}")#Raises an error

    try:#Tries to parse the expression
        parsed_expression = ast.parse(expression, mode='eval')
        visitor = FunctionalVisitor()
        return visitor.visit(parsed_expression.body)#Returns the parsed expression
    except SyntaxError as e:#If the expression is not supported
        print(f"Error parsing expression: {e}")
        return None


'''
@param expression: The functional expression to be converted to infix notation
@return: The infix notation of the functional expression
a recursive function that converts a functional expression to infix notation
'''
def functional_to_infix(expression):
    if isinstance(expression, tuple):#If the expression is a tuple
        operator, *operands = expression#Gets the operator and its operands
        if operator == 'add':#If the operator is addition
            return f"({'+'.join(map(functional_to_infix, operands))})"#Returns the infix notation of the addition
        elif operator == 'mul':#If the operator is multiplication
            return f"({'*'.join(map(functional_to_infix, operands))})"#Returns the infix notation of the multiplication
        elif operator == 'sin':#If the operator is sin
            return f"sin({functional_to_infix(operands[0])})"#Returns the infix notation of the sin
        elif operator == 'cos':#If the operator is cos
            return f"cos({functional_to_infix(operands[0])})"#Returns the infix notation of the cos
        elif operator == 'protectedDiv':#If the operator is protected division
            return f"({functional_to_infix(operands[0])}/{functional_to_infix(operands[1])})"#Returns the infix notation of the protected division
        elif operator == 'neg':#If the operator is negation
            return f"(-{functional_to_infix(operands[0])})"#Returns the infix notation of the negation
        elif operator == 'rand101':#If the operator is random number
            return str(operands[0])#Returns the random number
        elif operator == 'sub':#If the operator is subtraction
            return f"({functional_to_infix(operands[0])}-{functional_to_infix(operands[1])})"#Returns the infix notation of the subtraction
        elif operator == 'if_then_else':#If the operator is if then else
            return f"if {functional_to_infix(operands[0])} then {functional_to_infix(operands[1])} else {functional_to_infix(operands[2])}"#Returns the infix notation of the if then else
        elif operator == 'protectedLog':#If the operator is protected log
            return f"log({functional_to_infix(operands[0])})"#Returns the infix notation of the protected log
    else:
        return str(expression)#Returns the expression as a string