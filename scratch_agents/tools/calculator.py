from .base_tool import BaseTool

calculator_tool_definition = { 
    "type": "function",
    "function": {
        "name": "calculator", 
        "description": "Perform basic arithmetic operations between two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {
                    "type": "number",
                    "description": "First number for the calculation"
                },
                "second_number": {
                    "type": "number",
                    "description": "Second number for the calculation"
                }
            },
            "required": ["operator", "first_number", "second_number"],
        }
    }
}

def calculator(operator: str, first_number: float, second_number: float) -> float:
   if operator == 'add':
       return first_number + second_number
   elif operator == 'subtract':
       return first_number - second_number
   elif operator == 'multiply':
       return first_number * second_number
   elif operator == 'divide':
       if second_number == 0:
           raise ValueError("Cannot divide by zero")
       return first_number / second_number
   else:
       raise ValueError(f"Unsupported operator: {operator}")

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic arithmetic operations between two numbers.",
            tool_definition=calculator_tool_definition
        )
        
    def execute(self, operator: str, first_number: float, second_number: float) -> float:
        return calculator(operator, first_number, second_number)
    
calculator = CalculatorTool()

