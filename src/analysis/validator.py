import ast
from typing import Tuple, List, Set
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default whitelist of modules that can be imported or used in the analysis script
ALLOWED_MODULES: Set[str] = {"pandas", "pd", "numpy", "np", "datetime", "math", "scipy", "statsmodels", "src.analysis.stats_engine", "stats_engine"}

# Default blacklist of built-in functions that pose security risks
FORBIDDEN_BUILTINS: Set[str] = {
    "eval", "exec", "open", "compile", "getattr", "setattr", "delattr", 
    "__import__", "dir", "locals", "globals", "vars", "breakpoint", 
    "input", "super", "system"
}

class SecurityVisitor(ast.NodeVisitor):
    """
    AST Visitor that walks the parsed code tree to enforce security rules:
    - No forbidden imports.
    - No forbidden built-in functions.
    - No double-underscore (dunder) attributes (prevents sandbox escapes).
    """
    def __init__(self):
        self.errors: List[str] = []
        
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Import of module '{alias.name}' is forbidden.")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Import from module '{node.module}' is forbidden.")
        else:
            self.errors.append("Relative imports are forbidden.")
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Prevent access to dunder methods/attributes (e.g. obj.__class__, obj.__subclasses__)
        if node.attr.startswith('__'):
            self.errors.append(f"Access to private/dunder attribute '{node.attr}' is forbidden.")
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call) -> None:
        # Check if calling a forbidden built-in
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FORBIDDEN_BUILTINS:
                self.errors.append(f"Call to forbidden function '{func_name}' is blocked.")
        # Check for calls like obj.system() or similar using string checks if necessary,
        # but the attribute check covers obj.__getattr__ or private variables.
        self.generic_visit(node)

def validate_pandas_code(code_str: str) -> Tuple[bool, List[str]]:
    """
    Validates a string of Python/Pandas code for security and syntax correctness.
    
    Args:
        code_str: The Python code to validate.
        
    Returns:
        A tuple of (is_valid: bool, errors: List[str]).
    """
    logger.info("Starting security validation of generated code...")
    
    if not code_str or not code_str.strip():
        return False, ["Code is empty."]
        
    try:
        # Parse the code into an AST
        tree = ast.parse(code_str)
    except SyntaxError as e:
        logger.warning(f"Syntax error during validation: {str(e)}")
        return False, [f"Syntax Error: {e.msg} at line {e.lineno}, col {e.offset}"]
        
    # Walk the AST using our SecurityVisitor
    visitor = SecurityVisitor()
    visitor.visit(tree)
    
    if visitor.errors:
        for err in visitor.errors:
            logger.warning(f"Security validation failure: {err}")
        return False, visitor.errors
        
    logger.info("Security validation passed successfully.")
    return True, []
