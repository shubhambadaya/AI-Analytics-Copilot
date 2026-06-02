import io
import sys
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.analysis.validator import validate_pandas_code
from src.utils.logger import get_logger
import src.analysis.stats_engine as stats_engine

logger = get_logger(__name__)

def execute_analysis(
    df: pd.DataFrame,
    code_str: str,
    all_datasets: Optional[Dict[str, pd.DataFrame]] = None,
    extra_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes AST-validated Pandas Python code on a DataFrame (or multiple DataFrames).
    Injects all loaded tables into the namespace to support multi-table joins.

    Args:
        df: The active pandas DataFrame.
        code_str: The Python code containing the pandas operations.
        all_datasets: Dict mapping dataset filenames/names to loaded DataFrames.
        extra_vars: Optional extra variables to inject into the namespace (e.g.
            results computed by earlier investigation steps, so a step can build on
            prior work). Keys must be valid identifiers.

    Returns:
        A dictionary containing success indicators, output dataframe, stdout, and stderr.
    """
    logger.info("Initiating pandas code execution...")
    
    # 1. Security check
    is_valid, validation_errors = validate_pandas_code(code_str)
    if not is_valid:
        logger.warning("Code blocked by AST Validator.")
        return {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "\n".join(validation_errors),
            "error": "Validation Blocked"
        }
        
    # 2. Redirect stdout/stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    
    # 3. Build sandboxed namespace with permitted builtins
    # Provide a restricted __import__ so LLM-generated code can use
    # `import pandas as pd` etc. Only modules approved by the AST validator pass.
    SAFE_MODULES = {"pandas", "numpy", "datetime", "math", "scipy", "statsmodels", "stats_engine", "sklearn", "ml_engine"}
    
    def _safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if base not in SAFE_MODULES:
            raise ImportError(f"Import of '{name}' is not permitted in the sandbox.")
        return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)
    
    import datetime
    import math
    import src.analysis.ml_engine as ml_engine
    
    # IMPORTANT: Use a single unified namespace (exec_globals only).
    # Python's exec() binds function closures to the globals dict.
    # If pd/np are only in locals, functions like analyze() can't see them.
    exec_globals = {
        "__builtins__": {
            "__import__": _safe_import,
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "hash": hash,
            "hex": hex, "id": id, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len, "list": list,
            "map": map, "max": max, "min": min, "next": next, "object": object,
            "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip
        },
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "datetime": datetime,
        "math": math,
        "stats_engine": stats_engine,
        "ml_engine": ml_engine
    }
    
    # Scalability & Multi-table: Inject all active datasets as global variables
    # named after the clean table basenames so code can join them directly
    if all_datasets:
        logger.info(f"Injecting {len(all_datasets)} datasets into execution namespace...")
        for tbl_name, tbl_df in all_datasets.items():
            # Create a clean variable name (e.g., 'subscriber_profile.csv' -> 'subscriber_profile')
            clean_var = tbl_name.split(".")[0].replace(" ", "_").replace("-", "_")
            exec_globals[clean_var] = tbl_df.copy()
            logger.info(f"Preloaded table variable: '{clean_var}'")

    # Inject results from earlier investigation steps so later steps can build on them.
    if extra_vars:
        for var_name, value in extra_vars.items():
            if var_name.isidentifier():
                exec_globals[var_name] = value.copy() if hasattr(value, "copy") else value
                logger.info(f"Injected prior-step variable: '{var_name}'")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    success = False
    result_df = None
    error_msg = None
    
    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        
        # Execute the code block with a single namespace
        exec(code_str, exec_globals)
        
        # Extract resulting dataframe or metrics
        if "analyze" in exec_globals and callable(exec_globals["analyze"]):
            logger.info("Found analyze() function in code. Calling it...")
            res = exec_globals["analyze"](df.copy())
            if isinstance(res, (pd.DataFrame, pd.Series)):
                if isinstance(res, pd.Series):
                    result_df = res.to_frame()
                else:
                    result_df = res
            else:
                if isinstance(res, dict):
                    result_df = pd.DataFrame([res])
                else:
                    result_df = pd.DataFrame({"result_value": [res]})
        elif "result_df" in exec_globals:
            result_df = exec_globals["result_df"]
        elif "result" in exec_globals:
            res = exec_globals["result"]
            if isinstance(res, (pd.DataFrame, pd.Series)):
                if isinstance(res, pd.Series):
                    result_df = res.to_frame()
                else:
                    result_df = res
            else:
                if isinstance(res, dict):
                    result_df = pd.DataFrame([res])
                else:
                    result_df = pd.DataFrame({"result_value": [res]})
        else:
            logger.warning("No explicit 'analyze' function or 'result_df'/'result' variable found. Using the last state of 'df'")
            result_df = exec_globals["df"]
            
        success = True
        logger.info("Pandas code executed successfully.")
        
    except Exception as e:
        success = False
        error_msg = traceback.format_exc()
        logger.error(f"Runtime error during pandas code execution: {str(e)}")
        print(f"Runtime Error: {str(e)}", file=sys.stderr)
        
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
    return {
        "success": success,
        "result": result_df,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error_msg,
        "sandbox_globals": exec_globals
    }
