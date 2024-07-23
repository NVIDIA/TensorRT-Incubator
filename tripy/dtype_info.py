import inspect

TYPE_VERIFICATION = {}
RETURN_VALUE = "RETURN_VALUE"


def dtype_info(dtype_variables: dict, dtype_constraints: dict):

    def decorator(func_obj):

        func_sig = inspect.signature(func_obj)
        param_dict = func_sig.parameters
        return_annotation = func_sig.return_annotation
        # construct inputs_dict
        inputs_dict = {}
        for param_name, param_type in param_dict.items():
            param_type = param_type.annotation
            param_constraint_dict = {}
            if param_type.startswith("tripy."):
                param_type = param_type[6:]
            param_constraint_dict[param_type] = {"dtype": dtype_constraints[param_name]}
            inputs_dict[param_name] = param_constraint_dict
        # construct returns_dict:
        returns_dict = {}
        if return_annotation.startswith("tripy."):
            return_annotation = return_annotation[6:]
        returns_dict[return_annotation] = {"dtype": dtype_constraints[RETURN_VALUE]}

        parsed_dict = {"inputs": inputs_dict, "returns": returns_dict, "types": dtype_variables}
        TYPE_VERIFICATION[func_obj.__qualname__] = (func_obj, parsed_dict, dtype_constraints)
        return func_obj

    return decorator
