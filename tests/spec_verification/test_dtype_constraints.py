from typing import List
from tripy.common.datatype import DATA_TYPES
from colored import fg, attr
import itertools
import inspect
import pytest
from tests.spec_verification.object_builders import create_obj
from tripy.dtype_info import TYPE_VERIFICATION, RETURN_VALUE

# imports necessary for creating inputs and running exec:
import tripy as tp
import numpy as np
import cupy as cp
import pytest


def method_handler(kwargs, func_obj):
    """
    handles any function that has a special way of being called such as methods
    """
    _METHOD_OPS = {
        "__add__": (lambda a, b: f"{a} + {b}"),
        "__sub__": (lambda a, b: f"{a} - {b}"),
        "__rsub__": (lambda a, b: f"{a} - {b}"),
        "__pow__": (lambda a, b: f"{a}**{b}"),
        "__rpow__": (lambda a, b: f"{a}**{b}"),
        "__mul__": (lambda a, b: f"{a} * {b}"),
        "__rmul__": (lambda a, b: f"{a} * {b}"),
        "__truediv__": (lambda a, b: f"{a} / {b}"),
        "__lt__": (lambda a, b: f"{a} < {b}"),
        "__le__": (lambda a, b: f"{a} <= {b}"),
        "__eq__": (lambda a, b: f"{a} == {b}"),
        "__ne__": (lambda a, b: f"{a} != {b}"),
        "__ge__": (lambda a, b: f"{a} >= {b}"),
        "__gt__": (lambda a, b: f"{a} > {b}"),
    }
    if not func_obj.__name__.startswith("_"):
        return f"{RETURN_VALUE} = " + f"tp.{func_obj.__name__}(**kwargs)"
    elif _METHOD_OPS.get(func_obj.__name__, None):
        rtn_builder = _METHOD_OPS.get(func_obj.__name__, None)
        args_list = list(kwargs.values())
        if len(args_list) > 2:
            print("WARNING WILL ONLY USE FIRST TWO PARAMS")
        return f"{RETURN_VALUE} = " + rtn_builder(args_list[0], args_list[1])
    else:
        raise RuntimeError(f"{fg('red')}Could not figure out method in function: {func_obj.__name__}{attr('reset')}")


func_list = []
for func_obj, parsed_dict, types_assignments in TYPE_VERIFICATION.values():
    inputs = parsed_dict["inputs"]
    returns = parsed_dict["returns"]
    types = parsed_dict["types"]
    # exclude quantization types until q/dq MR is merged
    types_to_exclude = ["int4", "float8"]
    # get all of the dtypes for positive test case
    positive_test_dtypes = {}
    for name, dt in types.items():
        positive_test_dtypes[name] = list(filter(lambda item: item not in types_to_exclude, set(dt)))

    # get all dtypes for negative test case
    negative_test_dtypes = {}
    for name, dt in types.items():
        total_dtypes = set(filter(lambda item: item not in types_to_exclude, map(str, DATA_TYPES.values())))
        cleaned_dtypes = set()
        for inp_type in dt:
            cleaned_dtypes.add(inp_type)
        temp_list = list(total_dtypes - cleaned_dtypes)
        negative_test_dtypes[name] = list(total_dtypes - cleaned_dtypes)
    for positive_case in [True, False]:
        dtype_lists_list = []
        if positive_case:
            #  positive test case dtypes:
            dtype_keys = positive_test_dtypes.keys()
            dtype_lists_list.append(dict(zip(dtype_keys, positive_test_dtypes.values())))
            case_name = "valid: "
        else:
            # deal with all negative cases:
            # create a list of dictionary lists and then go over each dictionary
            for name_temp, dt in negative_test_dtypes.items():
                # creating a temp dictionary:
                temp_dict = {name_temp: dt}
                for name_not_equal in negative_test_dtypes.keys():
                    if name_temp != name_not_equal:
                        temp_dict[name_not_equal] = set(
                            filter(
                                lambda item: item not in types_to_exclude,
                                map(str, DATA_TYPES.values()),
                            )
                        )
                dtype_lists_list.append(temp_dict)
            case_name = "invalid: "
        for dtype_lists in dtype_lists_list:
            for combination in itertools.product(*(dtype_lists.values())):
                # Create a tuple with keys and corresponding elements
                namespace = dict(zip(dtype_lists.keys(), map(lambda val: getattr(tp, val), combination)))
                ids = [f"{dtype_name}={dtype}" for dtype_name, dtype in namespace.items()]
                func_list.append((func_obj, inputs, returns, namespace, positive_case, case_name + ", ".join(ids)))


@pytest.mark.parametrize("test_data", func_list, ids=lambda val: val[5])
def test_dtype_constraints(test_data):
    func_obj, inputs, returns, namespace, positive_case, _ = test_data
    # create all inputs for positive test case
    kwargs = {}
    for param_name, input_desc in inputs.items():
        kwargs[param_name] = create_obj(func_obj, param_name, input_desc, namespace)
    # run api call
    api_call_locals = {"kwargs": kwargs}
    exec(method_handler(kwargs, func_obj), globals(), api_call_locals)
    # assert with output
    if positive_case:
        assert api_call_locals[RETURN_VALUE].dtype == namespace[list(returns.values())[0]["dtype"]]
    else:
        assert (
            not api_call_locals[RETURN_VALUE].dtype == namespace[list(returns.values())[0]["dtype"]]
        ), f"{fg('red')}invalid test case failure: {list(namespace.values())[0]} is a valid dtype for {list(namespace.keys())[0]}{attr('reset')}"
