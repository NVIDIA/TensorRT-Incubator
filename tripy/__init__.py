__version__ = "0.1.0"

# export.public_api() will expose things here. To make sure that happens, we just need to
# import all the submodules so that the decorator is actually executed.
__all__ = []

import tripy


def __discover_modules():
    import importlib
    import pkgutil

    mods = [tripy]
    while mods:
        mod = mods.pop(0)

        yield mod

        if hasattr(mod, "__path__"):
            mods.extend(
                [
                    importlib.import_module(f"{mod.__name__}.{submod.name}")
                    for submod in pkgutil.iter_modules(mod.__path__)
                ]
            )


_ = list(__discover_modules())


def __getattr__(name: str):
    from tripy.common.exception import search_for_missing_attr

    look_in = [(tripy, "tripy")]
    search_for_missing_attr("tripy", name, look_in)
