__version__ = "0.1.0"

# export.public_api() will expose things here. To make sure that happens, we just need to
# import all the submodules so that the decorator is actually executed.
__all__ = []


def __discover_modules():
    import importlib
    import pkgutil

    import tripy

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
