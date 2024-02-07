from dataclasses import dataclass, field

from colored import Fore
from typing import List, Dict
import copy


@dataclass
class VerbosityConfig:
    prefix: str
    color: str
    enables: List[str] = field(default_factory=list)
    """
    Name(s) of other verbosities that this verbosity will automatically enable.
    """


def make_verbosity_map() -> Dict[str, VerbosityConfig]:
    verbosity_map = {
        "verbose": VerbosityConfig("[V] ", Fore.light_magenta, ["info"]),
        "info": VerbosityConfig("[I] ", "", ["warning"]),
        "warning": VerbosityConfig("[W] ", Fore.light_yellow, ["error"]),
        "error": VerbosityConfig("[E] ", Fore.light_red),
        "trace": VerbosityConfig("==== Trace IR ====\n", Fore.magenta),
        "flat_ir": VerbosityConfig("==== Flat IR ====\n", Fore.magenta),
        "stablehlo": VerbosityConfig("==== StableHLO IR ====\n", Fore.magenta),
        # Shorthand for enabling all IR dumps,
        # `logger.ir` probably shouldn't be called but `"ir"` may be used as a verbosity option.
        "ir": VerbosityConfig("", "", enables=["trace", "flat_ir", "stablehlo"]),
        "timing": VerbosityConfig("==== Timing ====\n", Fore.cyan),
    }

    # Do a pass to recursively expand enables
    for verbosity in verbosity_map.values():
        new_enables = copy.copy(verbosity.enables)

        index = 0
        while index < len(new_enables):
            new_enables.extend(verbosity_map[new_enables[index]].enables)
            index += 1

        verbosity.enables = list(set(new_enables))

    return verbosity_map
