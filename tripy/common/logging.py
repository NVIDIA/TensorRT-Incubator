import logging
from colored import Fore, attr


class LoggerModes:
    """
    Logger settings to determine the types of messages to be logged.
    """

    IR = 1 << 0
    TIMING = 1 << 1
    VERBOSE = IR | TIMING


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "IR_printer": Fore.magenta,
        "Timing": Fore.cyan,
        "Verbose": Fore.blue,
    }

    def format(self, record):
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + attr("reset")


class LogFilter(logging.Filter):
    def __init__(self, levels):
        self.levels = levels

    def filter(self, record):
        return record.levelno & self.levels


def create_level_log_method(level):
    def log_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    return log_method


# Setup the logger
logging.addLevelName(LoggerModes.IR, "IR_printer")
logging.addLevelName(LoggerModes.TIMING, "Timing")
logging.addLevelName(LoggerModes.VERBOSE, "Verbose")
G_LOGGER = logging.getLogger(__name__)

for level in [LoggerModes.IR, LoggerModes.TIMING, LoggerModes.VERBOSE]:
    setattr(logging.Logger, logging.getLevelName(level).lower(), create_level_log_method(level))


def set_logger_mode(loggerModes: LoggerModes):
    global G_LOGGER
    if loggerModes & LoggerModes.VERBOSE:
        G_LOGGER.setLevel(LoggerModes.VERBOSE)

    if loggerModes & LoggerModes.TIMING:
        G_LOGGER.setLevel(LoggerModes.TIMING)

    if loggerModes & LoggerModes.IR:
        G_LOGGER.setLevel(LoggerModes.IR)

    ch = logging.StreamHandler()
    ch.addFilter(LogFilter(loggerModes))
    formatter = ColoredFormatter("\n====%(levelname)s==== \n%(message)s")
    ch.setFormatter(formatter)
    G_LOGGER.addHandler(ch)
