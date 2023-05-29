class Error(Exception):
    """Base class for other exceptions"""
    pass


class ArgumentError(Error):
    """Raised when a function got a wrong argument"""
    pass


class SampleRateError(Error):
    """Raised when a loaded audio has different sample rate than the one defined in the configuration"""
    pass
