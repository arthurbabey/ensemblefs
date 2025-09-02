import warnings
warnings.warn(
    "Package 'ensemblefs' has been renamed to 'moosefs'. "
    "Please update your imports: `import moosefs`.",
    DeprecationWarning,
    stacklevel=2,
)
from moosefs import *  # re-export public API

