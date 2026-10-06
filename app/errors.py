"""Domain errors for the exposure-merge service.

All invalid-input / unprocessable conditions raise :class:`MergeError`; the API
layer maps it to HTTP 422.
"""


class MergeError(Exception):
    """Raised for invalid requests or failure to merge the stack."""
