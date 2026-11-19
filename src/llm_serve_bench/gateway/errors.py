"""Gateway error types and their HTTP meaning."""


class BackendError(RuntimeError):
    """Base class for failures originating in a backend."""


class BackendNotReadyError(BackendError):
    """Backend cannot serve requests yet -> HTTP 503."""


class BackendTimeoutError(BackendError):
    """Backend exceeded the configured deadline -> HTTP 504."""


class BackendRequestError(BackendError):
    """Backend call failed for another reason -> HTTP 502."""
