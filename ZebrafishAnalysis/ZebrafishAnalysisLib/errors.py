class AnalysisInputError(ValueError):
    """Raised by ZebrafishAnalysisLogic when analysis inputs fail validation."""


class MRMLAdapterError(RuntimeError):
    """Raised when MRML scene integration fails after a successful analysis."""


class ModelNotCachedError(RuntimeError):
    """Raised when a required model file is not present in the local cache.

    Download models via ZebrafishAnalysisLib.model_downloader.download_models()
    before running analysis.
    """


class InferenceWorkerError(Exception):
    """Raised when the inference worker exits with a non-zero code."""


class InferenceProtocolError(Exception):
    """Raised when the worker output does not match the expected protocol."""
