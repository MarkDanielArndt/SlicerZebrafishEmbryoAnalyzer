class AnalysisInputError(ValueError):
    """Raised by ZebrafishAnalysisLogic when analysis inputs fail validation."""


class MRMLAdapterError(RuntimeError):
    """Raised when MRML scene integration fails after a successful analysis."""


class ModelNotCachedError(RuntimeError):
    """Raised when a required model file is not present in the local cache.

    Download models via ZebrafishAnalysisLib.model_downloader.download_models()
    before running analysis.
    """
