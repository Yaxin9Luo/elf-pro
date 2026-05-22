"""Small HOPE metric hook used by cluster training jobs."""

import logging


logger = logging.getLogger(__name__)


class HopeMetricHook:
    """Report scalar metrics through HOPE tracking when running inside HOPE."""

    def __init__(self, enabled):
        self.enabled = enabled
        self._tracking = None
        self._warned = False

        if not enabled:
            return

        try:
            from hope import tracking
        except Exception as exc:
            self.enabled = False
            self._warned = True
            logger.warning("HOPE metric tracking unavailable: %s", exc)
            return

        self._tracking = tracking
        logger.info("HOPE metric tracking initialized.")

    @property
    def available(self):
        return self.enabled and self._tracking is not None

    def log_scalar(self, tag, value, step):
        if not self.available or value is None:
            return

        try:
            self._tracking.context(self._tracking.metric(tag, float(value), int(step)))
        except Exception as exc:
            if not self._warned:
                logger.warning(
                    "HOPE metric tracking failed; further failures will be suppressed: %s",
                    exc,
                )
                self._warned = True

    def log_scalars(self, scalar_dict, step):
        for tag, value in scalar_dict.items():
            self.log_scalar(tag, value, step)

    def close(self):
        if not self.available:
            return

        try:
            safe_close = getattr(self._tracking, "safe_close", None)
            if callable(safe_close):
                safe_close()
        except Exception as exc:
            if not self._warned:
                logger.warning("HOPE metric tracking close failed: %s", exc)
                self._warned = True
