import logging
from rich.logging import RichHandler


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    # Reduce noise from ffmpeg-python and other libraries if needed
    logging.getLogger("asyncio").setLevel(logging.WARNING)
