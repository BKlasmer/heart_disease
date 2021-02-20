#!/usr/bin/env python3
# -*- coding: latin-1 -*-

import logging
import sys

""" Logging information using the Python logging for
    all the other modules in this codebase.
"""


class Logging(object):
    _default_logging_level = "INFO"

    @property
    def default_logging_level(self) -> str:
        # gets the current default logging level
        return Logging._default_logging_level

    @default_logging_level.setter
    def default_logging_level(self, default_logging_level: str) -> None:
        # sets the default logging level
        Logging._default_logging_level = default_logging_level

    @staticmethod
    def create_logger(logger_name: str, logger_level: str = None) -> logging.Logger:
        # instantiate a logging object
        logger = logging.getLogger(logger_name)

        if not logger.hasHandlers():
            # format the logging string
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)-s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            screen_handler = logging.StreamHandler(stream=sys.stdout)
            screen_handler.setFormatter(formatter)

            # check if we use the input parameter or the default value
            if isinstance(logger_level, str):
                logger.setLevel(logger_level)
            else:
                logger.setLevel(Logging._default_logging_level)
            logger.addHandler(screen_handler)
        return logger
