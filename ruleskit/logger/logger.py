import logging
import os


class Logger:
    log = None

    @classmethod
    def get_log(cls):
        if cls.log is None:
            level = os.environ.get("LOGGER_LEVEL", "INFO")
            cls.log = logging.getLogger()
            cls.log.setLevel(level)
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter("%(levelname)8s in %(filename)s at %(asctime)s - %(message)s")
            ch.setFormatter(formatter)
            cls.log.addHandler(ch)
        return cls.log


log = Logger.get_log()
