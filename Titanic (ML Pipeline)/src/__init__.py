import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="====LOG====>>>>>> %(name)s | %(levelname)s | %(asctime)s \n %(message)s \n",
    handlers=[
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("Logger is set up.")
