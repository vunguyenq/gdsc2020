import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(module)s %(funcName)s line %(lineno)d %(message)s',
                    handlers=[logging.StreamHandler()])
