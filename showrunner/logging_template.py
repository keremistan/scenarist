import logging
import datetime
import sys

def setup_logging(script_name="script"):
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"./showrunner/logs/log_{timestamp}.txt"

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-15s - %(message)s', # Adds time to every line
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout) # Print to screen
        ]
    )
    
    # Return a logger object
    return logging.getLogger(script_name)