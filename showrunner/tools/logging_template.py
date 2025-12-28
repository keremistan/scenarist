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

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    # Ignore KeyboardInterrupt (Ctrl+C) so we can exit normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Create a special logger just for crashes
    crit_log = setup_logging("CRITICAL")
    crit_log.critical("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))

# Register the crash handler
sys.excepthook = handle_uncaught_exception