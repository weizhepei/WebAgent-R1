import argparse
import http.server
import logging
import os
import pathlib
import socketserver
import subprocess
import sys
import threading

# setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# setup files config
lock_file_path = "reset.lock"
fail_file_path = "fail_message"

def write_fail_message(message: str):
    with open(fail_file_path, 'w') as f:
        f.write(message)

def read_fail_message():
    with open(fail_file_path, 'r') as f:
        fail_message = f.read()
    return fail_message

def reset_ongoing():
    return os.path.exists(lock_file_path)

def initiate_reset():

    # Attempt to acquire lock (create lock file atomically)
    try:
        fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as file:
            file.write('')  # empty file
    except FileExistsError:
        return False

    # Execute reset (and then release lock) in a separate thread
    def reset_fun():
        try:
            # Execute the reset script
            subprocess.run(['bash', 'reset.sh'], check=True)
            logger.info("Reset successful!")
            write_fail_message("")
        except subprocess.CalledProcessError as e:
            logger.info("Reset failed :(")
            write_fail_message(str(e))

        # Release lock (remove lock file)
        pathlib.Path(lock_file_path).unlink(missing_ok=True)

    thread = threading.Thread(target=reset_fun)
    thread.start()

    return True


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = self.path
        logger.info(f"{parsed_path} request received")
        match parsed_path:
            case '/reset':
                if initiate_reset():
                    logger.info("Running reset script...")
                    self.send_response(200)  # OK
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f'Reset initiated, check status <a href="/status">here</a>'.encode())
                else:
                    logger.warning("Reset already running, ignoring request.")
                    self.send_response(418)  # I'm a teapot
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f'Reset already running, check status <a href="/status">here</a>'.encode())
            case "/status":
                fail_message = read_fail_message()
                if reset_ongoing():
                    logger.info("Returning ongoing status")
                    self.send_response(200)  # OK
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f'Reset ongoing'.encode())
                elif fail_message:
                    logger.error("Returning error status")
                    self.send_response(500)  # Internal Server Error
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f'Error executing reset script:<p>{fail_message}</p>'.encode())
                else:
                    logger.info("Returning ready status")
                    self.send_response(200)  # OK
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f'Ready for duty!'.encode())
            case _:
                logger.info("Wrong request")
                self.send_response(404)  # Not Found
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'Endpoint not found'.encode())


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Start a simple HTTP server to execute a reset script.')
parser.add_argument('--port', type=int, help='Port number the server will listen to')
args = parser.parse_args()

# Clear fail and lock files
write_fail_message("")
if reset_ongoing():
    os.remove(lock_file_path)

# Run the server
with http.server.ThreadingHTTPServer(('', args.port), CustomHandler) as httpd:
    logger.info(f'Serving on port {args.port}...')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
