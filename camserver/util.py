from http import server as httpserver
import json
import queue
import select
import sys
import threading
from typing import Any

import multipart
import numpy as np


class Duration:
    def __init__(self, d):
        self.d = np.int64(d)

    def Minutes(self):
        i = self.d / Minute
        j = self.d % Minute
        return float(i) + float(j)/Minute

    def Seconds(self):
        i = self.d / Second
        j = self.d % Second
        return float(i) + float(j)/Second

    def Milliseconds(self):
        return self.d / 1e6

    def Microseconds(self):
        return self.d / 1e3


Second = np.int64(1e9)
Minute = 60*Second
Hour = 60*Minute


def ReadMultipart(handler: httpserver.BaseHTTPRequestHandler):
    wsgi = {}
    wsgi["REQUEST_METHOD"] = "POST"
    wsgi["CONTENT_LENGTH"] = handler.headers["Content-Length"]
    wsgi["CONTENT_TYPE"] = handler.headers["Content-Type"]
    wsgi["wsgi.input"] = handler.rfile
    forms, files = multipart.parse_form_data(wsgi)
    for part in files.values():
        # Read and then close TemporaryFile.
        raw = part.raw
        part.close()

        part.file = raw
    return forms, files
        

def HTTPRespJ(handler: httpserver.BaseHTTPRequestHandler, resp: Any):
    b = json.dumps(resp)
    handler.send_response(200)
    handler.send_header('Content-Type', 'text/json')
    handler.end_headers()
    handler.wfile.write(b.encode("utf-8"))


class StdinWindows:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def readline(self, dur):
        try:
            line = self.queue.get(timeout=dur.Seconds())
        except queue.Empty:
            line = ""
        return line.strip()

    def _run(self):
        while True:
            line = sys.stdin.readline()
            self.queue.put(line)


class StdinUnix:
    def __init__(self):
       pass

    def readline(self, dur):
        rlist, _, _ = select.select([sys.stdin], [], [], dur.Seconds())
        if rlist:
            line = sys.stdin.readline()
        else:
            line = ""
        return line.strip()


def NewStdinReader():
    if sys.platform == "win32":
        return StdinWindows()
    return StdinUnix()
