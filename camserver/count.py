# Example usage:
# python count.py -c='{"yolo": "yolo_best.pt"}'

import argparse
import datetime
import http
from http import server as httpserver
import json
import inspect
import json
import logging
import threading

# import torch
# import ultralytics

import util


def Analyze(handler: httpserver.BaseHTTPRequestHandler):
    forms, files = util.ReadMultipart(handler)
    logging.info("req %s", forms["myname"])
    logging.info("req %s", files["f"].filename)
    logging.info("req %s", files["f"].file)
    util.HTTPRespJ(handler, {"hello": "world"})


class MyServerHandler(httpserver.BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/Quit":
            threading.Thread(target=self.server.shutdown).start()
        elif self.path == "/Analyze":
            Analyze(self)
        else:
            util.HTTPRespJ(self, {"hello": "world"})

    def log_message(self, format, *args):
        return


class StructuredMessage:
    def __init__(self, event, /, **kwargs):
        stack = inspect.stack()
        self.longfile = stack[1][1] + ":" + str(stack[1][2])
        self.event = event
        self.kwargs = kwargs

    def __str__(self):
        self.kwargs["Ltime"] = datetime.datetime.utcnow().isoformat()+"Z"
        self.kwargs["Llongfile"] = self.longfile
        self.kwargs["Levent"] = self.event
        return json.dumps(self.kwargs)

_l = StructuredMessage


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c")
    args = parser.parse_args()

    err = mainWithErr(args)
    if err:
        logging.fatal("%s", err)


def mainWithErr(args):
    cfg = json.loads(args.c)
    logging.info(_l("config", **cfg))

    httpd = http.server.HTTPServer(("localhost", 8080), MyServerHandler)
    port = httpd.server_address[1]
    print('{"port":%d}' % port)
    httpd.serve_forever()


if __name__ == "__main__":
        main()
