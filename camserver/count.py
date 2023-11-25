# Example usage:
# python count.py -c='{"yolo": "yolo_best.pt"}'

import argparse
import datetime
import inspect
import json
import logging

import torch
import ultralytics


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


if __name__ == "__main__":
        main()
