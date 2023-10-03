# Example usage for Redmi Note 4X
# python rtsp.py -o=redmi -i=wlx00ebd8c3df3d -m=4c:49:e3:3a:87:4a -u=admin -password=0000 -port=8080 -path="/h264_ulaw.sdp"

import argparse
import calendar
import datetime
import fractions
import inspect
import logging
import os
import subprocess
import time

import numpy as np
import cv2
import av


def getLink(interface, macAddr, username, password, port, urlPath):
    for i in range(10):
        link, err = getLinkOne(interface, macAddr, username, password, port, urlPath)
        if not err:
            return link, None

        time.sleep(1)
    return "", err


def getLinkOne(interface, macAddr, username, password, port, urlPath):
        arps, arpOut, err = getARP(interface)
        if err:
            return "", err
        if macAddr not in arps:
            return "", f"{arps} {arpOut} {inspect.getframeinfo(inspect.currentframe())}"

        ip = arps[macAddr].ip
        link = f"rtsp://{username}:{password}@{ip}:{port}{urlPath}"
        return link, None


def getARP(interface):
        outStr = subprocess.run(["arp-scan", f"--interface={interface}", "-l", "-x"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = outStr.split("\n")
        # Skip last empty line.
        lines = lines[:len(lines)-1]

        arps = {}
        for line in lines:
            records = line.split("\t")
            records = list(filter(None, records))
            if len(records) != 3:
                return None, "", f"{records} {inspect.getframeinfo(inspect.currentframe())}"
            arp = argparse.Namespace()
            arp.ip = records[0]
            arp.hw = records[1]
            arps[arp.hw] = arp
        return arps, outStr, None


def epochMilliSec(t):
    epoch = calendar.timegm(t.utctimetuple())
    milliSec = int(t.microsecond/1000)
    return epoch * 1000 + milliSec
 


def getDivisor(t):
    return t.strftime("%Y%m%d_%H%M_") + str(int(t.second / 10))


class Writer:
    def __init__(self, dir, start, img):
        self.dir = dir
        self.start = start

        # Folder to save.
        fdir = os.path.join(self.dir, self.start.strftime("%Y%m%d"))
        os.makedirs(fdir, exist_ok=True)

        # Path to save.
        milliSec = str(int(self.start.microsecond/1000)).zfill(3)
        base = self.start.strftime("%Y%m%d_%H%M%S_") + milliSec + ".mp4"
        self.fpath = os.path.join(fdir, base)

        self.container = av.open(self.fpath, mode="w")
        self.stream = self.container.add_stream("h264", rate=30)
        self.stream.width = img.shape[1]
        self.stream.height = img.shape[0]
        self.stream.pix_fmt = "yuv420p"
        # Time base in milliseconds.
        self.stream.codec_context.time_base = fractions.Fraction(1, 1000)

    def write(self, t, img):
        sms = epochMilliSec(self.start)
        tms = epochMilliSec(t)

        f = av.VideoFrame.from_ndarray(img, format="bgr24")
        f.pts = tms - sms
        for packet in self.stream.encode(f):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-o")
    parser.add_argument("-l")
    parser.add_argument("-i")
    parser.add_argument("-m")
    parser.add_argument("-u")
    parser.add_argument("-password")
    parser.add_argument("-port")
    parser.add_argument("-path")
    args = parser.parse_args()

    err = mainWithErr(args)
    if err:
        logging.fatal("%s", err)


def mainWithErr(args):
    link = args.l
    if link == "":
        link, err = getLink(args.i, args.m, args.u, args.password, args.port, args.path)
        if err:
            return err

    vc = cv2.VideoCapture(link, apiPreference=cv2.CAP_FFMPEG)
    if not vc.isOpened():
        return f"{link} {inspect.getframeinfo(inspect.currentframe())}"
    for i in range(10):
        ok, img = vc.read()
        if ok:
            break
    if not ok:
        return f"{link} {inspect.getframeinfo(inspect.currentframe())}"
    t = datetime.datetime.now(datetime.timezone.utc)
    w = Writer(args.o, t, img)
    w.write(t, img)

    while True:
        ok, img = vc.read()
        if not ok:
            return f"{link} {inspect.getframeinfo(inspect.currentframe())}"

        t = datetime.datetime.now(datetime.timezone.utc)
        if getDivisor(t) != getDivisor(w.start):
            w.close()
            w = Writer(args.o, t, img)

        w.write(t, img)


    return None


if __name__ == "__main__":
        main()
