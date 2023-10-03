# sudo env "PYTHONPATH=/usr/local/detectron2:/home/topunion/.local/lib/python3.8/site-packages" python3 ard.py

import logging
import os
import serial


def getTTY(port):
    sysDir = "/sys/bus/usb/devices"
    dirs = os.listdir(sysDir)
    portDirs = []
    for d in dirs:
        if not d.startswith(port):
            continue
        portDirs.append(os.path.join(sysDir, d))

    for d in portDirs:
        ttyDir = os.path.join(d, "tty")
        if not os.path.isdir(ttyDir):
            continue
        ttys = os.listdir(ttyDir)
        if len(ttys) == 0:
            return "", f"no tty {ttyDir}"
        return ttys[0], None

    return "", f"no tty {portDirs}"


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    tty, err = getTTY("1-5")
    if err:
        raise Exception(err)
    s = serial.Serial(f"/dev/{tty}", baudrate=115200)
    while True:
        data = s.readline()
        print(data)


if __name__ == "__main__":
    main()
