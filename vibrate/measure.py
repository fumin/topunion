import sys
# For klippy itself.
sys.path.append("/usr/local/klipper/klippy")

import argparse
import logging
import importlib
import inspect
import os
import queue
import threading
import traceback
import time
import zlib

import numpy as np


def klippyImport(name, relpath):
    fpath = os.path.join("/usr/local/klipper/klippy", relpath)
    return importlib.machinery.SourceFileLoader(name, fpath).load_module()
klippy = argparse.Namespace()
klippy.reactor = klippyImport("reactor", "reactor.py")
klippy.clocksync = klippyImport("clocksync", "clocksync.py")
klippy.serialhdl = klippyImport("serialhdl", "serialhdl.py")
klippy.mcu = klippyImport("mcu", "mcu.py")
klippy.adxl345 = klippyImport("extras.adxl345", os.path.join("extras", "adxl345.py"))


class ADXL345:
    def __init__(self, usbPort):
        self.spi_oid = 0
        self.adxl_oid = 1

        # Klipper generic.
        self.serial, err = self._getSerial(usbPort)
        if err:
            self.err = err
            return
        self.reactor = klippy.reactor.Reactor()
        self.clocksync = klippy.clocksync.ClockSync(self.reactor)
        self.ser = klippy.serialhdl.SerialReader(self.reactor)

        # ADXL345 specific.
        self.clocksyncreg = None
        self.spi_send_cmd = None
        self.spi_transfer_cmd = None
        self.query_adxl345_cmd = None
        self.query_adxl345_end_cmd = None
        self.query_adxl345_status_cmd = None

        # ADXL345 data handling.
        self.lock = threading.Lock()
        self.data = []
        self.start_status = None
        self.last_error_count = 0

        # Reactor dispatch.
        self.method_connect = 0
        self.method_start = 1
        self.method_stop = 2
        self.method_close = 3
        self.pipe_r, self.pipe_w = os.pipe()
        self.queue = queue.Queue()
        self.pipe_r_handler = self.reactor.register_fd(self.pipe_r, self._process_pipe)
        threading.Thread(target=self._reactor_run).start()

        os.write(self.pipe_w, bytes([self.method_connect]))
        self.err = self.queue.get()

    def _reactor_run(self):
        self.reactor.run()

        self.ser.disconnect()
        self.reactor.finalize()
        self.queue.put(0)

    def _process_pipe(self, evt_time):
        method = ord(os.read(self.pipe_r, 1))
        if method == self.method_connect:
            return self.queue.put(self._connect())
        if method == self.method_start:
            return self.queue.put(self._start())
        if method == self.method_stop:
            return self.queue.put(self._stop())
        if method == self.method_close:
            return self._close()

    def close(self):
        os.write(self.pipe_w, bytes([self.method_close]))
        self.queue.get()

    def _close(self):
        self.reactor.unregister_fd(self.pipe_r_handler)
        self.reactor.end()

    def _connect(self):
        # Klipper general initialization.
        self.ser.connect_uart(self.serial, 250000)
        self.clocksync.connect(self.ser)
        self.ser.handle_default = self._handle_default
        self._setupMCU()

        msgparser = self.ser.get_msgparser()
        version, build_versions = msgparser.get_version_info()
        logging.info("%s / %s", version, build_versions)

        # ADXL345 specific initialization.
        self.ser.register_response(self._handle_data, "adxl345_data", self.adxl_oid)
        self._setupCommands()
        self._setupADXL345()

    def start(self):
        os.write(self.pipe_w, bytes([self.method_start]))
        err = self.queue.get()
        if err:
            return err
        return None

    def _start(self):
        # Store the current time and sequence ID,
        # so that we can compute the timestamp for each datum later.
        self.start_status, err = self._query_status()
        if err:
            return err

        systime = self.reactor.monotonic()
        ept = self.clocksync.estimated_print_time(systime) + klippy.adxl345.MIN_MSG_TIME
        reqclock = self.clocksync.print_time_to_clock(ept)
        data_rate = 3200
        rest_ticks = self.clocksync.print_time_to_clock(4. / data_rate)
        
        # Send query command and collect data in the background.
        with self.lock:
            self.data = []
        self.query_adxl345_cmd.send([self.adxl_oid, reqclock, rest_ticks])

        return None

    def stop(self):
        os.write(self.pipe_w, bytes([self.method_stop]))
        return self.queue.get()

    def _stop(self):
        status = self.query_adxl345_end_cmd.send([self.adxl_oid, 0, 0])
        samples = self._extract_samples(status)
        return samples

    def _query_status(self):
        for retry in range(5):
            params = self.query_adxl345_status_cmd.send([self.adxl_oid], minclock=0)
            fifo = params["fifo"] & 0x7f
            if fifo <= 32:
                return params, None
        return None, f"Unable to query adxl345 fifo"

    # def _update_clock(self):
    #     for retry in range(5):
    #         params = self.query_adxl345_status_cmd.send([self.adxl_oid], minclock=0)
    #         fifo = params['fifo'] & 0x7f
    #         if fifo <= 32:
    #             break
    #     else:
    #         return "Unable to query adxl345 fifo"
    #     mcu_clock = self.clocksync.clock32_to_clock64(params['clock'])
    #     sequence = (self.last_sequence & ~0xffff) | params['next_sequence']
    #     if sequence < self.last_sequence:
    #         sequence += 0x10000
    #     self.last_sequence = sequence
    #     buffered = params["buffered"]
    #     duration = params['query_ticks']
    #     
    #     msg_count = (sequence * klippy.adxl345.SAMPLES_PER_BLOCK + buffered // klippy.adxl345.BYTES_PER_SAMPLE + fifo)
    #     # The "chip clock" is the message counter plus .5 for average
    #     # inaccuracy of query responses and plus .5 for assumed offset
    #     # of adxl345 hw processing time.
    #     chip_clock = msg_count + 1
    #     self.clocksyncreg.update(mcu_clock + duration // 2, chip_clock)
    #     
    #     return None

    def _extract_samples(self, stop_status):
        with self.lock:
            raw_samples = self.data
            self.data = []

        startT = self.start_status["clock"] + self.start_status["query_ticks"]
        startT = self.clocksync.clock_to_print_time(startT)
        endT = stop_status["clock"] + stop_status["query_ticks"]
        endT = self.clocksync.clock_to_print_time(endT)
        duration = endT - startT
        startSeq = raw_samples[0]["sequence"]
        endSeq = raw_samples[len(raw_samples)-1]["sequence"]
        numDatum = (endSeq - startSeq) * klippy.adxl345.SAMPLES_PER_BLOCK
        tPerD = duration / numDatum

        (x_pos, x_scale), (y_pos, y_scale), (z_pos, z_scale) = [(0, klippy.adxl345.SCALE_XY), (1, klippy.adxl345.SCALE_XY), (2, klippy.adxl345.SCALE_Z)]
        
        count = 0
        samples = [None] * (len(raw_samples) * klippy.adxl345.SAMPLES_PER_BLOCK)
        for params in raw_samples:
            d = bytearray(params['data'])
            for i in range(len(d) // klippy.adxl345.BYTES_PER_SAMPLE):
                d_xyz = d[i*klippy.adxl345.BYTES_PER_SAMPLE:(i+1)*klippy.adxl345.BYTES_PER_SAMPLE]
                xlow, ylow, zlow, xzhigh, yzhigh = d_xyz
                if yzhigh & 0x80:
                        continue
                rx = (xlow | ((xzhigh & 0x1f) << 8)) - ((xzhigh & 0x10) << 9)
                ry = (ylow | ((yzhigh & 0x1f) << 8)) - ((yzhigh & 0x10) << 9)
                rz = ((zlow | ((xzhigh & 0xe0) << 3) | ((yzhigh & 0xe0) << 6))
                        - ((yzhigh & 0x40) << 7))
                raw_xyz = (rx, ry, rz)
                x = round(raw_xyz[x_pos] * x_scale, 6)
                y = round(raw_xyz[y_pos] * y_scale, 6)
                z = round(raw_xyz[z_pos] * z_scale, 6)
                # ptime = round(time_base + (msg_cdiff + i) * inv_freq, 6)
                ptime = round(startT + count * tPerD, 6)
                samples[count] = (ptime, x, y, z)
                count += 1
        del samples[count:]
        samples = np.array(samples)
        return samples

    # def _get_time_translation(self):
    #     base_mcu, base_chip, inv_cfreq = self.clocksyncreg.get_clock_translation()
    #     base_time = self.clocksync.clock_to_print_time(base_mcu)
    #     inv_freq = self.clocksync.clock_to_print_time(base_mcu + inv_cfreq) - base_time
    #     return base_time, base_chip, inv_freq

    def _handle_default(self, params):
        pass

    def _handle_data(self, params):
        with self.lock:
            self.data.append(params)

    def _read_reg(self, reg):
        params = self.spi_transfer_cmd.send([self.spi_oid, [reg | klippy.adxl345.REG_MOD_READ, 0x00]], minclock=0, reqclock=0)
        response = bytearray(params['response'])
        return response[1]

    def _set_reg(self, reg, val):
        self.spi_send_cmd.send([self.spi_oid, [reg, val & 0xFF]], minclock=0, reqclock=0)
        stored_val = self._read_reg(reg)
        if stored_val != val:
            err = f"{stored_val} {val} {inspect.getframeinfo(inspect.currentframe())}"
            return err
        return None

    def _getMCUConfig(self):
        cmd = klippy.mcu.CommandQueryWrapper(self.ser, "get_config", "config is_config=%c crc=%u is_shutdown=%c move_count=%hu", )
        params = cmd.send()
        return params

    def _getSerial(self, port):
        usbNum = int(port[0])
        usbX = f"usb{usbNum}"

        for i in range(20):
            tty, err = getTTY(port)
            if not err:
                serial = f"/dev/{tty}"
                return serial, None

            time.sleep(0.1)

        return "", f"no serial {port}"

    def _setupCommands(self):
        cq = self.ser.alloc_command_queue()
        self.spi_send_cmd = klippy.mcu.CommandWrapper(self.ser, "spi_send oid=%c data=%*s", cq)
        self.spi_transfer_cmd = klippy.mcu.CommandQueryWrapper(self.ser, "spi_transfer oid=%c data=%*s", "spi_transfer_response oid=%c response=%*s", self.spi_oid, cq, False, None)
        self.query_adxl345_cmd = klippy.mcu.CommandWrapper(self.ser, "query_adxl345 oid=%c clock=%u rest_ticks=%u", cq)
        self.query_adxl345_end_cmd = klippy.mcu.CommandQueryWrapper(
            self.ser,
            "query_adxl345 oid=%c clock=%u rest_ticks=%u",
            "adxl345_status oid=%c clock=%u query_ticks=%u next_sequence=%hu buffered=%c fifo=%c limit_count=%hu",
            self.adxl_oid, cq)
        self.query_adxl345_status_cmd = klippy.mcu.CommandQueryWrapper(
            self.ser,
            "query_adxl345_status oid=%c",
            "adxl345_status oid=%c clock=%u query_ticks=%u next_sequence=%hu buffered=%c fifo=%c limit_count=%hu",
            self.adxl_oid, cq)

    def _setupMCU(self):
        mcuConfig = self._getMCUConfig()
        if mcuConfig["is_config"]:
            return

        configCmds = [
                f"allocate_oids count=16",
                f"config_spi oid={self.spi_oid} pin=gpio9 cs_active_high=0",
                f"spi_set_software_bus oid={self.spi_oid} miso_pin=gpio12 mosi_pin=gpio11 sclk_pin=gpio10 mode=3 rate=500000",
                f"config_adxl345 oid={self.adxl_oid} spi_oid={self.spi_oid}",
        ]

        encodedConfig = "\n".join(configCmds).encode()
        configCRC = zlib.crc32(encodedConfig) & 0xffffffff
        configCmds.append(f"finalize_config crc={configCRC}")

        for c in configCmds:
            self.ser.send(c)

    def _setupADXL345(self):
        dev_id = self._read_reg(klippy.adxl345.REG_DEVID)
        if dev_id != klippy.adxl345.ADXL345_DEV_ID:
            return f"{dev_id} {klippy.adxl345.ADXL345_DEV_ID} {inspect.getframeinfo(inspect.currentframe())}"
        err = self._set_reg(klippy.adxl345.REG_POWER_CTL, 0x00)
        if err:
            return err
        err = self._set_reg(klippy.adxl345.REG_DATA_FORMAT, 0x0B)
        if err:
            return err
        err = self._set_reg(klippy.adxl345.REG_FIFO_CTL, 0x00)
        if err:
            return err
        err = self._set_reg(klippy.adxl345.REG_BW_RATE, klippy.adxl345.QUERY_RATES[3200])
        if err:
            return err
        err = self._set_reg(klippy.adxl345.REG_FIFO_CTL, klippy.adxl345.SET_FIFO_CTL)
        if err:
            return err
    
        return None


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


import select
import sys
import termios
import tty
class Keyboard:
    def __init__(self):
        # Make sure we get keyboard input without user pressing Enter.
        self.fd = sys.stdin.fileno()
        self.fdattr = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        self.lock = threading.Lock()
        self.kill = False

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def close(self):
        with self.lock:
            self.kill = True
        self.thread.join()

        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.fdattr)

    def get(self):
        try:
            c = self.queue.get(block=False)
        except queue.Empty as e:
            c = ""
        return c

    def _run(self):
        while True:
            with self.lock:
                if self.kill:
                    return

            ok, _, _, = select.select([sys.stdin], [], [], 0.1)
            if not ok:
                continue
            c = sys.stdin.read(1)

            self.queue.put(c, block=False)


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    keyboard = Keyboard()

    outDir = "data"

    usbPort = "1-5"
    adxl345 = ADXL345(usbPort)
    if adxl345.err:
        raise Exception(foo.err)

    while True:
        kb = keyboard.get()
        if kb == "q":
            break

        # Collect data.
        adxl345.start()
        logging.info("start")
        for i in range(3):
            time.sleep(1)
            logging.info("%d", i+1)
        logging.info("stop")
        data = adxl345.stop()

        # No knock, ignore this data.
        xyz = data[:, 1:]
        if np.max(xyz) - np.mean(xyz) < 20000:
            continue

        # Write data to disk.
        outPath = os.path.join(outDir, f"{int(time.time())}.csv")
        with open(outPath, "w") as f:
            f.write("t,x,y,z\n")
            for d in data:
                f.write(f"{d[0]},{d[1]},{d[2]},{d[3]}\n")
        os.system(f"chown topunion {outPath}")
        logging.info("written %s", outPath)

    adxl345.close()
    keyboard.close()


if __name__ == "__main__":
    main()
