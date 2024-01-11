import argparse
import logging
import threading
import time

import pymodbus
from pymodbus import datastore
from pymodbus import server


def runServer(context, identity, address):
    def target():
        pymodbus.server.StartTcpServer(context=context, identity=identity, address=address)
    threading.Thread(target=target, daemon=True).start()


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-src")
    args = parser.parse_args()

    store = pymodbus.datastore.ModbusSlaveContext(
        # Discrete inputs (1 bit read-only)
        di=pymodbus.datastore.ModbusSequentialDataBlock(0, [0]*100),
        # Coil (discrete output, 1 bit read-write)
        co=pymodbus.datastore.ModbusSequentialDataBlock(0, [0]*100),
        # Input register (16 bits read-only)
        ir=pymodbus.datastore.ModbusSequentialDataBlock(0, [0]*100),
        # Holding register (16 bits read-write)
        hr=pymodbus.datastore.ModbusSequentialDataBlock(0, [0]*100),
        zero_mode=True)
    context = pymodbus.datastore.ModbusServerContext(slaves=store)

    identity = pymodbus.device.ModbusDeviceIdentification()
    identity.VendorName = 'Pymodbus'
    identity.ProductCode = 'PM'
    identity.VendorUrl = 'http://github.com/riptideio/pymodbus/'
    identity.ProductName = 'Pymodbus Server'
    identity.ModelName = 'Pymodbus Server'
    identity.MajorMinorRevision = pymodbus.__version__

    modbusHost = "127.0.0.1"
    modbusPort = 5020
    runServer(context, identity, (modbusHost, modbusPort))

    while True:
        logging.info("d %s", store.store["d"].values)
        logging.info("c %s", store.store["c"].values)
        logging.info("i %s", store.store["i"].values)
        logging.info("h %s", store.store["h"].values)
        logging.info("")
        time.sleep(1)


if __name__ == "__main__":
    main()
