import logging
import os
import time

import pyrealsense2 as rs
import numpy as np
import cv2


def now_millisecs():
    return int(time.time() * 1000)


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    outDir = "output1111"
    os.makedirs(outDir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    colorizer = rs.colorizer()
    profile = pipeline.start(config)

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        logging.info("depth scale: %f", depth_scale)

        while True:
            frames = pipeline.wait_for_frames()
            fColor = frames.get_color_frame()
            fDepth = frames.get_depth_frame()
            fir1 = frames.get_infrared_frame(1)
            fir2 = frames.get_infrared_frame(2)
            if not (fColor and fDepth and fir1 and fir2):
                continue
            iColor = np.asanyarray(fColor.get_data())
            fDepth = colorizer.colorize(fDepth)
            iDepth = np.asanyarray(fDepth.get_data())
            ir1 = np.asanyarray(fir1.get_data())
            ir2 = np.asanyarray(fir2.get_data())

            depth_colormap = iDepth
            cv2.imshow("color", iColor)
            cv2.imshow("depth", depth_colormap)
            cv2.imshow("ir1", ir1)
            cv2.imshow("ir2", ir2)

            logging.info("%s %s", iColor.dtype, iColor.shape)

            t = now_millisecs()
            cv2.imwrite(os.path.join(outDir, f"{t}_c.png"), iColor)
            cv2.imwrite(os.path.join(outDir, f"{t}_d.png"), depth_colormap)
            cv2.imwrite(os.path.join(outDir, f"{t}_ir1.png"), ir1)
            cv2.imwrite(os.path.join(outDir, f"{t}_ir2.png"), ir2)

            if (cv2.waitKey(1) & 0xff) == ord('q'):
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
