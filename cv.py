import os
import time
import cv2


def now_millisecs():
	return time.time() * 1000


def main():
	dir = "data"
	if not os.path.exists(dir):
		os.makedirs(dir)

	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 30)
	if not cap.isOpened():
		raise Exception("not opened")

	t = now_millisecs()
	while True:
		ret, frame = cap.read()
		cv2.imshow("frame", frame)

		if now_millisecs() - t > 250:
			t = now_millisecs()
			fpath = os.path.join(dir, f"{t}.jpg")
			cv2.imwrite(fpath, frame)

		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

main()
