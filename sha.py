import argparse
import base64
import hashlib
import logging


def main():
	logging.basicConfig()
	lg = logging.getLogger()
	lg.setLevel(logging.INFO)
	lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

	err = mainWithErr()
	if err:
		logging.fatal(err)


def mainWithErr():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", type=str, help="input filepath")
	parser.add_argument("-b", type=int, help="number of bits in digest")
	parser.add_argument("-f", type=str, help="format hex or base64")
	parser.add_argument("-e", type=str, help="expected digest")
	args = parser.parse_args()

	# Calculate hash.
	with open(args.i, "rb") as f:
		bytes = f.read()
	if args.b == 256:
		hashInBinary = hashlib.sha256(bytes).digest()
	elif args.b == 512:
		hashInBinary = hashlib.sha512(bytes).digest()
	else:
		return f"unknown digest {args.b}"

	# Convert binary hash to string format.
	if args.f == "base64":
		digest = base64.b64encode(hashInBinary).decode('utf-8')
	elif args.f == "hex":
		digest = hashInBinary.hex()
	else:
		return f"unknown format {args.f}"

	# Check hash value.
	if digest != args.e:
		return f"wrong digest {digest}"
		
	print("ok")
	return None


if __name__ == "__main__":
	main()
