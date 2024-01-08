// http://www.e7n.ch/data/e10.pdf
// https://www.iacr.org/archive/crypto2008/51570204/51570204.pdf
package main

import (
	"flag"
	"log"
	"strconv"

	"github.com/dimchansky/keeloq-go"
)

func intBits(x uint64) [64]uint8 {
	var bits [64]uint8
	for i := 0; i < 64; i++ {
		bits[len(bits)-1-i] = uint8((x & (uint64(1) << i)) >> i)
	}
	return bits
}

func bitsInt(bits [64]uint8) uint64 {
	var n uint64
	for i := len(bits) - 1; i >= 0; i-- {
		digit := uint64(bits[i])

		n += digit * (uint64(1) << (len(bits) - 1 - i))
	}
	return n
}

func setBits(dst [64]uint8, dstStart, src [64]uint8, srcStart, srcLen int) [64]uint8 {
	for i := 0; i < srcLen; i++ {
		dst[dstStart+i] = src[srcStart+i]
	}
	return dst
}

func setLSB(dst [64]uint8, dstStart int, src [64]uint8, n int) [64]uint8 {
	return setBits(dst, dstStart, src, len(src)-n, n)
}

func getPlain(fn, serial, counter uint32) uint32 {
	discmn := serial & 0x0FFF

	var plain uint32
	plain |= (fn & 0x0F) << (32-4)
	plain |= discmn
	return plain

	var plainBits [64]uint8

	fn := intBits(fnInt)
	plainBits = setLSB(plainBits, 32+0, fn, 4)

	serial := intBits(serialInt)
	plainBits = setLSB(plainBits, 32+6, serial, 10)

	cnt := intBits(counter)
	plainBits = setLSB(plainBits, 32+16, cnt, 16)

	plain := uint32(bitsInt(plainBits))
	return plain
}

func newSeed1(serialInt, padInt2, padInt1 uint64) (uint64, uint64) {
	serial := intBits(serialInt)
	pad2 := intBits(padInt2)
	pad1 := intBits(padInt1)

	var seed2Bits [64]uint8
	seed2Bits = setLSB(seed2Bits, 32, pad2, 4)
	seed2Bits = setLSB(seed2Bits, 32+4, serial, 28)

	var seed1Bits [64]uint8
	seed1Bits = setLSB(seed1Bits, 32, pad1, 4)
	seed1Bits = setLSB(seed1Bits, 32+4, serial, 28)

	seed2 := bitsInt(seed2Bits)
	seed1 := bitsInt(seed1Bits)
	return seed2, seed1
}

func newSeed2(serialInt, randBitsInt uint64) (uint64, uint64) {
	serial := intBits(serialInt)

	var seed2Bits [64]uint8
	seed2Bits = setLSB(seed2Bits, 32+4, serial, 28)
	seed2 := bitsInt(seed2Bits)

	return seed2, randBitsInt
}

func newSeed3(serialInt, randBitsInt uint64) (uint64, uint64) {
	serial := intBits(serialInt)
	randBits := intBits(randBitsInt)

	var seed2Bits [64]uint8
	seed2Bits = setBits(seed2Bits, 32+4, serial, 4, 12)
	seed2Bits = setLSB(seed2Bits, 32+16, randBits, 16)
	seed2 := bitsInt(seed2Bits)

	return seed2, randBitsInt
}

func newSeed4(randBitsInt uint64) (uint64, uint64) {
	randBits := intBits(randBitsInt)

	var seed2Bits [64]uint8
	seed2Bits = setLSB(seed2Bits, 32+4, randBits, 28)
	seed2 := bitsInt(seed2Bits)

	return seed2, randBitsInt
}

func newDeviceKeyDecrypt(seed2, seed1, masterKey uint64) uint64 {
	deviceKey2 := intBits(keeloq.Decrypt(seed2, masterKey))
	deviceKey1 := intBits(keeloq.Decrypt(seed1, masterKey))

	var deviceKey [64]uint8
	deviceKey = setLSB(deviceKey, 0, deviceKey2, 32)
	deviceKey = setLSB(deviceKey, 32, deviceKey1, 32)
	return bitsInt(deviceKey)
}

func newDeviceKeyXOR(seed2, seed1, masterKey uint64) uint64 {
	deviceKey2 := intBits(seed2 ^ (masterKey >> 32))
	deviceKey1 := intBits(seed1 ^ (masterKey & 0xFFFFFFFF))

	var deviceKey [64]uint8
	deviceKey = setLSB(deviceKey, 0, deviceKey2, 32)
	deviceKey = setLSB(deviceKey, 32, deviceKey1, 32)
	return bitsInt(deviceKey)
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)
	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	deviceKeys := make([]uint64, 0)
	deviceKeys = append(deviceKeys, 0x000002A000000356)
	dkfs := []func(seed1, seed2, masterKey uint64)uint64{
		newDeviceKeyDecrypt,
		newDeviceKeyXOR,
	}
	for _, dkf := range dkfs {
		deviceKeys = append(deviceKeys, dkf(newSeed1()))
		{0, 0},
		{4, 2},
	}

	var deviceKey uint64 = 0x5CEC6701B79FD949
	var plain uint32 = 0x0CA69B92

	deviceKey = 0x000002A000000356
	serial := 0x00005B0E
	counter := 0x563C
	plain = getPlain(1, serial, counter)

	cipher := keeloq.Encrypt(plain, deviceKey)
	log.Printf("cipher 0x%X", cipher)

	decrypted := keeloq.Decrypt(cipher, deviceKey)
	log.Printf("decrypted 0x%X", decrypted)

	return nil
}
