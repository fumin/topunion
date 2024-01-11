// http://www.e7n.ch/data/e10.pdf
// https://www.iacr.org/archive/crypto2008/51570204/51570204.pdf
package main

import (
	"flag"
	"log"

	"github.com/dimchansky/keeloq-go"
)

func intBits(x uint64) []int {
	bits := make([]int, 64)
	for i := 0; i < 64; i++ {
		bits[len(bits)-1-i] = int((x >> i) & 1)
	}
	return bits
}

func bitsInt(bits []int) uint64 {
	var n uint64
	for i := len(bits) - 1; i >= 0; i-- {
		digit := uint64(bits[i] & 1)
		n |= (digit << (len(bits) - 1 - i))
	}
	return n
}

func packPlain(fn, serial, counter uint32) uint32 {
	fn = (fn & ((1 << 4) - 1))
	discmn := serial & ((1 << 10) - 1)
	counter = (counter & ((1 << 16) - 1))

	var plain uint32
	plain |= fn << (32 - 4)
	plain |= discmn << 16
	plain |= counter
	return plain
}

func unpackPlain(plain uint32) (uint32, uint32, uint32) {
	fn := plain >> (32 - 4)

	discmn := plain >> 16
	discmn &= ((1 << 12) - 1)

	counter := plain & ((1 << 16) - 1)

	return fn, discmn, counter
}

type seedpair struct {
	seed2 uint32
	seed1 uint32
}

func newSeed1(serial, pad2, pad1 uint32) seedpair {
	pad2 = (pad2 & ((1 << 4) - 1))
	pad2 = pad2 << (32 - 4)

	pad1 = (pad1 & ((1 << 4) - 1))
	pad1 = pad1 << (32 - 4)

	serial = (serial & ((1 << 28) - 1))

	var pair seedpair
	pair.seed2 |= pad2
	pair.seed2 |= serial

	pair.seed1 |= pad1
	pair.seed1 |= serial

	return pair
}

func newSeed12(serial, pad2, pad1 uint32) seedpair {
	pad2 = (pad2 & ((1 << 4) - 1))
	pad1 = (pad1 & ((1 << 4) - 1))

	serial = (serial & ((1 << 28) - 1))
	serial = serial << 4

	var pair seedpair
	pair.seed2 |= pad2
	pair.seed2 |= serial

	pair.seed1 |= pad1
	pair.seed1 |= serial

	return pair
}

func newSeed2(serial, randBits uint32) seedpair {
	serial = (serial & ((1 << 28) - 1))
	// serial |= (6 << 28)

	var pair seedpair
	pair.seed2 |= serial
	pair.seed1 |= randBits
	return pair
}

func newSeed3(serial, randBits uint32) seedpair {
	var mask uint32 = (1 << 12) - 1
	mask = mask << 16
	serial &= mask

	rand16 := randBits & ((1 << 16) - 1)

	var pair seedpair
	pair.seed2 |= serial
	pair.seed2 |= rand16

	pair.seed1 |= randBits

	return pair
}

func newSeed4(randBits uint32) seedpair {
	rand28 := randBits & ((1 << 28) - 1)
	var pair seedpair
	pair.seed2 |= rand28
	pair.seed1 |= randBits
	return pair
}

func concat(x32, y32 uint32) uint64 {
	x, y := uint64(x32), uint64(y32)
	return (x << 32) | y
}

func newDeviceKeyDecrypt(seed seedpair, masterKey uint64) uint64 {
	key2 := keeloq.Decrypt(seed.seed2, masterKey)
	key1 := keeloq.Decrypt(seed.seed1, masterKey)
	return concat(key2, key1)
}

func newDeviceKeyXOR(seed seedpair, masterKey uint64) uint64 {
	m2 := uint32(masterKey >> 32)
	m1 := uint32(masterKey & ((1 << 32) - 1))

	key2 := seed.seed2 ^ m2
	key1 := seed.seed1 ^ m1
	return concat(key2, key1)
}

func reverse8(x uint32) uint32 {
	x = (x&0xF0)>>4 | (x&0x0F)<<4
	x = (x&0xCC)>>2 | (x&0x33)<<2
	x = (x&0xAA)>>1 | (x&0x55)<<1
	return x
}

func reverse32(x uint32) uint32 {
	var b [4]uint32
	b[0] = (x >> (32 - 8)) & 0xFF
	b[1] = (x >> (32 - 16)) & 0xFF
	b[2] = (x >> (32 - 24)) & 0xFF
	b[3] = (x >> (32 - 32)) & 0xFF

	return (reverse8(b[3]) << 24) | (reverse8(b[2]) << 16) | (reverse8(b[1]) << 8) | (reverse8(b[0]))
}

func allSame(xs [][2]uint32) bool {
	for i, x := range xs[:len(xs)-1] {
		if xs[i+1][0] != x[0] {
			return false
		}

		diff := int(reverse32(xs[i+1][1])) - int(reverse32(x[1]))
		if diff < 0 {
			diff = -diff
		}
		if diff > 0xFF {
			return false
		}
	}
	return true
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)
	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	serials := []uint32{0, 0xFFFFFFFF, 0x5B0E, 0x68, 0x0E5B0000, 0x68000000, reverse32(0x5B0E), reverse32(0x68), 0xBD7587AE, reverse32(0xBD7587AE)}
	for i := 0; i < 16; i++ {
		k := 0x00005B0E | (uint32(i) << 28)
		serials = append(serials, k)
		k = 0x00000068 | (uint32(i) << 28)
		serials = append(serials, k)
	}
	randomBits := []uint32{0, 0xFFFFFFFF, 0x5B0E, 0x68, 0x0E5B0000, 0x68000000, reverse32(0x5B0E), reverse32(0x68), 0xBD7587AE, reverse32(0xBD7587AE)}
	masterKeys := []uint64{0, 0xFFFFFFFFFFFFFFFF, 0x000002A000000356, 0x00000356000002A0, 0xA002000056030000, 0x56030000A0020000, concat(reverse32(0x000002A0), reverse32(0x00000356)), concat(reverse32(0x00000356), reverse32(0x000002A0))}
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			m2 := 0x000002A0 | (uint32(i) << 28)
			m1 := 0x00000356 | (uint32(j) << 28)
			masterKeys = append(masterKeys, concat(m2, m1))
			masterKeys = append(masterKeys, concat(m1, m2))
		}
	}

	deviceKeys := make([]uint64, 0)
	deviceKeys = append(deviceKeys, 0)
	deviceKeys = append(deviceKeys, 0xFFFFFFFFFFFFFFFF)
	deviceKeys = append(deviceKeys, 0x000002A000000356)
	deviceKeys = append(deviceKeys, 0x00000356000002A0)
	deviceKeys = append(deviceKeys, 0xA002000056030000)
	deviceKeys = append(deviceKeys, 0x56030000A0020000)
	deviceKeys = append(deviceKeys, concat(reverse32(0x000002A0), reverse32(0x00000356)))
	deviceKeys = append(deviceKeys, concat(reverse32(0x00000356), reverse32(0x000002A0)))
	dkfs := []func(seedpair, uint64) uint64{
		newDeviceKeyDecrypt,
		newDeviceKeyXOR,
	}
	for _, dkf := range dkfs {
		for _, serial := range serials {
			for _, masterKey := range masterKeys {
				for _, randBits := range randomBits {
					deviceKeys = append(deviceKeys, dkf(seedpair{}, masterKey))
					deviceKeys = append(deviceKeys, dkf(seedpair{0xFFFFFFFF, 0xFFFFFFFF}, masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 0, 0), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed12(serial, 0, 0), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 4, 2), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed12(serial, 4, 2), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 2, 4), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed12(serial, 2, 4), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 2, 6), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed12(serial, 2, 6), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 6, 2), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed12(serial, 6, 2), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed2(serial, randBits), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed3(serial, randBits), masterKey))
					deviceKeys = append(deviceKeys, dkf(newSeed4(randBits), masterKey))
				}
			}
		}
	}

	var deviceKey uint64 = 0x000002A000000356
	var btn uint32 = 1
	var counter uint32 = 0x563C
	serial := serials[0]
	plain := packPlain(btn, serial, counter)
	cipher := keeloq.Encrypt(plain, deviceKey)
	log.Printf("cipher 0x%X", cipher)
	decrypted := keeloq.Decrypt(cipher, deviceKey)
	log.Printf("decrypted 0x%X", decrypted)
	log.Printf("serial    %#v", intBits(uint64(serial))[32:])
	log.Printf("decrypted %#v", intBits(uint64(decrypted))[32:])
	fnD, discmnD, counterD := unpackPlain(decrypted)
	log.Printf("%X %X %X", fnD, discmnD, counterD)

	sameBits := 16
	getFixeds := []func(uint32) [2]uint32{
		func(x uint32) [2]uint32 { return [2]uint32{x >> (32 - sameBits), x & ((1 << 16) - 1)} },
		func(x uint32) [2]uint32 { return [2]uint32{x & ((1 << sameBits) - 1), x >> 16} },
	}

	ciphers := [][2]uint32{
		{0xFCDFA07D, 0xE5A7A64A},
		{0x7DA0DFFC, 0x4AA6A7E5},
		{reverse32(0xFCDFA07D), reverse32(0xE5A7A64A)},
		{0x7DEEF151, 0x9CD4B5A4},
	}
	for i, deviceKey := range deviceKeys {
		for j, getFixed := range getFixeds {
			for k, cipherList := range ciphers {
				fixeds := make([][2]uint32, len(cipherList))
				for l, cipher := range cipherList {
					decrypted := keeloq.Decrypt(cipher, deviceKey)
					fixed := getFixed(decrypted)
					fixeds[l] = fixed
				}
				if allSame(fixeds) {
					log.Printf("!!!!!!! %d %d %d: %X %X", i, j, k, intBits(uint64(fixeds[0][0]))[64-sameBits:], deviceKey)
				}
			}
		}
	}

	return nil
}
