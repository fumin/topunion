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

func newSeed2(serial, randBits uint32) seedpair {
	serial = (serial & ((1 << 28) - 1))

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

func newDeviceKeyDecrypt(seed seedpair, masterKey uint64) uint64 {
	key2 := uint64(keeloq.Decrypt(seed.seed2, masterKey))
	key1 := uint64(keeloq.Decrypt(seed.seed1, masterKey))

	var deviceKey uint64
	deviceKey |= (key2 << 32)
	deviceKey |= key1
	return deviceKey
}

func newDeviceKeyXOR(seed seedpair, masterKey uint64) uint64 {
	m2 := uint32(masterKey >> 32)
	key2 := uint64(seed.seed2 ^ m2)
	m1 := uint32(masterKey & ((1 << 32) - 1))
	key1 := uint64(seed.seed1 ^ m1)

	var deviceKey uint64
	deviceKey |= (key2 << 32)
	deviceKey |= key1

	log.Printf("seed2      %X", intBits(uint64(seed.seed2))[32:])
	log.Printf("masterKey2 %X", intBits(masterKey)[:32])
	log.Printf("seed1      %X", intBits(uint64(seed.seed1))[32:])
	log.Printf("masterKey1 %X", intBits(masterKey)[32:])
	log.Printf("deviceKey  %X", intBits(deviceKey))
	return deviceKey
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)
	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	const serial uint32 = 0x5B0E
	const randBits uint32 = 0x68
	const masterKey uint64 = 0x000002A000000356

	deviceKeys := make([]uint64, 0)
	deviceKeys = append(deviceKeys, 0x000002A000000356)
	dkfs := []func(seedpair, uint64) uint64{
		newDeviceKeyDecrypt,
		newDeviceKeyXOR,
	}
	for _, dkf := range dkfs {
		deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 0, 0), masterKey))
		deviceKeys = append(deviceKeys, dkf(newSeed1(serial, 4, 2), masterKey))
		deviceKeys = append(deviceKeys, dkf(newSeed2(serial, randBits), masterKey))
		deviceKeys = append(deviceKeys, dkf(newSeed3(serial, randBits), masterKey))
		deviceKeys = append(deviceKeys, dkf(newSeed4(randBits), masterKey))
	}

	var deviceKey uint64 = 0x000002A000000356
	var btn uint32 = 1
	var counter uint32 = 0x563C
	plain := packPlain(btn, serial, counter)
	cipher := keeloq.Encrypt(plain, deviceKey)
	log.Printf("cipher 0x%X", cipher)
	decrypted := keeloq.Decrypt(cipher, deviceKey)
	log.Printf("decrypted 0x%X", decrypted)
	log.Printf("serial    %#v", intBits(uint64(serial))[32:])
	log.Printf("decrypted %#v", intBits(uint64(decrypted))[32:])
	fnD, discmnD, counterD := unpackPlain(decrypted)
	log.Printf("%X %X %X", fnD, discmnD, counterD)

	ciphers := []uint32{
		0xFCDFA07D,
		0xE5A7A64A,
	}
	for i, deviceKey := range deviceKeys {
		for j, cipher := range ciphers {
			decrypted := keeloq.Decrypt(cipher, deviceKey)
			fn, discmn, counter := unpackPlain(decrypted)
			log.Printf("%d %d: %X %X %X", i, j, fn, discmn, counter)
		}
	}

	return nil
}
