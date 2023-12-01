package util

import (
	"io"
	"math/rand"
	"os"

	"github.com/pkg/errors"
)

const (
	FormatDate     = "20060102"
	FormatDatetime = "20060102_150405"

	StdoutFilename = "stdout.txt"
	StderrFilename = "stderr.txt"
	StatusFilename = "status.txt"
)

func RandID() string {
	const alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	// Ensure first letter is A to z without numbers.
	s := string(alpha[rand.Intn(len(alpha))])
	// For an alphanumeric alphabet, 8 letters roughly equals 512 bits, which is very sufficient to prevent birthday attacks.
	const n = 7
	for i := 0; i < n; i++ {
		r := rand.Intn(len(alphabet))
		c := alphabet[r]
		s += string(c)
	}
	return s
}

func IsAlphaNumeric(s string) error {
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= '0' && c <= '9':
		case c >= 'A' && c <= 'Z':
		case c >= 'a' && c <= 'z':
		default:
			return errors.Errorf("%d not alphanumeric at %d", c, i)
		}
	}
	return nil
}

func WriteFile(dstPath string, src io.Reader) error {
	dst, err := os.Create(dstPath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	_, err = io.Copy(dst, src)
	if cerr := dst.Close(); cerr != nil && err == nil {
		err = cerr
	}
	return err
}

func CopyFile(dstPath, srcPath string) error {
	src, err := os.Open(srcPath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer src.Close()
	return WriteFile(dstPath, src)

}
