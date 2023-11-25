package util

import (
	"io"
	"os"

	"github.com/pkg/errors"
)

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
