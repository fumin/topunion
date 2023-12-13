package util

import (
	"context"
	"io"
	"math/rand"
	"os"
	"time"

	"github.com/pkg/errors"

	"camserver/ffmpeg"
)

const (
	FormatDate     = "20060102"
	FormatDateHour = "20060102_15"
	FormatDatetime = "20060102_150405"

	DoneFilename   = "done.txt"
	StdoutFilename = "stdout.txt"
	StderrFilename = "stderr.txt"
	StatusFilename = "status.txt"
)

var (
	TaipeiTZ = time.FixedZone("Asia/Taipei", int((8 * time.Hour).Seconds()))
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

func RunID() string {
	return time.Now().In(time.UTC).Format(FormatDatetime) + "_" + RandID()
}

func ParseRunTime(id string) (time.Time, error) {
	l := len(FormatDatetime)
	if len(id) < l {
		return time.Time{}, errors.Errorf("too short %d", len(id))
	}
	t, err := time.ParseInLocation(FormatDatetime, id[:l], time.UTC)
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	return t, nil
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

func ReadJSONFile(fpath string, v interface{}) error {
	b, err := os.ReadFile(fpath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if err := json.Unmarshal(b, v); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func ReadVideoDuration(ctx context.Context, r io.Reader) (float64, error) {
	f, err := os.CreateTemp("", "")
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	defer os.Remove(f.Name())
	if _, err := io.Copy(f, r); err != nil {
		return -1, errors.Wrap(err, "")
	}
	if err := f.Close(); err != nil {
		return -1, errors.Wrap(err, "")
	}

	probe, err := ffmpeg.FFProbe(ctx, []string{f.Name()})
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	return probe.Format.Duration, nil
}

func GetDoneTry(root string) (string, error) {
        retries, err := os.ReadDir(root)
        if err != nil {
                return "", errors.Wrap(err, "")
        }
        for j := len(retries) - 1; j >= 0; j-- {
                retry := retries[j]

                donePath := filepath.Join(root, retry.Name(), DoneFilename)
                if _, err := os.Stat(donePath); err != nil {
                        continue
                }
                return retry.Name(), nil
        }
	return "", errors.Errorf("not found")
}
