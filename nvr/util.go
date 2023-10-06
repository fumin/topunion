package nvr

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/carlmjohnson/deque"
	"github.com/pkg/errors"
)

type ByteQueue struct {
	size int
	q    *deque.Deque[byte]
}

func NewByteQueue(size int) *ByteQueue {
	q := &ByteQueue{size: size}
	q.q = deque.Make[byte](size)
	return q
}

func (q *ByteQueue) Write(p []byte) (int, error) {
	q.q.PushBackSlice(p)

	for q.q.Len() > q.size {
		q.q.RemoveFront()
	}
	return len(p), nil
}

func (q *ByteQueue) Slice() []byte {
	return q.q.Slice()
}

type quiterRunFn func(context.Context) error

func Loop(ctx context.Context, fn quiterRunFn) error {
	lastLogT := time.Now().AddDate(-1, 0, 0)
	for {
		err := fn(ctx)
		if err != nil {
			// Only print errors intermitently to prevent constantly failing functions from overwhelming our logs.
			now := time.Now()
			if now.Sub(lastLogT) > time.Minute {
				lastLogT = now
				log.Printf("%+v", err)
			}
		}

		select {
		case <-ctx.Done():
			return err
		default:
		}
	}
}

const ymdhmsFormat = "20060102_150405"

func TimeFormat(inT time.Time) string {
	t := inT.In(time.UTC)
	microSec := t.Nanosecond() / 1e3
	return t.Format(ymdhmsFormat) + "_" + fmt.Sprintf("%06d", microSec)
}

func TimeParse(name string) (time.Time, error) {
	ss := strings.Split(name, "_")
	if len(ss) != 3 {
		return time.Time{}, errors.Errorf("%d", len(ss))
	}
	t, err := time.Parse(ymdhmsFormat, ss[0]+"_"+ss[1])
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	microSec, err := strconv.Atoi(ss[2])
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	nanoSec := microSec * 1e3

	parsed := time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second(), nanoSec, t.Location())
	return parsed, nil
}
