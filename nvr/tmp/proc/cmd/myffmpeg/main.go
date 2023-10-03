package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/pkg/errors"
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	b := make([]byte, 1)
	for {
		n, err := os.Stdin.Read(b)
		if err != nil {
			return errors.Wrap(err, "")
		}
		if n == 0 {
			continue
		}

		if b[0] == 'q' {
			break
		}
	}

	fpath := time.Now().Format("20060102_150405.txt")
	if err := os.WriteFile(fpath, []byte("fpath"), 0644); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}
