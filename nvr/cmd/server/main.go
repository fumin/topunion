package main

import (
	_ "embed"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/pkg/errors"

	"nvr/server"
)

var (
	serverDir = flag.String("d", "serverData", "server directory")
)

func daily(f func() error) {
	go func() {
		for {
			if err := f(); err != nil {
				log.Printf("%+v", err)
			}

			now := time.Now()
			next := time.Date(now.Year(), now.Month(), now.Day(), 1, 0, 0, 0, now.Location())
			if next.Before(now) {
				next = next.AddDate(0, 0, 1)
			}
			<-time.After(next.Sub(now))
		}
	}()
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	addr := ":8080"
	s, err := server.NewServer(*serverDir, addr)
	if err != nil {
		return errors.Wrap(err, "")
	}

	daily(func() error { return s.DeleteOldVideos(10 * 24 * time.Hour) })

	log.Printf("listening at %s", s.Server.Addr)
	if err := s.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}
	return nil
}
