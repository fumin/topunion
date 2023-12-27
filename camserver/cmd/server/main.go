package main

import (
	"camserver/server"
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/pkg/errors"
)

var (
	dir = flag.String("d", "devData", "data directory")
)

func daily(f func() error) {
	go func() {
		for {
			if err := f(); err != nil {
				log.Printf("%+v", err)
			}

			now := time.Now()
			next := time.Date(now.Year(), now.Month(), now.Day(), 0, 1, 0, 0, now.Location())
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
	s, err := server.NewServer(*dir)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer s.Close()

	// Run background jobs.
	s.DoJobForever()
	daily(func() error { return server.DeleteOldVideos(s.VideoDir, 10*24*time.Hour) })
	daily(func() error { return server.DeleteOldVideoProcess(s.DB) })

	// Run HTTP server.
	idleConnsClosed := make(chan struct{})
	go func() {
		defer close(idleConnsClosed)

		sigint := make(chan os.Signal, 1)
		signal.Notify(sigint, os.Interrupt)
		<-sigint

		// We received an interrupt signal, shut down.
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		if err := s.Server.Shutdown(ctx); err != nil {
			log.Printf("HTTP server Shutdown: %+v", err)
		}
	}()
	log.Printf("listening on %s", s.Server.Addr)
	if err := s.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}
	<-idleConnsClosed

	return nil
}
