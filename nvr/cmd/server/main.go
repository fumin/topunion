package main

import (
	"context"
	_ "embed"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/pkg/errors"

	"nvr/server"
)

var (
	serverDir = flag.String("d", "devData", "server directory")
	addr      = flag.String("a", ":8080", "address to listen")
	multicast = flag.String("m", "239.0.0.0/28", "multicast subnet")
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
	s, err := server.NewServer(*serverDir, *addr, *multicast)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer s.Close()

	// Run background jobs.
	daily(func() error { return s.DeleteOldVideos(10 * 24 * time.Hour) })

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
	log.Printf("listening at %s", s.Server.Addr)
	if err := s.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}
	<-idleConnsClosed

	return nil
}
