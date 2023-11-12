package main

import (
	"context"
	_ "embed"
	"encoding/json"
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
	configPath = flag.String("c", "server/config/smpte.json", "configuration path")
)

func readConfig(fpath string) (server.Config, error) {
	b, err := os.ReadFile(fpath)
	if err != nil {
		return server.Config{}, errors.Wrap(err, "")
	}
	var config server.Config
	if err := json.Unmarshal(b, &config); err != nil {
		return server.Config{}, errors.Wrap(err, "")
	}
	return config, nil
}

func printConfig(config server.Config) {
	b, err := json.Marshal(config)
	if err != nil {
		log.Fatalf("%+v", err)
	}
	log.Printf("%s", b)
}

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
	config, err := readConfig(*configPath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	printConfig(config)

	s, err := server.NewServer(config)
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
	log.Printf("listening at %s", s.C.Addr)
	if err := s.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}
	<-idleConnsClosed

	return nil
}
