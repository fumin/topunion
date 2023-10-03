package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"time"

	"github.com/pkg/errors"
)

type Quiter struct {
	req  chan struct{}
	resp chan error
}

func NewQuiter() *Quiter {
	q := &Quiter{}
	q.req = make(chan struct{})
	q.resp = make(chan error)
	return q
}

func (q *Quiter) Loop(fn func(chan struct{}) error) {
	go func() {
		for {
			err := fn(q.req)
			if err != nil {
				log.Printf("%+v", err)
			}

			select {
			case <-q.req:
				q.resp <- err
			default:
			}
		}
	}()
}

func (q *Quiter) Quit() error {
	close(q.req)
	return <-q.resp
}

func RunProc(quit chan struct{}, duration time.Duration, shutdown func(stdin io.WriteCloser), program string, arg ...string) error {
	cmd := exec.Command(program, arg...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return errors.Wrap(err, "")
	}
	if err := cmd.Start(); err != nil {
		return errors.Wrap(err, "")
	}
	waitC := make(chan error)
	go func() {
		err := cmd.Wait()
		waitC <- err
	}()

	// Let process run for either its duration or early quited.
	select {
	case err := <-waitC:
		return err
	case <-time.After(duration):
	case <-quit:
	}

	// Attempt to cleanly stop process.
	shutdown(stdin)
	// Wait a while for the process to exit.
	const exitWaitSecs = 3
	select {
	case err := <-waitC:
		return err
	case <-time.After(exitWaitSecs * time.Second):
	}

	// Force kill process.
	msg := fmt.Sprintf("unable to exit in %d seconds", exitWaitSecs)
	if err := cmd.Process.Kill(); err != nil {
		msg += fmt.Sprintf(", kill err: %s", err)
	}
	return errors.Errorf(msg)
}

func run(quit chan struct{}) error {
	now := time.Now()
	const everySec = 5
	diff := everySec - (now.Second() % everySec)
	endT := now.Add(time.Duration(diff) * time.Second)
	duration := endT.Sub(now)

	shutdown := func(stdin io.WriteCloser) {
		stdin.Write([]byte("q"))
	}

	return RunProc(quit, duration, shutdown, "./myffmpeg")
}

func Quit(s *Server, w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := s.Server.Shutdown(ctx); err != nil {
		log.Printf("%+v", err)
	}
}

type Server struct {
	ServeMux *http.ServeMux
	Server   http.Server
}

func NewServer(dir, addr string) (*Server, error) {
	s := &Server{}
	s.ServeMux = http.NewServeMux()
	s.Server.Addr = addr
	s.Server.Handler = s.ServeMux

	handleFunc(s, "/Quit", Quit)

	return s, nil
}

func handleFunc(s *Server, httpPath string, fn func(*Server, http.ResponseWriter, *http.Request)) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		fn(s, w, r)
	}
	s.ServeMux.HandleFunc(httpPath, handler)
}

func handleJSON(s *Server, httpPath string, fn func(*Server, http.ResponseWriter, *http.Request) (interface{}, error)) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		res, err := fn(s, w, r)
		if err != nil {
			msg := struct {
				Error struct {
					Msg string
				}
			}{}
			msg.Error.Msg = fmt.Sprintf("%+v", err)
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(msg)
			return
		}
		if err := json.NewEncoder(w).Encode(res); err != nil {
			log.Printf("%+v", err)
			return
		}
	}
	s.ServeMux.HandleFunc(httpPath, handler)
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	quit := NewQuiter()
	quit.Loop(run)

	addr := ":8080"
	server, err := NewServer(".", addr)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("listening at %s", server.Server.Addr)
	if err := server.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}

	if err := quit.Quit(); err != nil {
		log.Printf("%+v", err)
	}

	return nil
}
