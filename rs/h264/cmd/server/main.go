package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"net"
	"net/http"
	"net/http/pprof"
	"strconv"

	"github.com/pkg/errors"
)

func Video(s *Server, w http.ResponseWriter, r *http.Request) {
	const boundary = "MJPEGBOUNDARY"
	w.Header().Set("Content-Type", "multipart/x-mixed-replace;boundary="+boundary)

	const width, height = 1280, 720
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// const fps time.Duration = 1024
	// sleep := 1e9 / fps * time.Nanosecond
	buf := bytes.NewBuffer(nil)
	for i := 0; i < 99999999; i++ {
		// <-time.After(sleep)

		// Prepare image.
		x := i % width
		if x == 0 {
			draw.Draw(img, img.Bounds(), image.Black, image.ZP, draw.Src)
		}
		img.Set(x, height/2, color.RGBA{255, 0, 0, 255})

		buf.Reset()
		if err := jpeg.Encode(buf, img, nil); err != nil {
			return
		}

		w.Write([]byte("--" + boundary + "\n"))
		w.Write([]byte("Content-Type: image/jpeg\n"))
		w.Write([]byte("Content-Length: " + strconv.Itoa(buf.Len()) + "\n"))
		w.Write([]byte("\n"))
		w.Write(buf.Bytes())
		if _, err := w.Write([]byte("\n")); err != nil {
			return
		}
	}
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
	}{}
	indexTmpl.Execute(w, page)
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

type Server struct {
	ServeMux *http.ServeMux
}

func NewServer() *Server {
	s := &Server{}
	s.ServeMux = http.NewServeMux()

	handleFunc(s, "/Video", Video)
	s.ServeMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	handleFunc(s, "/", Index)

	return s
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	server := NewServer()

	port := 8080
	addr := fmt.Sprintf(":%d", port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("server running at %s", addr)
	if err := http.Serve(listener, server.ServeMux); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}
