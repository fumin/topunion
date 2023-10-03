package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"nvr"
	"path/filepath"
	"sync"
	"text/template"

	"github.com/pkg/errors"
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		RTSP []rtspInfo
	}{}
	page.RTSP = s.rtsp.All()
	indexTmpl.Execute(w, page)
}

type rtspInfo struct {
	Name             string
	Link             string
	NetworkInterface string
	MacAddress       string
	Username         string
	Password         string
	Port             int
	Path             string

	Pid int
}

type rtspMap struct {
	sync.RWMutex
	m map[string]rtspInfo
}

func newRTSPMap() *rtspMap {
	m := &rtspMap{}
	m.m = make(map[string]rtspInfo)
	return m
}

func (m *rtspMap) Set(name string, info rtspInfo) {
	m.Lock()
	m.m[name] = info
	m.Unlock()
}

func (m *rtspMap) All() []rtspInfo {
	m.RLock()
	all := make([]rtspInfo, 0, len(m.m))
	for _, info := range m.m {
		all = append(all, info)
	}
	m.RUnlock()
	return all
}

type Server struct {
	ServeMux *http.ServeMux

	ScriptDir string
	Scripts   nvr.Scripts

	RTSPDir string

	rtsp *rtspMap
}

func NewServer(dir string) (*Server, error) {
	s := &Server{}
	s.ServeMux = http.NewServeMux()
	s.rtsp = newRTSPMap()

	s.ScriptDir = filepath.Join(dir, "script")
	var err error
	s.Scripts, err = nvr.NewScripts(s.ScriptDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.RTSPDir = filepath.Join(dir, "rtsp")

	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) startRTSP(info rtspInfo) {
	go func() {
		outDir := filepath.Join(s.RTSPDir, info.Name)
		for {
			// ffmpeg -i "rtsp://localhost:8554/rtsp" -an -hls_time 10 -hls_list_size 0 -strftime 1 -hls_flags second_level_segment_duration -hls_segment_filename "ffmpeg/%Y%m%d_%H%M%S_%%06t.ts" "ffmpeg/out.m3u8"
			p, err := nvr.NewRTSPProc(outDir, info.Link, info.NetworkInterface, info.MacAddress, info.Username, info.Password, info.Port, info.Path)
			if err != nil {
				log.Printf("%+v", err)
				continue
			}
			info.Pid = p.Cmd.Process.Pid
			s.rtsp.Set(info.Name, info)

			<-time.After(30 * time.Second)

			go func() {
				if err := p.Quit(); err != nil {
					log.Printf("%+v", err)
				}
			}()
		}
	}()
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
	server, err := NewServer(*serverDir)
	if err != nil {
		return errors.Wrap(err, "")
	}

	vlc := rtspInfo{
		Name: "vlc",
		Link: "rtsp://localhost:8554/rtsp",
	}
	server.startRTSP(vlc)

	port := 8080
	addr := fmt.Sprintf(":%d", port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("listening at %s", addr)
	if err := http.Serve(listener, server.ServeMux); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}
