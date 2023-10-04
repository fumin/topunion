package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"nvr"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"text/template"
	"time"

	"github.com/pkg/errors"
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

func Quit(s *Server, w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := s.Server.Shutdown(ctx); err != nil {
		log.Printf("%+v", err)
	}
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		RTSP []rtspInfo
	}{}
	page.RTSP = s.rtsp.All()
	sort.Slice(page.RTSP, func(i, j int) bool { return page.RTSP[i].Name < page.RTSP[j].Name })
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

	Pid   int
	Start time.Time
}

func (info rtspInfo) getLink() (string, error) {
	if info.Link != "" {
		return info.Link, nil
	}
	return info.Link, nil
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

func (m *rtspMap) Set(info rtspInfo) {
	m.Lock()
	m.m[info.Name] = info
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
	Server   http.Server

	ScriptDir string
	Scripts   nvr.Scripts

	RTSPDir string

	rtsp *rtspMap
}

func NewServer(dir, addr string) (*Server, error) {
	s := &Server{}
	s.ServeMux = http.NewServeMux()
	s.Server.Addr = addr
	s.Server.Handler = s.ServeMux

	s.ScriptDir = filepath.Join(dir, "script")
	var err error
	s.Scripts, err = nvr.NewScripts(s.ScriptDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.RTSPDir = filepath.Join(dir, "rtsp")

	s.rtsp = newRTSPMap()

	handleFunc(s, "/Quit", Quit)
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) startRTSP(quit *nvr.Quiter, info rtspInfo) {
	// ffmpeg -i "rtsp://localhost:8554/rtsp" -an -hls_time 10 -hls_list_size 0 -strftime 1 -hls_flags append_list+second_level_segment_duration -hls_segment_filename "ffmpeg/%Y%m%d_%H%M%S_%%06t.ts" "ffmpeg/out.m3u8"

	const program = "ffmpeg"
	const hlsFlags = "" +
		// Append to the same HLS index file.
		"append_list" +
		// Use microseconds in segment filename.
		"+second_level_segment_duration"
	run := func(quit chan struct{}) error {
		link, err := info.getLink()
		if err != nil {
			return errors.Wrap(err, "")
		}

		now := time.Now().In(time.UTC)
		endT := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
		endT = endT.AddDate(0, 0, 1)
		duration := endT.Sub(now)

		dayDir := filepath.Join(s.RTSPDir, now.Format("20060102"))
		segmentFName := filepath.Join(dayDir, "%s_%%06t.ts")
		indexFName := filepath.Join(dayDir, "index.m3u8")
		arg := []string{
			"-i", link,
			// No audio.
			"-an",
			// 10 seconds per segment.
			"-hls_time", "10",
			// No limit on number of segments.
			"-hls_list_size", "0",
			// Use strftime syntax for segment filenames.
			"-strftime", "1",
			"-hls_flags", hlsFlags,
			"-hls_segment_filename", segmentFName,
			indexFName,
		}

		program := "ffmpeg"
		// program, arg = "sleep", []string{9999}
		cmd := exec.Command(program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return errors.Wrap(err, "")
		}
		outerrSize := 4 * 1024
		cmd.Stdout = nvr.NewByteQueue(outerrSize)
		cmd.Stderr = nvr.NewByteQueue(outerrSize)
		if err := cmd.Start(); err != nil {
			return errors.Wrap(err, "")
		}
		info.Pid = cmd.Process.Pid
		info.Start = now
		s.rtsp.Set(info)

		// ffmpeg quits by sending q.
		shutdown := func() error {
			_, err := stdin.Write([]byte("q"))
			return err
		}

		if err := nvr.RunProc(quit, duration, shutdown, cmd); err != nil {
			outB, errB := cmd.Stdout.Slice(), cmd.Stderr.Slice()
			return errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", outB, errB))
		}
		return nil
	}
	quit.Loop(run)
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
	addr := ":8080"
	server, err := NewServer(*serverDir, addr)
	if err != nil {
		return errors.Wrap(err, "")
	}

	vlc := rtspInfo{
		Name: "vlc",
		Link: "rtsp://localhost:8554/rtsp",
	}
	vlcRTSPQuiter := nvr.NewQuiter()
	server.startRTSP(vlcRTSPQuiter, vlc)

	log.Printf("listening at %s", server.Server.Addr)
	if err := server.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}

	if err := vlcRTSPQuiter.Quit(); err != nil {
		log.Printf("%+v", err)
	}

	return nil
}
