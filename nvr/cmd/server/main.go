package main

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"net/url"
	"nvr"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

const (
	PathHLSIndex = "/HLSIndex"
	PathVideo    = "/Video"
	PathServe    = "/Serve"
)

func HLSIndex(s *Server, w http.ResponseWriter, r *http.Request) {
	fpath := r.FormValue("f")

	dir := filepath.Dir(fpath)

	f, err := os.Open(fpath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	out := bytes.NewBuffer(nil)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		var outLine string
		switch {
		case strings.HasPrefix(line, "#"):
			outLine = line
		default:
			outLine = s.serveURL(filepath.Join(dir, line))
		}
		out.WriteString(outLine + "\n")
	}
	if err := scanner.Err(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Write(out.Bytes())
}

//go:embed video.html
var videoHTML string
var videoTmpl = template.Must(template.New("").Parse(videoHTML))

func Video(s *Server, w http.ResponseWriter, r *http.Request) {
	src := r.FormValue("s")
	page := struct {
		Src string
	}{}
	page.Src = src
	videoTmpl.Execute(w, page)
}

func Serve(s *Server, w http.ResponseWriter, r *http.Request) {
	fpath := r.FormValue("f")
	http.ServeFile(w, r, fpath)
}

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
	rtsps := s.rtsp.All()
	now := time.Now().In(time.UTC)
	for i, info := range rtsps {
		v := url.Values{}
		v.Set("s", s.hlsIndex(s.videoIndex(info.Name, now)))
		rtsps[i].Live = PathVideo + "?" + v.Encode()
	}
	sort.Slice(rtsps, func(i, j int) bool { return rtsps[i].Name < rtsps[j].Name })

	page := struct {
		RTSP []rtspInfo
	}{}
	page.RTSP = rtsps
	indexTmpl.Execute(w, page)
}

type arpHW struct {
	ip string
	hw string
}

func arpScan(networkInterface string) (map[string]arpHW, error) {
	program := "arp-scan"
	arg := []string{
		"--interface=" + networkInterface,
		"-l",
		// Concise output to aid parsing.
		"-x",
	}
	cmd := exec.Command(program, arg...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	hws := make(map[string]arpHW)
	scanner := bufio.NewScanner(bytes.NewBuffer(b))
	for scanner.Scan() {
		line := scanner.Text()
		cols := strings.Split(line, "\n")
		if len(cols) < 2 {
			return nil, errors.Wrap(err, fmt.Sprintf("%#v %s", cols, b))
		}

		var hw arpHW
		hw.ip = cols[0]
		hw.hw = cols[1]
		hws[hw.hw] = hw
	}
	if scanner.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return hws, nil
}

type rtspInfo struct {
	// Fields related to the source stream.
	Name             string
	Link             string
	NetworkInterface string
	MacAddress       string
	Username         string
	Password         string
	Port             int
	Path             string

	// Fields related to background recording process.
	Pid   int
	Start time.Time

	// Fields related to UI.
	Live string
}

func (info rtspInfo) getLink() (string, error) {
	if info.Link != "" {
		return info.Link, nil
	}

	hws, err := arpScan(info.NetworkInterface)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	hw, ok := hws[info.MacAddress]
	if !ok {
		return "", errors.Errorf("%#v %#v", info, hws)
	}
	info.Link = fmt.Sprintf("rtsp://%s:%s@%s:%d%s", info.Username, info.Password, hw.ip, info.Port, info.Path)

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

	VideoDir string

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

	s.VideoDir = filepath.Join(dir, "video")

	s.rtsp = newRTSPMap()

	handleFunc(s, PathHLSIndex, HLSIndex)
	handleFunc(s, PathVideo, Video)
	handleFunc(s, PathServe, Serve)
	handleFunc(s, "/Quit", Quit)
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) serveURL(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathServe + "?" + v.Encode()
}

func (s *Server) hlsIndex(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathHLSIndex + "?" + v.Encode()
}

func (s *Server) videoIndex(name string, t time.Time) string {
	dayDir := filepath.Join(s.VideoDir, name, t.Format("20060102"))
	indexFName := filepath.Join(dayDir, "index.m3u8")
	return indexFName
}

func (s *Server) startRTSP(quit *nvr.Quiter, info rtspInfo) {
	dir := filepath.Join(s.VideoDir, info.Name)
	getInput := func() (string, error) {
		return info.getLink()
	}
	onStart := func(pid int, t time.Time) {
		info.Pid = pid
		info.Start = t
		s.rtsp.Set(info)
	}
	nvr.RecordVideo(quit, dir, getInput, onStart)
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

	rtsp0 := rtspInfo{
		Name:             "rtsp0",
		NetworkInterface: "",
		MacAddress:       "",
		Username:         "admin",
		Password:         "0000",
		Port:             8080,
		Path:             "/h264_ulaw.sdp",
	}
	// rtsp0.Link = "rtsp://localhost:8554/rtsp"
	rtsp0.Link = "sample/egg.mp4"
	vlcRTSPQuiter := nvr.NewQuiter()
	server.startRTSP(vlcRTSPQuiter, rtsp0)

	log.Printf("listening at %s", server.Server.Addr)
	if err := server.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}

	if err := vlcRTSPQuiter.Quit(); err != nil {
		log.Printf("%+v", err)
	}

	return nil
}
