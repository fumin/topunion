package server

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"

	"camserver"
	"camserver/util"
)

const (
	DBFilename = "db.sqlite"
)

func UploadVideo(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	if err := r.ParseMultipartForm(32 * 1024 * 1024); err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Camera ID.
	cam := r.FormValue("c")
	if cam == "" {
		return nil, errors.Errorf("empty camera ID")
	}
	if _, ok := s.Camera[cam]; !ok {
		return nil, errors.Errorf("unknown camera")
	}

	// Uploaded file.
	f, fh, err := r.FormFile("f")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	const formatDatetime = "20060102_150405"
	if len(fh.Filename) < len(formatDatetime) {
		return nil, errors.Errorf("%d", len(fh.Filename))
	}
	t, err := time.Parse(formatDatetime, fh.Filename[:len(formatDatetime)])
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Save uploaded file.
	ext := filepath.Ext(fh.Filename)
	noext := strings.TrimSuffix(fh.Filename, ext)
	dst := filepath.Join(s.VideoDir, t.Format("2006"), t.Format(util.FormatDate), cam, noext, "raw"+ext)
	if err := os.MkdirAll(filepath.Dir(dst), os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}
	if err := util.WriteFile(dst, f); err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Send background job.
	jobArg := camserver.ProcessVideoInput{
		Camera:   cam,
		Filepath: dst,
		Time:     t,
	}
	job := Job{Func: JobProcessVideo, Arg: jobArg}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := SendJob(ctx, s.DB, job); err != nil {
		return nil, errors.Wrap(err, "")
	}

	resp := struct {
		T int64
	}{}
	resp.T = s.StartTime.Unix()
	return resp, nil
}

//go:embed tmpl/index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct{}{}
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

type CameraConfig struct {
	ID     string
	Height int
	Width  int
	Count  camserver.CountConfig
}

type Camera struct {
	Config  CameraConfig
	Counter *camserver.Counter
}

type Config struct {
	Name string
	// Directory to store data.
	Dir string
	// Address to listen to.
	Addr string

	Camera []CameraConfig
}

type Server struct {
	C         Config
	StartTime time.Time

	ServeMux *http.ServeMux
	Server   http.Server

	ScriptDir string
	Scripts   camserver.Scripts

	BackgroundProcessDir string
	DB                   *sql.DB
	VideoDir             string

	Camera map[string]Camera
}

//go:embed static
var staticFS embed.FS

func NewServer(config Config) (*Server, error) {
	s := &Server{C: config, StartTime: time.Now()}
	s.ServeMux = http.NewServeMux()
	s.Server.Addr = s.C.Addr
	s.Server.Handler = s.ServeMux

	s.ScriptDir = filepath.Join(s.C.Dir, "script")
	var err error
	s.Scripts, err = camserver.NewScripts(s.ScriptDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.BackgroundProcessDir = filepath.Join(s.C.Dir, "proc")
	if err := os.MkdirAll(s.BackgroundProcessDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	dbPath := filepath.Join(s.C.Dir, DBFilename)
	dbV := url.Values{}
	dbV.Set("_journal_mode", "WAL")
	s.DB, err = sql.Open("sqlite3", "file:"+dbPath+"?"+dbV.Encode())
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.VideoDir = filepath.Join(s.C.Dir, "video")
	if err := os.MkdirAll(s.VideoDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.Camera = make(map[string]Camera)
	for _, camCfg := range s.C.Camera {
		if err := util.IsAlphaNumeric(camCfg.ID); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("\"%s\" %#v", camCfg.ID, camCfg))
		}
		if _, ok := s.Camera[camCfg.ID]; ok {
			return nil, errors.Errorf("duplicate camera \"%s\" %#v", camCfg.ID, camCfg)
		}
		cam := Camera{Config: camCfg}

		countDir := filepath.Join(s.BackgroundProcessDir, camCfg.ID, "count")
		camCfg.Count.Height = camCfg.Height
		camCfg.Count.Width = camCfg.Width
		cam.Counter, err = camserver.NewCounter(countDir, s.Scripts.Count, camCfg.Count)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		s.Camera[cam.Config.ID] = cam
	}

	handleJSON(s, "/UploadVideo", UploadVideo)
	s.ServeMux.Handle("/static/", http.FileServer(http.FS(staticFS)))
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) Close() error {
	if err := s.DB.Close(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
