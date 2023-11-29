package server

import (
	"camserver"
	"camserver/util"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"text/template"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

const (
	FormatDate = "20060102"

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
	if err := util.IsAlphaNumeric(cam); err != nil {
		return nil, errors.Wrap(err, "")
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
	dst := filepath.Join(s.VideoDir, t.Format("2006"), t.Format(FormatDate), cam, fh.Filename)
	if err := os.MkdirAll(filepath.Dir(dst), os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}
	if err := util.WriteFile(dst, f); err != nil {
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

type Config struct {
	// Directory to store data.
	Dir string
	// Address to listen to.
	Addr string
}

type Server struct {
	C         Config
	StartTime time.Time

	ServeMux *http.ServeMux
	Server   http.Server

	ScriptDir string
	Scripts   camserver.Scripts

	DB       *sql.DB
	VideoDir string
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
