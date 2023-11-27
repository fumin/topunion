package camserver

import (
	_ "embed"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

//go:embed count.py
var countPY string

//go:embed util.py
var utilPY string

type Scripts struct {
	Count string
	Util  string
}

func NewScripts(dir string) (Scripts, error) {
	s := Scripts{
		Count: filepath.Join(dir, "count.py"),
		Util:  filepath.Join(dir, "util.py"),
	}

	if err := os.MkdirAll(dir, 0775); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	if err := os.WriteFile(s.Count, []byte(countPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}
	if err := os.WriteFile(s.Util, []byte(utilPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	return s, nil
}
