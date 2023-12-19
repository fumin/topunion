package config

import (
	"camserver"
	"camserver/cuda"
	"camserver/util"
	_ "embed"
	"fmt"
	"sync"

	"github.com/pkg/errors"
)

//go:embed dev.json
var Dev []byte

type CameraConfig struct {
	ID     string
	Height int
	Width  int
	Count  camserver.CountConfig
}

type Config struct {
	Name string
	// Address to listen to.
	Addr string

	Camera []CameraConfig

	SqliteMaxOpenConns int
}

type PersistedConfig struct {
	sync.RWMutex
	path   string
	config Config

	cameraID map[string]int
}

func NewPersistedConfig(fpath string) (*PersistedConfig, error) {
	pc := &PersistedConfig{path: fpath, cameraID: make(map[string]int)}
	if err := util.ReadJSONFile(pc.path, &pc.config); err != nil {
		return nil, errors.Wrap(err, "")
	}

	for i, camCfg := range pc.config.Camera {
		if err := util.IsAlphaNumeric(camCfg.ID); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("\"%s\" %#v", camCfg.ID, camCfg))
		}
		if _, ok := pc.cameraID[camCfg.ID]; ok {
			return nil, errors.Errorf("duplicate camera \"%s\" %#v", camCfg.ID, camCfg)
		}

		pc.cameraID[camCfg.ID] = i
	}

	if pc.config.Name == "dev" && cuda.IsAvailable() {
		for i := range pc.config.Camera {
			pc.config.Camera[i].Count.Device = "cuda:0"
		}
	}

	return pc, nil
}

func (pc *PersistedConfig) Addr() string {
	pc.RLock()
	defer pc.RUnlock()
	return pc.config.Addr
}

func (pc *PersistedConfig) SqliteMaxOpenConns() int {
	pc.RLock()
	defer pc.RUnlock()
	return pc.config.SqliteMaxOpenConns
}

func (pc *PersistedConfig) GetCamera(id string) (CameraConfig, bool) {
	pc.RLock()
	defer pc.RUnlock()
	idx, ok := pc.cameraID[id]
	if !ok {
		return CameraConfig{}, false
	}
	return pc.config.Camera[idx], true
}

func (pc *PersistedConfig) SetCamera(id string, camCfg CameraConfig) error {
	pc.Lock()
	defer pc.Unlock()
	idx, ok := pc.cameraID[id]
	if !ok {
		return errors.Errorf("not found")
	}
	pc.config.Camera[idx] = camCfg

	if err := util.WriteJSONFile(pc.path, pc.config); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (pc *PersistedConfig) CameraIDs() []string {
	pc.RLock()
	defer pc.RUnlock()
	ids := make([]string, 0, len(pc.config.Camera))
	for _, c := range pc.config.Camera {
		ids = append(ids, c.ID)
	}
	return ids
}
