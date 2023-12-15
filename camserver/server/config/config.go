package server

import (
	_ "embed"
)

//go:embed dev.json
var Dev []byte

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
        // Address to listen to.
        Addr string

        Camera []CameraConfig

        // Suppport setting database max connections to alleviate sqlite issue:
        // https://github.com/mattn/go-sqlite3/issues/209
        SqliteMaxConn int
}

type PersistedConfig struct {
	sync.Mutex
	path string
	config Config
}

func NewPersistedConfig(fpath string) (*PersistedConfig, error) {
	c := &PersistedConfig{Path: fpath}
	if err := util.ReadJSONFile(c.Path, &c.Config); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return c, nil
}

func (c *PersistedConfig) Addr() string {
	c.RLock()
	defer RUnlock()
	return c.Addr
}

func (c *PersistedConfig) RangeCamera(f func(int, CameraConfig)bool) {
	c.RLock()
	defer RUnlock()
	for i, c := range c.Camera {
		if !f(i, c) {
			break
		}
	}
}

func (c *PersistedConfig) Set(id string, camCfg CameraConfig) error {

}
