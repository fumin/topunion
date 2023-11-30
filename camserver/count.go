package camserver

type CountConfig struct {
	Height int
	Width  int
	Device string
	Mask   struct {
		Enable bool
		Crop   struct {
			X int
			Y int
			W int
		}
		Mask struct {
			Slope int
			Y     int
			H     int
		}
	}
	Yolo struct {
		Weights string
		Size    int
	}
}

type Counter struct {
	cancel context.CancelFunc
	cmd    *exec.Cmd
}

func newCounter(script string, cfg CountConfig) (*Counter, error) {
	cfgB, err := json.Marshal(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	cmd := exec.Command("python", script, "-c="+string(cfgB))

	c := &Counter{}
	return c, nil
}
