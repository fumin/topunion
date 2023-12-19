package camserver

import (
	"bytes"
	"camserver/util"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"

	"github.com/pkg/errors"
)

type CountConfig struct {
	Height int
	Width  int
	Device string
	Mask   struct {
		Enable bool
		X      int
		Y      int
		Width  int
		Height int
		Shift  int
	}
	Yolo struct {
		Weights string
		Size    int
	}
}

type CountOutput struct {
	Passed int
}

func runCount(ctx context.Context, script, dst, src string, cfg CountConfig) (CountOutput, error) {
	var err error
	cfg.Height, cfg.Width, err = util.GetVideoSize(ctx, src)
	if err != nil {
		return CountOutput{}, errors.Wrap(err, "")
	}
	cfgB, err := json.Marshal(cfg)
	if err != nil {
		return CountOutput{}, errors.Wrap(err, "")
	}
	cmd := exec.CommandContext(ctx, "python", script, "-dst="+dst, "-src="+src, "-c="+string(cfgB))
	stdout, stderr := bytes.NewBuffer(nil), bytes.NewBuffer(nil)
	cmd.Stdout, cmd.Stderr = stdout, stderr

	if err = cmd.Run(); err != nil {
		outB, errB := stdout.Bytes(), stderr.Bytes()
		return CountOutput{}, errors.Wrap(err, fmt.Sprintf("\"%s\" \"%s\"", outB, errB))
	}
	outB, errB := stdout.Bytes(), stderr.Bytes()

	resp := struct {
		Status int
		Body   CountOutput
	}{}
	if err := json.Unmarshal(outB, &resp); err != nil {
		return CountOutput{}, errors.Wrap(err, fmt.Sprintf("\"%s\" \"%s\"", outB, errB))
	}
	if resp.Status != http.StatusOK {
		return CountOutput{}, errors.Errorf("\"%s\" \"%s\"", outB, errB)
	}
	return resp.Body, nil
}
