package camserver

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestCounter(t *testing.T) {
	t.Parallel()
	env := newEnvironment(t)
	defer env.close()

	cfg := shilinSDConfig()
	dst := filepath.Join(env.dir, "dst.ts")
	src := filepath.Join("testing", "shilin20230826_sd.mp4")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	out, err := runCount(ctx, env.scripts.Count, dst, src, cfg)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if out.Passed != 10 {
		t.Fatalf("%#v", out)
	}
}
