package ffmpeg

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestFFProbe(t *testing.T) {
	fpath := filepath.Join("..", "testing", "shilin20230826_sd.mp4")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	output, err := FFProbe(ctx, []string{fpath})
	if err != nil {
		t.Fatalf("%+v", err)
	}

	// Check duration.
	if output.Format.Duration != 17.1 {
		t.Fatalf("%#v", output)
	}

	// Find video stream.
	var video *Stream
	for _, s := range output.Streams {
		if s.CodecType == "video" {
			video = &s
			break
		}
	}
	if video == nil {
		t.Fatalf("%#v", output)
	}

	// Check video dimensions.
	if video.Height != 480 {
		t.Fatalf("%#v", output)
	}
	if video.Width != 640 {
		t.Fatalf("%#v", output)
	}
}
