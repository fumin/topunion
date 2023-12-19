package util

import (
	"camserver/ffmpeg"
	"context"
	"io"
	"os"

	"github.com/pkg/errors"
)

func ReadVideoDuration(ctx context.Context, r io.Reader) (float64, error) {
	f, err := os.CreateTemp("", "")
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	defer os.Remove(f.Name())
	if _, err := io.Copy(f, r); err != nil {
		return -1, errors.Wrap(err, "")
	}
	if err := f.Close(); err != nil {
		return -1, errors.Wrap(err, "")
	}

	probe, err := ffmpeg.FFProbe(ctx, []string{f.Name()})
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	return probe.Format.Duration, nil
}

func GetVideoSize(ctx context.Context, fpath string) (int, int, error) {
	output, err := ffmpeg.FFProbe(ctx, []string{fpath})
	if err != nil {
		return -1, -1, errors.Wrap(err, "")
	}

	var video *ffmpeg.Stream
	for _, s := range output.Streams {
		if s.CodecType == "video" {
			video = &s
			break
		}
	}
	if video == nil {
		return -1, -1, errors.Errorf("no video")
	}
	return video.Height, video.Width, nil
}
