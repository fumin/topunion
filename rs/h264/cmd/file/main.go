package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"io"
	"log"
	"os"

	"github.com/gen2brain/x264-go"
	"github.com/pkg/errors"
	"github.com/yapingcat/gomedia/go-mp4"
)

type cacheWriterSeeker struct {
	buf    []byte
	offset int
}

func newCacheWriterSeeker(capacity int) *cacheWriterSeeker {
	return &cacheWriterSeeker{
		buf:    make([]byte, 0, capacity),
		offset: 0,
	}
}

func (ws *cacheWriterSeeker) Write(p []byte) (n int, err error) {
	if cap(ws.buf)-ws.offset >= len(p) {
		if len(ws.buf) < ws.offset+len(p) {
			ws.buf = ws.buf[:ws.offset+len(p)]
		}
		copy(ws.buf[ws.offset:], p)
		ws.offset += len(p)
		return len(p), nil
	}
	tmp := make([]byte, len(ws.buf), cap(ws.buf)+len(p)*2)
	copy(tmp, ws.buf)
	if len(ws.buf) < ws.offset+len(p) {
		tmp = tmp[:ws.offset+len(p)]
	}
	copy(tmp[ws.offset:], p)
	ws.buf = tmp
	ws.offset += len(p)
	return len(p), nil
}

func (ws *cacheWriterSeeker) Seek(offset int64, whence int) (int64, error) {
	if whence == io.SeekCurrent {
		if ws.offset+int(offset) > len(ws.buf) {
			return -1, errors.New(fmt.Sprint("SeekCurrent out of range", len(ws.buf), offset, ws.offset))
		}
		ws.offset += int(offset)
		return int64(ws.offset), nil
	} else if whence == io.SeekStart {
		if offset > int64(len(ws.buf)) {
			return -1, errors.New(fmt.Sprint("SeekStart out of range", len(ws.buf), offset, ws.offset))
		}
		ws.offset = int(offset)
		return offset, nil
	} else {
		return 0, errors.New("unsupport SeekEnd")
	}
}

func main() {
	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	buf := bytes.NewBuffer(make([]byte, 0))

	opts := &x264.Options{
		Width:     640,
		Height:    480,
		FrameRate: 25,
		Tune:      "zerolatency",
		Preset:    "veryfast",
		Profile:   "baseline",
		LogLevel:  x264.LogDebug,
	}

	enc, err := x264.NewEncoder(buf, opts)
	if err != nil {
		return errors.Wrap(err, "")
	}

	mp4Buf := newCacheWriterSeeker(4096)
	muxer, err := mp4.CreateMp4Muxer(mp4Buf)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if err := muxer.WriteInitSegment(mp4Buf); err != nil {
		return errors.Wrap(err, "")
	}
	vtid := muxer.AddVideoTrack(mp4.MP4_CODEC_H264)

	img := x264.NewYCbCr(image.Rect(0, 0, opts.Width, opts.Height))
	draw.Draw(img, img.Bounds(), image.Black, image.ZP, draw.Src)

	for i := 0; i < opts.Width/2; i++ {
		img.Set(i, opts.Height/2, color.RGBA{255, 0, 0, 255})

		beforeIdx := buf.Len()

		err = enc.Encode(img)
		if err != nil {
			return errors.Wrap(err, "")
		}

		frame := buf.Bytes()[beforeIdx:]
		ptsS, dtsS := enc.GetTimestamp()
		pts, dts := uint64(ptsS), uint64(dtsS)
		if err := muxer.Write(vtid, frame, pts, dts); err != nil {
			return errors.Wrap(err, "")
		}
	}

	err = enc.Flush()
	if err != nil {
		return errors.Wrap(err, "")
	}

	err = enc.Close()
	if err != nil {
		return errors.Wrap(err, "")
	}

	if err := muxer.WriteTrailer(); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.WriteFile("out.mp4", mp4Buf.buf, 0644); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}
