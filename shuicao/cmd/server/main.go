package main

import (
	"flag"
	"image"
	"log"

	"github.com/bluenviron/gortsplib/v4"
	gortsplib_base "github.com/bluenviron/gortsplib/v4/pkg/base"
	gortsplib_format "github.com/bluenviron/gortsplib/v4/pkg/format"
	"github.com/bluenviron/gortsplib/v4/pkg/format/rtph264"
	"github.com/pion/rtp"
	"github.com/pkg/errors"

	"shuicao/h264"
)

func onImg(img image.Image) {
	log.Printf(img.Bounds())
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("+v", err)
	}
}

func mainWithErr() error {
	src := `rtsp://admin:0000@192.168.1.121:8080/h264_ulaw.sdp`
	return nil
}
