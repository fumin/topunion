package hls

import (
	"bytes"
	"fmt"
	"math"
)

type Segment struct {
	URL      string
	Duration float64
}

type Playlist struct {
	MediaSequence int
	Segment       []Segment
}

func (p Playlist) Bytes() []byte {
	var targetDuration float64 = -1
	for _, s := range p.Segment {
		if targetDuration < s.Duration {
			targetDuration = s.Duration
		}
	}
	// Add some margin to target duration, so that it is strictly larger.
	targetDuration = math.Ceil(targetDuration) + 1

	b := bytes.NewBuffer(nil)
	b.WriteString("#EXTM3U\n")
	b.WriteString(fmt.Sprintf("#EXT-X-TARGETDURATION:%f\n", targetDuration))
	b.WriteString("#EXT-X-VERSION:4\n")
	b.WriteString(fmt.Sprintf("#EXT-X-MEDIA-SEQUENCE:%d\n", p.MediaSequence))
	for _, s := range p.Segment {
		b.WriteString(fmt.Sprintf("#EXTINF:%f,\n", s.Duration))
		b.WriteString(s.URL + "\n")
	}
	return b.Bytes()
}
