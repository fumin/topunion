package main

import (
	"bytes"
	"encoding/xml"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/pkg/errors"
	"golang.org/x/image/draw"
)

type BoundingBox struct {
	Xmin int `xml:"xmin"`
	Ymin int `xml:"ymin"`
	Xmax int `xml:"xmax"`
	Ymax int `xml:"ymax"`
}

type Annotation struct {
	Objects []struct {
		Name   string      `xml:"name"`
		Bndbox BoundingBox `xml:"bndbox"`
	} `xml:"object"`
}

func readAnnotation(srcDir, defect, imgName string) (Annotation, error) {
	apath := filepath.Join(srcDir, "Annotations", defect, imgName+".xml")
	b, err := os.ReadFile(apath)
	if err != nil {
		return Annotation{}, errors.Wrap(err, "")
	}
	var annt Annotation
	if err := xml.Unmarshal(b, &annt); err != nil {
		return Annotation{}, errors.Wrap(err, "")
	}
	return annt, nil
}

type Image interface {
	SubImage(image.Rectangle) image.Image
}

func extract(img Image, box BoundingBox) image.Image {
	// Enlarge image to accomodate random rotation.
	preprocessingRatio := math.Sqrt(2)
	// Enlarge image to accomodate random crop.
	preprocessingRatio /= 0.8
	// Enlarge image to accomodate random aspect ratio.
	preprocessingRatio *= (4 / 3)
	// Calculate the difference to enlarge.
	ratio := (preprocessingRatio - 1) / 2
	wd, hd := int(float64(box.Xmax-box.Xmin)*ratio), int(float64(box.Ymax-box.Ymin)*ratio)
	// Get the sub image.
	r := image.Rect(box.Xmin-wd, box.Ymin-hd, box.Xmax+wd, box.Ymax+hd)
	sub := img.SubImage(r)

	w := int(224 * preprocessingRatio)
	scaled := image.NewNRGBA(image.Rect(0, 0, w, w))
	draw.ApproxBiLinear.Scale(scaled, scaled.Bounds(), sub, sub.Bounds(), draw.Over, nil)

	return scaled
}

func saveImg(fpath string, img image.Image) error {
	b := bytes.NewBuffer(nil)
	if err := jpeg.Encode(b, img, nil); err != nil {
		return errors.Wrap(err, "")
	}

	dir := filepath.Dir(fpath)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	if err := os.WriteFile(fpath, b.Bytes(), os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func process(dstDir, srcDir, defect, imgName string) error {
	ext := filepath.Ext(imgName)
	noext := imgName[:len(imgName)-len(ext)]
	annotation, err := readAnnotation(srcDir, defect, noext)
	if err != nil {
		return errors.Wrap(err, "")
	}

	src := filepath.Join(srcDir, "images", defect, imgName)
	b, err := os.ReadFile(src)
	if err != nil {
		return errors.Wrap(err, "")
	}
	decoded, _, err := image.Decode(bytes.NewBuffer(b))
	if err != nil {
		return errors.Wrap(err, "")
	}
	img, ok := decoded.(Image)
	if !ok {
		return errors.Errorf("not an image")
	}

	for i, a := range annotation.Objects {
		extracted := extract(img, a.Bndbox)
		dst := filepath.Join(dstDir, defect, fmt.Sprintf("%s_%d.jpg", noext, i))
		if err := saveImg(dst, extracted); err != nil {
			return errors.Wrap(err, "")
		}
	}
	return nil
}

func getDst(dsts []string) string {
	prob := rand.Intn(100)
	switch {
	case prob < 10:
		return dsts[1]
	default:
		return dsts[0]
	}
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)
	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	dstRoot := "data"
	src := "PCB-DATASET"

	dsts := make([]string, 0, 2)
	for _, mode := range []string{"train", "test"} {
		dsts = append(dsts, filepath.Join(dstRoot, mode))
	}

	imgDir := filepath.Join(src, "images")
	defectTypes, err := os.ReadDir(imgDir)
	if err != nil {
		return errors.Wrap(err, "")
	}

	kill := make(chan struct{})
	defer close(kill)
	type Input struct {
		err    error
		defect string
		img    string
	}
	inputC := make(chan Input)
	go func() {
		defer close(inputC)

		for _, defectTypeEntry := range defectTypes {
			defectType := defectTypeEntry.Name()
			imgs, err := os.ReadDir(filepath.Join(imgDir, defectType))
			if err != nil {
				select {
				case <-kill:
				case inputC <- Input{err: err, defect: defectType}:
				}
				return
			}

			for _, img := range imgs {
				select {
				case <-kill:
					return
				case inputC <- Input{defect: defectType, img: img.Name()}:
				}
			}
		}
	}()

	type Output struct {
		input Input
		err   error
	}
	outC := make(chan Output)
	var wg sync.WaitGroup
	concurrency := runtime.NumCPU()
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for in := range inputC {
				if in.err != nil {
					select {
					case <-kill:
					case outC <- Output{input: in, err: in.err}:
					}
					return
				}

				dst := getDst(dsts)
				err := process(dst, src, in.defect, in.img)
				select {
				case <-kill:
					return
				case outC <- Output{input: in, err: err}:
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(outC)
	}()

	for out := range outC {
		if out.err != nil {
			return errors.Wrap(out.err, fmt.Sprintf("%#v", out.input))
		}
	}
	return nil
}
