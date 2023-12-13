package main

import (
	"bytes"
	"camserver/util"
	"context"
	"flag"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

func uploadVideoBody(t time.Time) (string, io.Reader, error) {
	vidPath := filepath.Join("testing", "shilin20230826_sd.mp4")
	vid, err := os.Open(vidPath)
	if err != nil {
		return "", nil, errors.Wrap(err, "")
	}
	defer vid.Close()

	body := bytes.NewBuffer(nil)
	w := multipart.NewWriter(body)

	if err := w.WriteField("c", "camera0"); err != nil {
		return "", nil, errors.Wrap(err, "")
	}

	fname := t.In(time.UTC).Format(util.FormatDatetime) + ".mp4"
	fw, err := w.CreateFormFile("f", fname)
	if err != nil {
		return "", nil, errors.Wrap(err, "")
	}
	if _, err := io.Copy(fw, vid); err != nil {
		return "", nil, errors.Wrap(err, "")
	}

	if err := w.Close(); err != nil {
		return "", nil, errors.Wrap(err, "")
	}
	return w.FormDataContentType(), body, nil
}

func uploadVideo(t time.Time) error {
	urlStr := "http://localhost:8080/UploadVideo"
	contentType, reqBody, err := uploadVideoBody(t)
	if err != nil {
		return errors.Wrap(err, "")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", urlStr, reqBody)
	if err != nil {
		return errors.Wrap(err, "")
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return errors.Wrap(err, "")
	}

	if resp.StatusCode != http.StatusOK {
		return errors.Errorf("\"%s\"", body)
	}
	return nil
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	for {
		startT := time.Now()
		if err := uploadVideo(startT); err != nil {
			return errors.Wrap(err, "")
		}
		<-time.After(startT.Add(10 * time.Second).Sub(time.Now()))
	}
	return nil
}
