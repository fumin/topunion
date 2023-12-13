package main

import (
	"bytes"
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

func uploadVideoBody() (string, io.Reader, error) {
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

	fw, err := w.CreateFormFile("f", "20060102_150405")
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

func uploadVideo() error {
	urlStr := "http://localhost:8080/UploadVideo"
	contentType, reqBody, err := uploadVideoBody()
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
	if err := uploadVideo(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
