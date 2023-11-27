package camserver

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"testing"
)

func TestCount(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	scripts, err := NewScripts(dir)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	body := bytes.NewBuffer(nil)
	mw := multipart.NewWriter(body)
	mw.WriteField("myname", "john")
	part, err := mw.CreateFormFile("f", "test.mp4")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if _, err := io.Copy(part, bytes.NewBufferString("music")); err != nil {
		t.Fatalf("%+v", err)
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("%+v", err)
	}

	urlStr := "http://localhost:8080/Analyze"
	req, err := http.NewRequest("POST", urlStr, body)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	req.Header.Add("Content-Type", mw.FormDataContentType())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer resp.Body.Close()
	respB, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	t.Logf("%s", respB)
}
