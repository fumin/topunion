package camserver

import (
	"camserver/cuda"
	"path/filepath"
	"testing"
)

func TestCounter(t *testing.T) {
	t.Parallel()
	env := newEnvironment(t)
	defer env.close()

	device := "cpu"
	if cuda.IsAvailable() {
		device = "cuda:0"
	}
	cfg := CountConfig{Height: 480, Width: 640, Device: device}
	cfg.Mask.Enable = true
	cfg.Mask.Crop.W = 999999
	cfg.Mask.Mask.Slope = 5
	cfg.Mask.Mask.Y = 160
	cfg.Mask.Mask.H = 70
	cfg.Yolo.Weights = "yolo_best.pt"
	cfg.Yolo.Size = 640
	counter, err := NewCounter(env.dir, env.scripts.Count, cfg)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer counter.Close()

	dst := filepath.Join(env.dir, "dst.ts")
	src := filepath.Join("testing", "shilin20230826_sd.mp4")
	out, err := counter.Analyze(dst, src)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if out.Passed != 10 {
		t.Fatalf("%#v", out)
	}
}

// func TestCount(t *testing.T) {
// 	t.Parallel()
// 	dir, err := os.MkdirTemp("", "")
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	defer os.RemoveAll(dir)
// 	scripts, err := NewScripts(dir)
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
//
// 	body := bytes.NewBuffer(nil)
// 	mw := multipart.NewWriter(body)
// 	mw.WriteField("myname", "john")
// 	part, err := mw.CreateFormFile("f", "test.mp4")
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	if _, err := io.Copy(part, bytes.NewBufferString("music")); err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	if err := mw.Close(); err != nil {
// 		t.Fatalf("%+v", err)
// 	}
//
// 	urlStr := "http://localhost:8080/Analyze"
// 	req, err := http.NewRequest("POST", urlStr, body)
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	req.Header.Add("Content-Type", mw.FormDataContentType())
// 	resp, err := http.DefaultClient.Do(req)
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	defer resp.Body.Close()
// 	respB, err := io.ReadAll(resp.Body)
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	t.Logf("%s", respB)
// }
