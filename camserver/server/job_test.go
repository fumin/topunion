package server

import (
	"camserver"
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

func TestJobSuccess(t *testing.T) {
	s := newServer(t)
	defer closeServer(s)

	var jobHasRun bool
	jobForTestingFunc = func(ctx context.Context) error {
		jobHasRun = true
		return nil
	}
	jb := Job{Func: jobForTesting}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := SendJob(ctx, s.DB, jb); err != nil {
		t.Fatalf("%+v", err)
	}

	if _, err := s.doJob(); err != nil {
		t.Fatalf("%+v", err)
	}
	if !jobHasRun {
		t.Fatalf("job not run")
	}

	// Check that finished jobs are deleted.
	var jobsInQueue int
	if err := s.DB.QueryRowContext(ctx, `SELECT count(1) FROM `+camserver.TableJob).Scan(&jobsInQueue); err != nil {
		t.Fatalf("%+v", err)
	}
	if jobsInQueue != 0 {
		t.Fatalf("%d", jobsInQueue)
	}
}

func TestJobTimeout(t *testing.T) {
	s := newServer(t)
	defer closeServer(s)

	jobForTestingFunc = func(ctx context.Context) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Hour):
			return nil
		}
	}
	jobForTestingDuration = time.Millisecond
	jb := Job{Func: jobForTesting}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := SendJob(ctx, s.DB, jb); err != nil {
		t.Fatalf("%+v", err)
	}

	if _, err := s.doJob(); errors.Cause(err) != context.DeadlineExceeded {
		t.Fatalf("%+v", err)
	}

	// Check that retries count is incremented.
	var retries int
	if err := s.DB.QueryRowContext(ctx, `SELECT retries FROM `+camserver.TableJob+` WHERE lease == 0`).Scan(&retries); err != nil {
		t.Fatalf("%+v", err)
	}
	if retries != 1 {
		t.Fatalf("%d", retries)
	}
}

func TestJobError(t *testing.T) {
	s := newServer(t)
	defer closeServer(s)

	jobErr := errors.Errorf("job error")
	jobForTestingFunc = func(ctx context.Context) error {
		return jobErr
	}
	jobForTestingDuration = time.Millisecond
	jb := Job{Func: jobForTesting}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := SendJob(ctx, s.DB, jb); err != nil {
		t.Fatalf("%+v", err)
	}

	if _, err := s.doJob(); errors.Cause(err) != jobErr {
		t.Fatalf("%+v", err)
	}

	// Check that job is ready to be retried again.
	var queued int
	if err := s.DB.QueryRowContext(ctx, `SELECT count(1) FROM `+camserver.TableJob+` WHERE lease == 0`).Scan(&queued); err != nil {
		t.Fatalf("%+v", err)
	}
	if queued != 1 {
		t.Fatalf("%d", queued)
	}

	// Retry a few more times.
	for i := 0; i < 3; i++ {
		if _, err := s.doJob(); errors.Cause(err) != jobErr {
			t.Fatalf("%+v", err)
		}
	}

	// Check that the dead letter queue is empty before the last try.
	var dead int
	if err := s.DB.QueryRowContext(ctx, `SELECT count(1) FROM `+camserver.TableDeadJob).Scan(&dead); err != nil {
		t.Fatalf("%+v", err)
	}
	if dead != 0 {
		t.Fatalf("%d", dead)
	}

	// Last try.
	if _, err := s.doJob(); err != nil {
		t.Fatalf("%+v", err)
	}

	// After the last try, job should be moved to the dead letter queue.
	if err := s.DB.QueryRowContext(ctx, `SELECT count(1) FROM `+camserver.TableJob+` WHERE lease == 0`).Scan(&queued); err != nil {
		t.Fatalf("%+v", err)
	}
	if queued != 0 {
		t.Fatalf("%d", queued)
	}
	var jobB []byte
	if err := s.DB.QueryRowContext(ctx, `SELECT job FROM `+camserver.TableDeadJob+` LIMIT 1`).Scan(&jobB); err != nil {
		t.Fatalf("%+v", err)
	}
	var deadJob Job
	if err := json.Unmarshal(jobB, &deadJob); err != nil {
		t.Fatalf("%+v", err)
	}
	if deadJob.Func != jobForTesting {
		t.Fatalf("%s", jobB)
	}
}

func newServer(t *testing.T) *Server {
	dir, err := os.MkdirTemp("", t.Name())
	if err != nil {
		t.Fatalf("%+v", err)
	}

	cfg := Config{
		Dir:  dir,
		Addr: "localhost:0",
	}
	s, err := NewServer(cfg)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := camserver.CreateTables(ctx, s.DB); err != nil {
		t.Fatalf("%+v", err)
	}

	return s
}

func closeServer(s *Server) {
	s.Close()
	os.RemoveAll(s.C.Dir)
}
