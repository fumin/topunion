package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"

	"github.com/pkg/errors"

	"camserver"
	"camserver/util"
)

const (
	jobForTesting   JobFunc = "forTesting"
	JobProcessVideo JobFunc = "ProcessVideo"
)

var (
	// Only for testing purpose.
	jobForTestingFunc     func(context.Context) error
	jobForTestingDuration time.Duration
)

type JobFunc string

type Job struct {
	Func JobFunc
	Arg  interface{}
}

func SendJob(ctx context.Context, db *sql.DB, jb Job) error {
	id := util.RandID()
	createAt := time.Now().Unix()
	jobB, err := json.Marshal(jb)
	if err != nil {
		return errors.Wrap(err, "")
	}
	sqlStr := `INSERT INTO ` + camserver.TableJob + ` (id, createAt, job, lease, retries) VALUES (?, ?, ?, 0, 0)`
	arg := []interface{}{id, createAt, jobB}
	if _, err := db.ExecContext(ctx, sqlStr, arg...); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) DoJobForever() {
	concurrency := max(1, runtime.NumCPU()-2)
	for i := 0; i < concurrency; i++ {
		go func() {
			for {
				emptyQueue, err := s.doJob()
				if err != nil {
					log.Printf("%+v", err)
				}
				if emptyQueue {
					// Apply randomness to reduce database contention
					secs := 5 + rand.Intn(10)
					<-time.After(time.Duration(secs) * time.Second)
				}
			}
		}()
	}
}

type unknownJobError struct {
	id string
	fn JobFunc
	b  []byte
}

func (e unknownJobError) Error() string {
	return fmt.Sprintf("unknown job \"%s\" \"%s\" \"%s\"", e.fn, e.id, e.b)
}

type jobRun struct {
	id       string
	createAt time.Time
	b        []byte
	retries  int
	job      Job

	fn       func(context.Context) error
	duration time.Duration
}

func (s *Server) dispatchJob(id string, createAt int64, jobB []byte) (jobRun, error) {
	jb := struct {
		Func JobFunc
		Arg  json.RawMessage
	}{}
	if err := json.Unmarshal(jobB, &jb); err != nil {
		return jobRun{}, errors.Wrap(err, "")
	}

	jbRun := jobRun{id: id, createAt: time.Unix(createAt, 0), b: jobB, job: Job{Func: jb.Func}}
	var err error
	switch jb.Func {
	case jobForTesting:
		jbRun.duration = jobForTestingDuration
		jbRun.fn = jobForTestingFunc
	case JobProcessVideo:
		var arg camserver.ProcessVideoInput
		err = json.Unmarshal(jb.Arg, &arg)
		jbRun.job.Arg = arg
		jbRun.duration = 5 * time.Minute
		jbRun.fn = func(ctx context.Context) error {
			cam, ok := s.Camera[arg.Camera]
			if !ok {
				return errors.Wrap(err, fmt.Sprintf("unknown camera \"%s\"", arg.Camera))
			}
			return camserver.ProcessVideo(ctx, s.DB, cam.Counter, arg)
		}
	default:
		return jobRun{}, unknownJobError{fn: jb.Func, id: id, b: jobB}
	}
	if err != nil {
		return jobRun{}, errors.Wrap(err, "")
	}
	return jbRun, nil
}

func (s *Server) doJob() (bool, error) {
	jr, queueEmpty, err := s.receiveJob()
	if err != nil {
		return false, errors.Wrap(err, "")
	}
	if queueEmpty {
		return true, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), jr.duration)
	defer cancel()
	if err := jr.fn(ctx); err != nil {
		resetLease(s.DB, jr.id)
		return false, errors.Wrap(err, fmt.Sprintf("%#v", jr))
	}

	if err := deleteJob(s.DB, jr.id); err != nil {
		return false, errors.Wrap(err, "")
	}
	return false, nil
}

func (s *Server) receiveJob() (jobRun, bool, error) {
	for {
		jr, emptyQueue, err := s.receiveJobOnce()
		if err != nil {
			return jobRun{}, false, errors.Wrap(err, "")
		}
		if emptyQueue {
			return jobRun{}, true, nil
		}
		if jr.retries > 3 {
			if err := moveDeadJob(s.DB, jr); err != nil {
				return jobRun{}, false, errors.Wrap(err, "")
			}
			continue
		}
		return jr, false, nil
	}
}

func (s *Server) receiveJobOnce() (jobRun, bool, error) {
	tx, err := s.DB.Begin()
	if err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}
	defer tx.Rollback()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	selectSQL := `SELECT id, createAt, job, retries FROM ` + camserver.TableJob + ` WHERE lease < ? LIMIT 1`
	now := time.Now().Unix()
	var id string
	var createAt int64
	var jobB []byte
	var retries int
	if err := tx.QueryRowContext(ctx, selectSQL, now).Scan(&id, &createAt, &jobB, &retries); err != nil {
		if err == sql.ErrNoRows {
			return jobRun{}, true, nil
		}
		return jobRun{}, false, errors.Wrap(err, "")
	}

	jr, err := s.dispatchJob(id, createAt, jobB)
	if err != nil {
		return jobRun{}, false, errors.Wrap(err, fmt.Sprintf("\"%s\" \"%s\"", id, jobB))
	}
	jr.retries = retries

	// Add some margin to ensure no two jobs are running at the same time.
	const margin = time.Minute
	lease := time.Now().Add(jr.duration + margin).Unix()
	updateSQL := `UPDATE ` + camserver.TableJob + ` SET
		lease=?, retries=retries+1
		WHERE id=?`
	if _, err := tx.ExecContext(ctx, updateSQL, lease, id); err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}
	return jr, false, nil
}

func resetLease(db *sql.DB, id string) {
	updateSQL := `UPDATE ` + camserver.TableJob + ` SET lease=0 WHERE id=?`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	db.ExecContext(ctx, updateSQL, id)
}

func deleteJob(db *sql.DB, id string) error {
	deleteSQL := `DELETE FROM ` + camserver.TableJob + ` WHERE id=?`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := db.ExecContext(ctx, deleteSQL, id)
	return err
}

func moveDeadJob(db *sql.DB, jr jobRun) error {
	tx, err := db.Begin()
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer tx.Rollback()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	deleteSQL := `DELETE FROM ` + camserver.TableJob + ` WHERE id=?`
	if _, err := db.ExecContext(ctx, deleteSQL, jr.id); err != nil {
		return errors.Wrap(err, "")
	}

	insertSQL := `INSERT INTO ` + camserver.TableDeadJob + ` (id, createAt, job) VALUES (?, ?, ?)`
	if _, err := db.ExecContext(ctx, insertSQL, jr.id, jr.createAt.Unix(), jr.b); err != nil {
		return errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
