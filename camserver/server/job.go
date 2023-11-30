package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
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
	jobB, err := json.Marshal(jb)
	if err != nil {
		return errors.Wrap(err, "")
	}
	lease := 0
	sqlStr := `INSERT INTO ` + camserver.TableJob + ` (id, job, lease) VALUES (?, ?, ?)`
	arg := []interface{}{id, jobB, lease}
	if _, err := db.ExecContext(ctx, sqlStr, arg...); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
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
	b        []byte
	job      Job
	fn       func(context.Context) error
	duration time.Duration
}

func (s *Server) dispatchJob(id string, jobB []byte) (jobRun, error) {
	jb := struct {
		Func JobFunc
		Arg  json.RawMessage
	}{}
	if err := json.Unmarshal(jobB, &jb); err != nil {
		return jobRun{}, errors.Wrap(err, "")
	}

	jbRun := jobRun{id: id, b: jobB, job: Job{Func: jb.Func}}
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
		jbRun.fn = func(ctx context.Context) error { return camserver.ProcessVideo(ctx, s.DB, arg) }
	default:
		return jobRun{}, unknownJobError{fn: jb.Func, id: id, b: jobB}
	}
	if err != nil {
		return jobRun{}, errors.Wrap(err, "")
	}
	return jbRun, nil
}

func (s *Server) doJob() error {
	jr, queueEmpty, err := s.receiveJob()
	if err != nil {
		if ujErr, ok := errors.Cause(err).(unknownJobError); ok {
			deleteJob(s.DB, ujErr.id)
		}
		return errors.Wrap(err, "")
	}
	if queueEmpty {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), jr.duration)
	defer cancel()
	if err := jr.fn(ctx); err != nil {
		resetLease(s.DB, jr.id)
		return errors.Wrap(err, fmt.Sprintf("%#v", jr))
	}

	if err := deleteJob(s.DB, jr.id); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) receiveJob() (jobRun, bool, error) {
	tx, err := s.DB.Begin()
	if err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}
	defer tx.Rollback()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	selectSQL := `SELECT id, job FROM ` + camserver.TableJob + ` WHERE lease < ? LIMIT 1`
	now := time.Now().Unix()
	var id string
	var jobB []byte
	if err := tx.QueryRowContext(ctx, selectSQL, now).Scan(&id, &jobB); err != nil {
		if err == sql.ErrNoRows {
			return jobRun{}, true, nil
		}
		return jobRun{}, false, errors.Wrap(err, "")
	}

	jr, err := s.dispatchJob(id, jobB)
	if err != nil {
		return jobRun{}, false, errors.Wrap(err, fmt.Sprintf("\"%s\" \"%s\"", id, jobB))
	}

	// Add some margin to ensure no two jobs are running at the same time.
	const margin = time.Minute
	lease := time.Now().Add(jr.duration + margin).Unix()
	updateSQL := `UPDATE ` + camserver.TableJob + ` SET lease=? WHERE id=?`
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
