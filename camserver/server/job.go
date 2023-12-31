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
	jobForTestingFunc func(context.Context) error
)

type JobFunc string

type Job struct {
	Func        JobFunc
	Arg         interface{}
	DurationSec int
}

func SendJob(ctx context.Context, db *sql.DB, jb Job) error {
	id := util.RandID()
	createAt := time.Now().Unix()
	jobB, err := json.Marshal(jb)
	if err != nil {
		return errors.Wrap(err, "")
	}
	shard := rand.Int()
	sqlStr := `INSERT INTO ` + camserver.TableJob + ` (id, createAt, job, duration, shard, lease, retries) VALUES (?, ?, ?, ?, ?, 0, 0)`
	arg := []interface{}{id, createAt, jobB, jb.DurationSec, shard}
	if _, err := db.ExecContext(ctx, sqlStr, arg...); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) DoJobForever() {
	concurrency := max(1, runtime.NumCPU()-2)
	for i := 0; i < concurrency; i++ {
		go func(shard JobShard) {
			milliSec := i*1000 + rand.Intn(2000)
			<-time.After(time.Duration(milliSec) * time.Millisecond)
			for {
				emptyQueue, _, err := s.DoJob(shard)
				if err != nil {
					log.Printf("%+v", err)
				}
				if emptyQueue {
					// Apply randomness to reduce database contention
					secs := 5 + rand.Intn(10)
					<-time.After(time.Duration(secs) * time.Second)
				}
			}
		}(JobShard{Shard: i, Divisor: concurrency})
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

	fn func(context.Context) error
}

func (s *Server) dispatchJob(jr *jobRun) error {
	jb := struct {
		Func        JobFunc
		Arg         json.RawMessage
		DurationSec int
	}{}
	if err := json.Unmarshal(jr.b, &jb); err != nil {
		return errors.Wrap(err, "")
	}
	jr.job.Func = jb.Func
	jr.job.DurationSec = jb.DurationSec

	var err error
	switch jb.Func {
	case jobForTesting:
		jr.fn = jobForTestingFunc
	case JobProcessVideo:
		var arg camserver.ProcessVideoInput
		err = json.Unmarshal(jb.Arg, &arg)
		jr.job.Arg = arg
		jr.fn = func(ctx context.Context) error {
			c, ok := s.C.GetCamera(arg.Camera)
			if !ok {
				return errors.Wrap(err, fmt.Sprintf("unknown camera \"%s\"", arg.Camera))
			}
			return camserver.ProcessVideo(ctx, s.DB, s.Scripts, c.Count, arg)
		}
	default:
		return unknownJobError{fn: jb.Func, id: jr.id, b: jr.b}
	}
	if err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) DoJob(shard JobShard) (bool, error, error) {
	jr, queueEmpty, err := s.receiveJob(shard)
	if err != nil {
		return false, nil, errors.Wrap(err, "")
	}
	if queueEmpty {
		return true, nil, nil
	}

	if err := s.dispatchJob(&jr); err != nil {
		return false, nil, errors.Wrap(err, fmt.Sprintf("\"%s\" \"%s\"", jr.id, jr.b))
	}

	// Run the job logic.
	timeout := time.Duration(jr.job.DurationSec) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	jobErr := jr.fn(ctx)

	// Update job status given the results of the job run.
	switch {
	case jobErr == nil:
		err = deleteJob(s.DB, jr.id)
	case jr.retries < 3:
		err = resetLease(s.DB, jr.id)
	default:
		err = moveDeadJob(s.DB, jr, jobErr)
	}
	if err != nil {
		msg := fmt.Sprintf("%s %v %d", jr.id, jobErr == nil, jr.retries)
		return false, nil, errors.Wrap(err, msg)
	}

	return false, jobErr, nil
}

type JobShard struct {
	Shard   int
	Divisor int
}

func (s *Server) receiveJob(shard JobShard) (jobRun, bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tx, err := s.DB.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}
	defer tx.Rollback()

	selectSQL := `SELECT id, createAt, job, retries FROM ` + camserver.TableJob + ` WHERE lease < ? AND (shard % ?)=? LIMIT 1`
	arg := []interface{}{time.Now().Unix(), shard.Divisor, shard.Shard}
	var jr jobRun
	var createAt int64
	if err := tx.QueryRowContext(ctx, selectSQL, arg...).Scan(&jr.id, &createAt, &jr.b, &jr.retries); err != nil {
		if err == sql.ErrNoRows {
			return jobRun{}, true, nil
		}
		return jobRun{}, false, errors.Wrap(err, "")
	}
	jr.createAt = time.Unix(createAt, 0)

	// Add some margin to ensure no two jobs are running at the same time.
	const margin = time.Minute
	lease := time.Now().Add(margin).Unix()
	updateSQL := `UPDATE ` + camserver.TableJob + ` SET
		lease=?+duration, retries=retries+1
		WHERE id=?`
	if _, err := tx.ExecContext(ctx, updateSQL, lease, jr.id); err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return jobRun{}, false, errors.Wrap(err, "")
	}
	return jr, false, nil
}

func resetLease(db *sql.DB, id string) error {
	updateSQL := `UPDATE ` + camserver.TableJob + ` SET lease=0 WHERE id=?`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := db.ExecContext(ctx, updateSQL, id)
	return err
}

func deleteJob(db *sql.DB, id string) error {
	deleteSQL := `DELETE FROM ` + camserver.TableJob + ` WHERE id=?`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := db.ExecContext(ctx, deleteSQL, id)
	return err
}

func moveDeadJob(db *sql.DB, jr jobRun, jobErr error) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer tx.Rollback()

	jobErrStr := fmt.Sprintf("%+v", jobErr)
	insertSQL := `INSERT INTO ` + camserver.TableDeadJob + ` (id, createAt, job, err) VALUES (?, ?, ?, ?)`
	if _, err := tx.ExecContext(ctx, insertSQL, jr.id, jr.createAt.Unix(), jr.b, jobErrStr); err != nil {
		return errors.Wrap(err, "")
	}

	deleteSQL := `DELETE FROM ` + camserver.TableJob + ` WHERE id=?`
	if _, err := tx.ExecContext(ctx, deleteSQL, jr.id); err != nil {
		log.Printf("%s %s", deleteSQL, jr.id)
		return errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
