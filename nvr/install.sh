DIR=/var/nvr
EXE=$HOME/go/bin/nvr
CONFIG=$DIR/config.json

mkdir $DIR
# Download 上聯-研發製造/1.蛋品機械/9.影像辨識/數蛋機/yolo_best.pt
cp server/config/production.json $CONFIG
go build -race cmd/server/main.go
mv main $EXE

$LOGDIR=$DIR/log
mkdir $LOGDIR
cat > /etc/cron.d/nvr <<- EOM
@reboot topunion $EXE -c=$CONFIG >> $LOGDIR/log.txt 2>&1
EOM
