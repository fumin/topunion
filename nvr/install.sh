USER=topunion
DIR=/var/nvr
EXE=$HOME/go/bin/nvr
CONFIG=$DIR/config.json

# Create folder.
sudo mkdir $DIR
sudo chown $USER $DIR
# Download 上聯-研發製造/1.蛋品機械/9.影像辨識/數蛋機/yolo_best.pt
curl --location "https://www.dropbox.com/scl/fi/jsa1wxnint85wcvta5y99/yolo_best.pt?rlkey=ddhlhd9x9yvlgw7t2uu3eq2n6&dl=0" -o $DIR/yolo_best.pt
# cp yolo_best.pt $DIR/yolo_best.pt
cp server/config/production.json $CONFIG
go run cmd/fixture/main.go -d=$DIR

# Create executable.
go build -race cmd/server/main.go
mv main $EXE

# Create cron job.
LOGDIR=$DIR/log
mkdir $LOGDIR
cat > /tmp/cronnvr <<- EOM
@reboot $USER $EXE -c=$CONFIG >> $LOGDIR/log.txt 2>&1
EOM
sudo mv /tmp/cronnvr /etc/cron.d/nvr
