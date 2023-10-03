FPATH=$1
cvlc -vvv $FPATH --loop --sout-keep --sout '#gather:rtp{sdp=rtsp://localhost:8554/rtsp}'
