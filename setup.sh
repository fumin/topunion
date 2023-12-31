WIFI_LAN_IFACE=$1
WIFI_LAN_USERNAME=$2
WIFI_LAN_PASSWORD=$3
if [ -z ${WIFI_LAN_IFACE} ]; then
	echo "empty wifi local network interface"
	exit 1
fi

MYUSER=topunion
HOME=/home/$MYUSER
cd $HOME

# Stop ubuntu updates from messing with nvidia and cuda!
apt remove -y unattended-upgrades
mv /usr/bin/update-manager /usr/bin/update-manager-xxx
mv /usr/bin/update-notifier /usr/bin/update-notifier-xxx
# mv /usr/local/firefox/updater /usr/local/firefox/updater-xxx

FFMPEG_DIR=/usr/local/ffmpeg-6.0.1-amd64-static

# Add `source .mybashrc` to .bashrc
cat > .mybashrc <<- EOM
export PATH=\$PATH:/usr/local/go/bin:\$HOME/go/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$FFMPEG_DIR:\$PATH

alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'

open() {
        nautilus \$1 &
}

nvidia-gpu-utilization() {
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,temperature.gpu,utilization.gpu,utilization.memory --format=csv -l 1
}
EOM

apt update
apt install -y vim git
# git config --global user.email "you@example.com"
# git config --global user.name "Your Name"
git config --global core.editor "vim"

# Make wifi fully functioning to increase its speed.
# This is important, since install cuda downloads several GB of data.
apt install -y iw
iw reg set US

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring*
apt update
apt install -y language-pack-gnome-zh-hans fonts-noto-cjk-extra ibus-libpinyin
# apt install -y nvidia-driver-535
# reboot
apt install -y --allow-downgrades cuda=11.8.0-1
cat > saxpy.cu <<- EOM
#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(void) {
	int N = 1<<20;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	maxError = max(maxError, abs(y[i]-4.0f));
	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}
EOM
/usr/local/cuda/bin/nvcc -o saxpy saxpy.cu
./saxpy

ETHERNET_IFACE=$(nmcli --terse --fields DEVICE,TYPE dev | grep ethernet | cut -d ":" -f 1)
ETHERNET_PREFIX=192.168.0
ETHERNET_IP=${ETHERNET_PREFIX}.2
ETHERNET_MACADDR=$(nmcli dev show ${ETHERNET_IFACE} | grep HWADDR | cut -d ":" -f 2- | tr -d " ")
cat > /etc/netplan/01-network-manager-all.yaml <<- EOM
# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    ${ETHERNET_IFACE}:
      addresses: [${ETHERNET_IP}/24]
      # This is a local network, so don't set it as a default gateway.
      dhcp4-overrides:
        use-routes: false
  wifis:
    ${WIFI_LAN_IFACE}:
      regulatory-domain: US
      access-points:
        "${WIFI_LAN_USERNAME}":
          password: "${WIFI_LAN_PASSWORD}"
      dhcp4: true
      # This is a local network, so don't set it as a default gateway.
      # Check there is indeed no default gateway by running \`ip route\`.
      dhcp4-overrides:
        use-routes: false
EOM
netplan --debug apply

# Force scan for our wifi network.
# This is because netplan has bugs that prevent automatically scanning 5G networks.
SCAN_SCRIPT=/usr/local/bin/scan_${WIFI_LAN_IFACE}.sh
cat > $SCAN_SCRIPT <<- EOM
#!/bin/bash
for (( i = 0; i < 5*60; i++ )); do
        SCANOUT=\$(sudo iw dev ${WIFI_LAN_IFACE} scan -u)
        echo "\$SCANOUT"
        LINE=\$(echo "\$SCANOUT" | grep ${WIFI_LAN_USERNAME})
        if [ ! -z "\$LINE" ]; then
                break
        fi
        sleep 1
done
EOM
chmod +x $SCAN_SCRIPT
cat > /etc/cron.d/scan_${WIFI_LAN_IFACE} <<- EOM
@reboot root $SCAN_SCRIPT > /tmp/scan_${WIFI_LAN_IFACE}.txt 2>&1
EOM

apt install -y kea
cat > /etc/kea/kea-dhcp4.conf <<- EOM
{
"Dhcp4": {
	"interfaces-config": {"interfaces": ["${ETHERNET_IFACE}"]},
	"subnet4": [
	{
		"id": 0,
		"subnet": "${ETHERNET_PREFIX}.0/24",
		"pools": [{"pool": "${ETHERNET_PREFIX}.16 - ${ETHERNET_PREFIX}.254"}],
		"option-data": [{"name": "routers", "data": "${ETHERNET_IP}"}],
		"reservations": [
			{"hw-address": "${ETHERNET_MACADDR}", "ip-address": "${ETHERNET_IP}"},
			{"hw-address": "4C:77:66:81:25:F1", "ip-address": "${ETHERNET_PREFIX}.3"}
		]
	}
	],
	"loggers": [{"name": "kea-dhcp4", "severity": "DEBUG"}]
}
}
EOM
systemctl restart kea-dhcp4-server

# Confine multicast to loopback.
IP_SCRIPT=/usr/local/bin/ip_init.sh
cat > $IP_SCRIPT <<-EOM
ip link set lo multicast on
ip route add 239.0.0.0/24 scope host dev lo
EOM
chmod +x $IP_SCRIPT
cat > /etc/cron.d/ip_init <<-EOM
@reboot root $IP_SCRIPT
EOM
# Mac OSX
# route -n add -net 239.0.0.0/24 -iface lo0
# Windows
# route add 239.0.0.0 MASK 255.255.255.0 127.0.0.1

apt install -y cmake python-is-python3 python3-pip sqlite3 xclip ffmpeg v4l-utils net-tools arp-scan curl tree gnuradio htop
snap install wps-office

if [ ! -d $FFMPEG_DIR ]; then
        cd /tmp
        curl -O -C - --retry 999 https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
        tar xf ffmpeg-release-amd64-static.tar.xz
        mv ffmpeg-6.0.1-amd64-static $FFMPEG_DIR
	cd $HOME
fi

# Debian stupidlly removed vls and ffmpeg's ability to play RTSP streams.
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=982299
snap install vlc

# Make it so that everyone can run arp-scan.
chmod u+s /usr/sbin/arp-scan

GOBINFULL=/usr/local/go/bin/go
GOVERSION=$($GOBINFULL version)
if [ "$GOVERSION" != "go version go1.21.3 linux/amd64" ]; then
	GOTAR=go1.21.3.linux-amd64.tar.gz
	wget https://go.dev/dl/$GOTAR
	rm -rf /usr/local/go && tar -C /usr/local -xzf $GOTAR
	rm $GOTAR
fi
sudo -u $MYUSER $GOBINFULL install golang.org/x/tools/cmd/goimports@latest

SITE_PACKAGES=$(python -m site --user-site)
if [ ! -d $SITE_PACKAGES/torch ]; then
	mkdir $HOME/whl
	cd $HOME/whl
	sudo -u $MYUSER pip3 download torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	sudo -u $MYUSER pip3 install ./*
	cd $HOME
fi

# detectron2 must be installed right after torch.
# This is because detectron2 compilation can easily fail after installing other packages.
# https://github.com/facebookresearch/detectron2/issues/5152
if [ ! -d $SITE_PACKAGES/detectron2 ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/facebookresearch/detectron2.git
	sudo -u $MYUSER python -m pip install -e detectron2
	cd $HOME
fi

sudo -u $MYUSER pip install matplotlib ipympl opencv-python jupyterlab av

if [ ! -d /usr/local/klipper ]; then
	cd /tmp
	git clone https://github.com/Klipper3d/klipper.git
	mv klipper /usr/local
	cd $HOME
fi

if [ ! -d $SITE_PACKAGES/segment-anything ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/facebookresearch/segment-anything.git
	cd segment-anything
	sudo -u $MYUSER pip install -e .
	cd $HOME
fi

if [ ! -d $SITE_PACKAGES/ByteTrack ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/ifzhang/ByteTrack.git
	cd ByteTrack
	sudo -u $MYUSER pip3 install -r requirements.txt
	sudo -u $MYUSER python -m pip install -e ByteTrack


	sudo -u $MYUSER pip install cython_bbox
	# cython_bbox breaks cocoapi, so reinstall it.
	sudo -u $MYUSER pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

	# ByteTrack breaks these libraries, so reinstall them.
	sudo -u $MYUSER pip uninstall -y lap
	sudo -u $MYUSER pip install lap
	sudo -u $MYUSER pip uninstall -y psutil
	sudo -u $MYUSER pip install psutil
fi

if [ ! -d $SITE_PACKAGES/ultralytics ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/ultralytics/ultralytics.git
	sudo -u $MYUSER python -m pip install -e ultralytics
	cd $HOME
fi
