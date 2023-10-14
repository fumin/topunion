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

# Download firefox to /usr/local
mv /usr/bin/update-manager /usr/bin/update-manager-xxx
mv /usr/bin/update-notifier /usr/bin/update-notifier-xxx
# mv /usr/local/firefox/updater /usr/local/firefox/updater-xxx

# Add `source .mybashrc` to .bashrc
cat > .mybashrc <<- EOM
export PATH=\$PATH:/usr/local/go/bin:\$HOME/go/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64

alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'

open() {
        nautilus \$1 &
}
EOM

apt update
apt install -y vim git
# git config --global user.email "you@example.com"
# git config --global user.name "Your Name"
git config --global core.editor "vim"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring*
apt update
apt install -y language-pack-zh-hans fonts-noto-cjk-extra ibus-libpinyin
apt install -y nvidia-driver-535=535.104.12-0ubuntu1
# reboot
apt install -y cuda=11.8.0-1
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
ETHERNET_PREFIX=192.168.2
ETHERNET_IP=${ETHERNET_PREFIX}.1
cat > /etc/netplan/01-network-manager-all.yaml <<- EOM
# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    ${ETHERNET_IFACE}:
      addresses: [${ETHERNET_IP}/24]
  wifis:
    ${WIFI_LAN_IFACE}:
      regulatory-domain: TW
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

apt install -y isc-dhcp-server
cat > /etc/dhcp/dhcpd.conf <<- EOM
default-lease-time 600;
max-lease-time 7200;

# IP conventions:
# xxx.xxx.xxx.0:   network
# xxx.xxx.xxx.1:   router
# xxx.xxx.xxx.255: broadcast
subnet ${ETHERNET_PREFIX}.0 netmask 255.255.255.0 {
 range ${ETHERNET_PREFIX}.2 ${ETHERNET_PREFIX}.254;
 option routers ${ETHERNET_IP};
 # option domain-name-servers ${ETHERNET_IP};
 # option domain-name "topunion.com";
}
EOM
systemctl restart isc-dhcp-server.service

apt install -y python-is-python3 python3-pip xclip ffmpeg v4l-utils vlc net-tools arp-scan iw curl
snap install wps-office

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

sudo -u $MYUSER pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

sudo -u $MYUSER pip install cython
sudo -u $MYUSER pip install cython_bbox matplotlib ipympl opencv-python jupyterlab av

if [ ! -d /usr/local/klipper ]; then
	cd /tmp
	git clone https://github.com/Klipper3d/klipper.git
	mv klipper /usr/local
	cd $HOME
fi

SITE_PACKAGES=$(python -m site --user-site)
if [ ! -d $SITE_PACKAGES/detectron2 ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/facebookresearch/detectron2.git
	sudo -u $MYUSER python -m pip install -e detectron2
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
	git clone https://github.com/ifzhang/ByteTrack.git
	cd ByteTrack
	pip3 install -r requirements.txt
	python3 setup.py develop
fi

sudo -u $MYUSER pip install loguru lap
if [ ! -d $SITE_PACKAGES/ultralytics ]; then
	cd $SITE_PACKAGES
	sudo -u $MYUSER git clone https://github.com/ultralytics/ultralytics.git
	sudo -u $MYUSER python -m pip install -e ultralytics
	cd $HOME
fi
