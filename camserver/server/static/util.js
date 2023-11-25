let util = {};

util.playHLS = function(videoElement, src) {
	if (Hls.isSupported()) {
                let config = {
                        liveSyncDurationCount: 1,
                        liveMaxLatencyDurationCount: 5,
                        liveDurationInfinity: true,
                };
                let hls = new Hls(config);
                hls.on(Hls.Events.ERROR, function(event, data){
                        let errorType = data.type;
                        let errorDetails = data.details;
                        let errorFatal = data.fatal;
                        console.log(errorType, errorDetails, errorFatal, event, data);

                        if (errorFatal) {
                                switch (errorDetails) {
                                case Hls.ErrorDetails.MANIFEST_LOAD_ERROR:
                                case Hls.ErrorDetails.LEVEL_EMPTY_ERROR:
					hls.detachMedia();
					hls.destroy();

                                        let secs = 5;
                                        setTimeout(function(){
						util.playHLS(videoElement, src);
                                        }, secs*1000);
                                }
			}
                });
                hls.loadSource(src);
                hls.attachMedia(videoElement);
        } else if (videoElement.canPlayType("application/vnd.apple.mpegurl")) {
                videoElement.src = src;
        } else {
                let info = document.querySelector("#info");
                info.textContent = "Your browser does not support HLS.";
        }
};

util.playMPEGTS = function(videoElement, urlStr) {
	if (mpegts.getFeatureList().mseLivePlayback) {
		let player = mpegts.createPlayer({
			type: "mpegts",
			isLive: true,
			url: urlStr,
		});
		player.on(mpegts.Events.ERROR, function(errType, errDetail, err){
			console.log(errType, errDetail, err);
			if (errType == mpegts.ErrorTypes.MEDIA_ERROR) {
				player.pause();
				player.unload();
				player.detachMediaElement();
	
				player.attachMediaElement(videoElement);
				player.load();
				player.play();
			} else {
				let secs = 1;
				setTimeout(function(){
				        window.location.reload();
				}, secs*1000);
			}
		});
		player.attachMediaElement(videoElement);
		player.load();
		player.play();
	} else {
		alert("browser not supported");
	}
};
