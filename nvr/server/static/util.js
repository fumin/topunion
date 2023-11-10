let util = {};

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
