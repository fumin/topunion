let util = {};

util.TimeFormat = function(d) {
  var dtStr = `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`
  return dtStr;
}

util.Base64URLSafe = function(s) {
  let res = btoa(unescape(encodeURIComponent(s)));
  res = res.replaceAll("+", "-");
  res = res.replaceAll("/", "_");
  res = res.replaceAll("=", "");
  return res;
}

let infoElement = document.querySelector("#info");
util.ReqJSON = function(method, urlStr, loadEnd, okCallback) {
        let req = new XMLHttpRequest();
        req.addEventListener("loadend", function(ev){
                loadEnd(ev);
        });
        req.addEventListener("error", function(ev){
                infoElement.textContent = "http error";
        });
        req.addEventListener("load", function(ev){
                let resp = JSON.parse(ev.target.responseText);
                if (resp.Error) {
                        infoElement.textContent = resp.Error.Msg;
                        return;
                }
                okCallback(resp);
        });
        req.open(method, urlStr);
        req.send();
}

util.ReqSSE = function(urlStr, okCallback, errCallback) {
  var evtSrc = new EventSource(urlStr);
  evtSrc.addEventListener("error", function(ev){
    evtSrc.close();
    let logDiv = document.createElement("div");
    logDiv.textContent = `event source error: `+ev.target.url;
    infoElement.appendChild(logDiv);
  });
  evtSrc.addEventListener("message", function(ev){
    let data = JSON.parse(ev.data);
    if (data.Resp) {
      if (data.Resp.Error) {
        errCallback(data.Resp.Error);
      } else {
        okCallback(data.Resp.Resp);
      }
      evtSrc.close();
      return;
    }

    let logDiv = document.createElement("div");
    logDiv.textContent = data.Progress;
    infoElement.insertBefore(logDiv, infoElement.firstChild);
  })
}

util.PlayHLS = function(videoElement, src) {
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

util.PlayMPEGTS = function(videoElement, urlStr) {
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
