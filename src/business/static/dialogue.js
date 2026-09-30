(() => {
  'use strict';
  const el = (id) => document.getElementById(`dialogue-${id}`);
  let session = null, generation = 0, state = 'idle', controller = null;
  let recorder = null, mic = null, recordTimer = null, idleTimer = null;
  let context = null, nextTime = 0, sources = new Set();
  let playbackResolve = null, responseEnded = false, sessionEnded = false;

  function status(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }
  function setState(value) {
    state = value;
    el('record').disabled = value !== 'idle';
    el('record').textContent = session ? '继续录音' : '开始对话并录音';
    el('stop').disabled = value !== 'recording';
    el('end').disabled = !session && value === 'idle';
    el('voice').disabled = value !== 'idle';
    document.getElementById('voice-debug').inert = value !== 'idle' || !!session;
  }
  async function responseError(response) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.error?.message || `请求失败（${response.status}）`);
    error.status = response.status;
    return error;
  }
  function row(label, text, customer = false) {
    const node = document.createElement('p');
    node.textContent = `${label}：${text}`;
    if (customer) node.className = 'customer';
    el('log').appendChild(node);
    el('log').scrollTop = el('log').scrollHeight;
  }
  function releaseMic() {
    clearTimeout(recordTimer);
    recordTimer = null;
    if (recorder) {
      recorder.onstop = null;
      recorder.ondataavailable = null;
      if (recorder.state !== 'inactive') recorder.stop();
    }
    recorder = null;
    mic?.getTracks().forEach((track) => track.stop());
    mic = null;
  }
  function stopAudio() {
    for (const source of sources) {
      source.onended = null;
      try { source.stop(); } catch {}
    }
    sources.clear();
    nextTime = 0;
    playbackResolve?.();
    playbackResolve = null;
    el('resume').hidden = true;
  }
  function deleteSession(id, keepalive = false) {
    if (id) fetch(`/api/dialogue/sessions/${id}`, {method: 'DELETE', keepalive}).catch(() => {});
  }
  function end(message = '会话已结束。点击按钮可开始新的对话。', keepalive = false) {
    generation += 1;
    clearTimeout(idleTimer);
    controller?.abort();
    controller = null;
    releaseMic();
    stopAudio();
    deleteSession(session?.session_id, keepalive);
    session = null;
    setState('idle');
    status(message);
    el('turn').textContent = '会话已结束';
  }
  function armIdle() {
    clearTimeout(idleTimer);
    if (session) idleTimer = setTimeout(() => {
      end('空闲时间已到，会话已结束。请重新点击开始。');
    }, session.limits.idle_seconds * 1000);
  }
  async function audioContext() {
    if (!context) context = new (window.AudioContext || window.webkitAudioContext)();
    if (context.state !== 'running') await context.resume();
    return context;
  }
  function maybePlaybackDone() {
    if (responseEnded && sources.size === 0) {
      playbackResolve?.();
      playbackResolve = null;
    }
  }
  async function queueAudio(event, run) {
    const bytes = Uint8Array.from(atob(event.audio_base64), (c) => c.charCodeAt(0));
    const buffer = await context.decodeAudioData(bytes.buffer);
    if (run !== generation) return;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    sources.add(source);
    source.onended = () => {
      sources.delete(source);
      if (run === generation) maybePlaybackDone();
    };
    const starts = Math.max(context.currentTime + 0.03, nextTime);
    nextTime = starts + buffer.duration;
    source.start(starts);
    setState('playing');
    if (context.state !== 'running') {
      el('resume').hidden = false;
      status('浏览器暂停了声音，请点击播放回答。');
    } else status('正在播放回答，请稍候…');
  }

  async function send(blob, run) {
    setState('processing');
    status('正在上传并识别…');
    responseEnded = false;
    sessionEnded = false;
    stopAudio();
    const played = new Promise((resolve) => { playbackResolve = resolve; });
    controller = new AbortController();
    const form = new FormData();
    form.append('file', blob, 'recording');
    form.append('voice', el('voice').value);
    let receivedDone = false;
    let nextIndex = 0;
    try {
      const response = await fetch(`/api/dialogue/sessions/${session.session_id}/turns`, {
        method: 'POST', body: form, signal: controller.signal
      });
      if (!response.ok) throw await responseError(response);
      if (!response.body) throw new Error('浏览器不支持语音流。');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';
      const line = async (raw) => {
        if (!raw.trim() || run !== generation) return;
        if (receivedDone) throw new Error('回答流顺序异常。');
        const event = JSON.parse(raw);
        if (event.request_id) el('metrics').textContent = `请求 ${event.request_id}`;
        switch (event.type) {
          case 'stage':
            status({asr: '正在识别…', deepseek: '导购正在回答…', tts: '正在生成语音…'}[event.stage]);
            break;
          case 'transcript': row('您', event.text, true); break;
          case 'answer':
            row('导购', event.text);
            sessionEnded = event.session_ended;
            el('turn').textContent = `第 ${event.turn} / ${session.limits.max_turns} 轮`;
            break;
          case 'audio':
            if (event.index !== nextIndex++) throw new Error('语音片段顺序异常。');
            await queueAudio(event, run);
            break;
          case 'error':
            sessionEnded = event.session_ended;
            throw new Error(event.error?.message || '本轮回答失败，请重新录音。');
          case 'done':
            if (!nextIndex || nextIndex !== event.sentence_count) throw new Error('语音片段不完整。');
            receivedDone = true;
            responseEnded = true;
            sessionEnded = event.session_ended;
            el('metrics').textContent = `请求 ${event.request_id} · ASR ${event.timings.asr_ms} ms · ` +
              `模型 ${event.timings.deepseek_ms} ms · TTS ${event.timings.tts_ms} ms · ` +
              `首段音频 ${event.timings.first_audio_ms} ms`;
            maybePlaybackDone();
            break;
          default: throw new Error('无法识别回答事件。');
        }
      };
      while (run === generation) {
        const {value, done} = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), {stream: !done});
        let newline;
        while ((newline = pending.indexOf('\n')) >= 0) {
          const raw = pending.slice(0, newline);
          pending = pending.slice(newline + 1);
          await line(raw);
        }
        if (done) break;
      }
      if (pending.trim()) await line(pending);
      if (run !== generation) return;
      if (!receivedDone) throw new Error('语音连接提前中断，请重新录音。');
      await played;
      if (run !== generation) return;
      if (sessionEnded) end('本次对话已结束。点击按钮可开始新的对话。');
      else { status('播放完成，可以继续录音。'); setState('idle'); armIdle(); }
    } catch (error) {
      if (run !== generation) return;
      controller?.abort();
      stopAudio();
      if (sessionEnded || error.status === 410) end();
      else { setState('idle'); armIdle(); }
      status(error.message, true);
    } finally {
      if (run === generation) controller = null;
    }
  }

  el('record').onclick = async () => {
    if (document.getElementById('record').disabled || document.getElementById('synthesize').disabled) {
      status('请先结束独立语音测试，再开始导购。', true);
      return;
    }
    document.getElementById('stop-speech').click();
    document.getElementById('speech').pause();
    document.getElementById('recording').pause();
    const run = ++generation;
    clearTimeout(idleTimer);
    setState('starting');
    status('正在准备麦克风…');
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error('请使用支持录音的浏览器，通过 HTTPS 或 localhost 打开页面。');
      }
      // Unlock playback during the physical button gesture, before network requests.
      await audioContext();
      if (run !== generation) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}
      });
      if (run !== generation) { stream.getTracks().forEach((track) => track.stop()); return; }
      mic = stream;
      if (!session) {
        const response = await fetch('/api/dialogue/sessions', {method: 'POST'});
        if (!response.ok) throw await responseError(response);
        const created = await response.json();
        if (run !== generation) { deleteSession(created.session_id); return; }
        session = created;
        el('log').replaceChildren();
        el('notice').textContent = created.notice;
        el('limits').textContent = `每次最多 ${created.limits.reply_sentences} 句，` +
          `共 ${created.limits.reply_chars} 字；最多 ${created.limits.max_turns} 轮，` +
          `空闲 ${created.limits.idle_seconds} 秒结束。`;
        el('turn').textContent = `第 0 / ${created.limits.max_turns} 轮`;
      }
      const lease = await fetch(`/api/dialogue/sessions/${session.session_id}/recording`, {method: 'POST'});
      if (!lease.ok) throw await responseError(lease);
      if (run !== generation) return;
      const type = ['audio/webm;codecs=opus', 'audio/mp4'].find((t) => MediaRecorder.isTypeSupported(t));
      recorder = new MediaRecorder(mic, type ? {mimeType: type} : undefined);
      const parts = [];
      const currentRecorder = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size) parts.push(event.data); };
      recorder.onerror = () => {
        if (run === generation) end('录音失败，请重新开始。');
      };
      recorder.onstop = () => {
        const blob = new Blob(parts, {type: currentRecorder.mimeType || 'audio/webm'});
        releaseMic();
        if (run === generation) send(blob, run);
      };
      recorder.start();
      setState('recording');
      status(`正在录音，最多 ${session.max_audio_seconds} 秒。说完请点击停止并发送。`);
      recordTimer = setTimeout(() => {
        if (run === generation && recorder?.state === 'recording') recorder.stop();
      }, session.max_audio_seconds * 1000);
    } catch (error) {
      if (run !== generation) return;
      releaseMic();
      if (error.status === 410) end();
      else { setState('idle'); armIdle(); }
      status(`无法开始录音：${error.message}`, true);
    }
  };
  el('stop').onclick = () => {
    if (state === 'recording' && recorder?.state === 'recording') {
      setState('processing');
      recorder.stop();
    }
  };
  el('end').onclick = () => end();
  el('resume').onclick = async () => {
    try {
      await audioContext();
      el('resume').hidden = context.state === 'running';
      status('正在播放回答…');
    } catch { status('无法播放声音，请检查浏览器声音权限。', true); }
  };
  window.addEventListener('pagehide', () => end('会话已结束。', true));
})();
