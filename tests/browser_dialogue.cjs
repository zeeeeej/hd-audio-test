// Optional browser regression: start business-api, then run with playwright-core in NODE_PATH.
const assert = require('node:assert/strict');
const {chromium} = require('playwright-core');

function wav(seconds = 0.8) {
  const samples = Math.round(8000 * seconds);
  const bytes = Buffer.alloc(44 + samples * 2);
  bytes.write('RIFF'); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write('WAVEfmt ', 8);
  bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20); bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(8000, 24); bytes.writeUInt32LE(16000, 28);
  bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34);
  bytes.write('data', 36); bytes.writeUInt32LE(samples * 2, 40);
  return bytes.toString('base64');
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
  });
  try {
    const context = await browser.newContext({permissions: ['microphone']});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    let sessions = 0, turns = 0, deleted = 0, mode = 'normal';
    let maxAudio = 60, idle = 60;
    const records = [];
    await page.route('**/api/dialogue/**', async (route) => {
      const url = new URL(route.request().url());
      if (route.request().method() === 'DELETE') {
        deleted += 1;
        return route.fulfill({status: 204});
      }
      if (url.pathname.endsWith('/sessions')) {
        sessions += 1;
        return route.fulfill({status: 201, json: {
          session_id: `session-${sessions}`, notice: '全部为演示数据。', max_audio_seconds: maxAudio,
          limits: {max_turns: 6, reply_sentences: 2, reply_chars: 60, idle_seconds: idle},
        }});
      }
      if (url.pathname.endsWith('/recording')) return route.fulfill({status: 204});
      if (url.pathname.endsWith('/turns')) {
        turns += 1;
        records.push(url.pathname);
        assert.ok(route.request().postDataBuffer().length > 0);
        const events = [
          {type: 'stage', stage: 'asr'}, {type: 'transcript', text: turns === 1 ? '推荐无糖茶。' : '它多少钱？'},
          {type: 'stage', stage: 'deepseek'},
          {type: 'answer', text: '东方树叶演示价6元。', turn: turns, session_ended: mode === 'terminal'},
          {type: 'stage', stage: 'tts'},
        ];
        if (mode === 'error') events.push({type: 'error', error: {message: '语音播放失败，请查看文字回答。'}});
        else {
          events.push({type: 'audio', index: 0, text: '东方树叶演示价6元。', audio_base64: wav()});
          if (mode !== 'incomplete') events.push({
            type: 'done', sentence_count: 1, session_ended: mode === 'terminal',
            timings: {asr_ms: 20, deepseek_ms: 30, tts_ms: 40, first_audio_ms: 90},
          });
        }
        return route.fulfill({status: 200, contentType: 'application/x-ndjson',
          body: events.map((e) => JSON.stringify({...e, request_id: 'browser-smoke'})).join('\n') + '\n'});
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    await page.goto(process.env.APP_URL || 'http://127.0.0.1:18743');
    const record = page.locator('#dialogue-record');
    const stop = page.locator('#dialogue-stop');
    const end = page.locator('#dialogue-end');
    const status = page.locator('#dialogue-status');
    async function begin() {
      await record.click();
      await page.waitForFunction(() => !document.getElementById('dialogue-stop').disabled);
    }
    async function complete() {
      await begin();
      await page.waitForTimeout(150);  // Produce a nonempty MediaRecorder chunk.
      await stop.click();
      await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('正在播放回答'));
      assert.equal(await record.isDisabled(), true);
      await page.waitForFunction(() => !document.getElementById('dialogue-record').disabled);
      assert.match(await status.textContent(), /播放完成/);
    }
    await complete();
    await complete();
    assert.equal(sessions, 1);
    assert.equal(turns, 2);
    assert.equal(records[0], records[1]);
    assert.match(await page.locator('#dialogue-log').textContent(), /它多少钱/);

    await begin();
    await end.click();
    assert.equal(turns, 2); // Ending while recording must not submit stale audio.
    await page.waitForFunction(() => !document.getElementById('dialogue-record').disabled);
    await begin();
    assert.equal(sessions, 2);
    assert.equal(await page.locator('#dialogue-log p').count(), 0);
    await end.click();

    maxAudio = 0.2;
    await record.click(); // Automatic recording limit submits without another click.
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('播放完成'));
    assert.equal(turns, 3);
    await end.click();

    mode = 'error';
    await record.click();
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('语音播放失败'));
    assert.equal(await record.isEnabled(), true);
    assert.match(await page.locator('#dialogue-log').textContent(), /演示价6元/);
    await end.click();

    mode = 'incomplete';
    await record.click();
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('提前中断'));
    assert.equal(await record.isEnabled(), true);
    await end.click();

    mode = 'normal'; idle = 0.2;
    await record.click();
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('空闲时间已到'));
    assert.equal(await end.isDisabled(), true);
    assert.ok(deleted >= 6);

    idle = 60;
    await record.click();
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('正在播放回答'));
    await end.click();
    await page.waitForTimeout(1000);
    assert.match(await status.textContent(), /会话已结束/);
    assert.equal(await record.isEnabled(), true);

    mode = 'terminal';
    await record.click();
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('正在播放回答'));
    assert.equal(await record.isDisabled(), true);
    await page.waitForFunction(() => document.getElementById('dialogue-status').textContent.includes('本次对话已结束'));
    assert.equal(await record.isEnabled(), true);
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: recording, multi-turn, playback lock, cancel recording/playback, new customer, recording limit, TTS error, incomplete stream, idle timeout, final answer playback.');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
