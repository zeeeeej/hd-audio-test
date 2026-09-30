from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import APIRouter, Form, Header, UploadFile
from fastapi.responses import Response, StreamingResponse

from src.business.retail import (
    CatalogProvider,
    DialogueError,
    DialogueSettings,
    Session,
    SessionStore,
    ask_deepseek,
    bounded_reply,
    catalog_hotwords,
    deepseek_key,
    messages_for,
    system_prompt,
)
from src.common import (
    CHINESE_VOICES,
    MAX_AUDIO_SECONDS,
    MAX_UPLOAD_BYTES,
    error_response,
    is_allowed_audio_type,
    request_id,
)

logger = logging.getLogger(__name__)
settings = DialogueSettings()
store = SessionStore(settings)
catalog_provider = CatalogProvider()
VOICE_BASE_URL = os.getenv("VOICE_BASE_URL", "http://127.0.0.1:18000").rstrip("/")
VOICE_API_TOKEN = os.getenv("VOICE_API_TOKEN", "")
VOICE_TIMEOUT = float(os.getenv("VOICE_TIMEOUT_SECONDS", "120"))


@asynccontextmanager
async def lifespan(app):
    async def cleanup():
        while True:
            await asyncio.sleep(min(10, settings.idle_seconds))
            store.cleanup()

    async with anyio.create_task_group() as group:
        group.start_soon(cleanup)
        try:
            yield
        finally:
            for session_id in list(store.sessions):
                store.close(session_id)
            group.cancel_scope.cancel()


router = APIRouter(prefix="/api/dialogue", lifespan=lifespan)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=VOICE_TIMEOUT)


def voice_headers(req_id: str) -> dict[str, str]:
    headers = {"X-Request-ID": req_id}
    if VOICE_API_TOKEN:
        headers["Authorization"] = f"Bearer {VOICE_API_TOKEN}"
    return headers


@router.post("/sessions", status_code=201)
async def create_session() -> Response:
    req_id = request_id()
    try:
        deepseek_key()
        catalog = catalog_provider.load()
        words = catalog_hotwords(catalog)
        prompt = system_prompt(catalog, settings)
        messages_for(prompt, [], "", settings)  # Validate static budget before recording.
        session = Session(id=request_id(), catalog=catalog, prompt=prompt, hotwords=words)
        store.add(session)
        return Response(
            json.dumps({
                "session_id": session.id, "limits": settings.public(),
                "max_audio_seconds": MAX_AUDIO_SECONDS,
                "notice": catalog.notice, "mode": catalog.mode,
            }, ensure_ascii=False),
            status_code=201, media_type="application/json", headers={"X-Request-ID": req_id},
        )
    except DialogueError as exc:
        return error_response(exc.status, exc.code, exc.message, req_id)


@router.delete("/sessions/{session_id}", status_code=204)
async def end_session(session_id: str) -> Response:
    store.close(session_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/recording", status_code=204)
async def begin_recording(session_id: str) -> Response:
    req_id = request_id()
    try:
        session = store.reserve(session_id)
        store.release(session)
        # Bounded lease: a 60-second recording must not expire at the 60-second idle limit.
        session.recording_until = time.monotonic() + MAX_AUDIO_SECONDS + 5
        return Response(status_code=204)
    except DialogueError as exc:
        return error_response(exc.status, exc.code, exc.message, req_id)


def event_line(kind: str, req_id: str, **values) -> bytes:
    return (json.dumps({"type": kind, "request_id": req_id, **values}, ensure_ascii=False)
            + "\n").encode()


async def turn_events(
    session: Session, data: bytes, content_type: str, filename: str, voice: str,
    key: str, req_id: str,
):
    stage = "asr"
    started = time.perf_counter()
    timings: dict[str, int] = {}
    try:
        with anyio.CancelScope() as scope:
            session.cancel_scope = scope
            if session.closed:
                return
            async with make_client() as client:
                yield event_line("stage", req_id, stage=stage)
                words = session.hotwords
                async with asyncio.timeout(VOICE_TIMEOUT):
                    response = await client.post(
                        f"{VOICE_BASE_URL}/v1/audio/transcriptions",
                        headers=voice_headers(req_id),
                        files={"file": (filename, data, content_type)},
                        data={"language": "Chinese", "hotwords": json.dumps(words)},
                    )
                if response.status_code != 200:
                    raise DialogueError("asr_failed", "语音识别失败，请重新录音。")
                text = response.json()["text"]
                if not isinstance(text, str):
                    raise ValueError("invalid ASR response")
                text = text.strip()
                if not text:
                    raise DialogueError("empty_transcript", "没有听清，请重新说一遍。", 422)
                messages = messages_for(session.prompt, session.history, text, settings)
                timings["asr_ms"] = round((time.perf_counter() - started) * 1000)
                yield event_line("transcript", req_id, text=text)

                stage = "deepseek"
                yield event_line("stage", req_id, stage=stage)
                model_started = time.perf_counter()
                answer = await ask_deepseek(client, key, messages, settings, req_id)
                timings["deepseek_ms"] = round((time.perf_counter() - model_started) * 1000)
                if session.closed:
                    return
                reply = bounded_reply(answer, settings)
                session.turns += 1
                # JSON mode needs assistant history in the same format as the requested output.
                # Only the bounded, displayed reply is retained, never the original long answer.
                session.history.append((text, json.dumps(
                    {"intent": answer.intent, "reply": reply}, ensure_ascii=False,
                )))
                session.history = (
                    session.history[-settings.history_turns:] if settings.history_turns else []
                )
                end_reason = (
                    "goodbye" if answer.intent == "end" else
                    "turn_limit" if session.turns >= settings.max_turns else None
                )
                session.closed = end_reason is not None
                yield event_line(
                    "answer", req_id, text=reply, intent=answer.intent, turn=session.turns,
                    remaining_turns=settings.max_turns - session.turns,
                    session_ended=session.closed, end_reason=end_reason,
                )

                stage = "tts"
                yield event_line("stage", req_id, stage=stage)
                tts_started = time.perf_counter()
                count = 0
                expected = None
                async with asyncio.timeout(VOICE_TIMEOUT):
                    async with client.stream(
                        "POST", f"{VOICE_BASE_URL}/v1/audio/speech/stream",
                        headers=voice_headers(req_id),
                        json={"input": reply, "voice": voice, "language": "Chinese",
                              "instruct": "用自然、清晰的语气说"},
                    ) as upstream:
                        if upstream.status_code != 200:
                            raise DialogueError("tts_failed", "语音播放失败，请查看文字回答。")
                        async for line in upstream.aiter_lines():
                            if len(line) > 24 * 1024 * 1024:
                                raise ValueError("oversized TTS event")
                            if not line.strip():
                                continue
                            event = json.loads(line)
                            kind = event["type"]
                            if kind == "start" and expected is None and count == 0:
                                expected = event["sentence_count"]
                                if type(expected) is not int or not 1 <= expected <= len(reply):
                                    raise ValueError("invalid TTS sentence count")
                            elif kind == "audio" and expected is not None and count < expected:
                                if (event["index"] != count
                                        or not isinstance(event["audio_base64"], str)
                                        or not event["audio_base64"]
                                        or not isinstance(event["text"], str)):
                                    raise ValueError("invalid TTS audio event")
                                if count == 0:
                                    timings["first_audio_ms"] = round(
                                        (time.perf_counter() - started) * 1000
                                    )
                                yield event_line(
                                    "audio", req_id, index=count, text=event["text"],
                                    audio_base64=event["audio_base64"], media_type="audio/wav",
                                )
                                count += 1
                            elif kind == "done" and expected == count and count > 0:
                                if event["sentence_count"] != count:
                                    raise ValueError("TTS final count mismatch")
                                timings["tts_ms"] = round(
                                    (time.perf_counter() - tts_started) * 1000
                                )
                                timings["total_ms"] = round(
                                    (time.perf_counter() - started) * 1000
                                )
                                yield event_line(
                                    "done", req_id, sentence_count=count, timings=timings,
                                    session_ended=session.closed, end_reason=end_reason,
                                    turn=session.turns,
                                )
                                logger.info("Dialogue complete request_id=%s timings=%s",
                                            req_id, timings)
                                return
                            elif kind == "error":
                                raise DialogueError(
                                    "tts_failed", "语音播放失败，请查看文字回答。"
                                )
                            else:
                                raise ValueError("invalid TTS event order")
                        raise DialogueError("tts_incomplete", "语音传输中断，请查看文字回答。")
    except DialogueError as exc:
        logger.warning("Dialogue failed request_id=%s stage=%s code=%s", req_id, stage, exc.code)
        yield event_line("error", req_id, stage=stage, session_ended=session.closed,
                         error={"code": exc.code, "message": exc.message, "request_id": req_id})
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        code = f"{stage}_timeout" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else (
            f"{stage}_invalid_response"
        )
        logger.warning("Dialogue failed request_id=%s stage=%s code=%s", req_id, stage, code)
        message = "语音播放失败，请查看文字回答。" if stage == "tts" else "服务暂不可用，请再试一次。"
        yield event_line("error", req_id, stage=stage, session_ended=session.closed,
                         error={"code": code, "message": message, "request_id": req_id})
    finally:
        store.release(session)


@router.post("/sessions/{session_id}/turns")
async def submit_turn(
    session_id: str, file: UploadFile, voice: str = Form(default="Vivian"),
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    session = None
    handed_off = False
    try:
        if voice not in CHINESE_VOICES:
            raise DialogueError("invalid_voice", "请选择支持的中文音色。", 422)
        if not is_allowed_audio_type(file.content_type):
            raise DialogueError("unsupported_audio", "不支持此录音格式。", 415)
        key = deepseek_key()
        session = store.reserve(session_id)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise DialogueError("empty_audio", "录音为空，请重新录音。", 400)
        if len(data) > MAX_UPLOAD_BYTES:
            raise DialogueError("audio_too_large", "录音文件过大，请缩短录音。", 413)
        handed_off = True
        return StreamingResponse(
            turn_events(session, data, file.content_type or "audio/webm",
                        "recording", voice, key, req_id),
            media_type="application/x-ndjson",
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except DialogueError as exc:
        return error_response(exc.status, exc.code, exc.message, req_id)
    finally:
        await file.close()
        if session and not handed_off:
            store.release(session)
