"""WhatsApp customer-support webhook: Meta Cloud API -> Gemini -> WhatsApp reply.

Endpoints
    GET  /webhook   Meta's subscription handshake (echoes hub.challenge).
    POST /webhook   Inbound messages: verify signature, ask Gemini, reply via Graph API.
    GET  /health    Configuration check - reports any unset environment variable.

Conversation state lives on the Gemini side: each reply is threaded onto the
previous one with previous_interaction_id, so this process only caches that id
per customer, never the transcript.

Free-tier note: a project with no billing account cannot be charged - over the
quota the API returns 429 and the customer gets BOT_BUSY_REPLY. Every reply logs
its token usage so the daily burn is visible before customers notice.

The whole exchange happens inside the POST request. This runs as one serverless
function (see api/index.py + vercel.json) and the process is frozen as soon as
the response is sent, so there is no after-response background work to defer to.
Duplicate deliveries - Meta retries when a 200 is slow - are dropped by message
id, best-effort (see ConversationCache).

Local run:
    uvicorn app:app --reload --port 8000

The signature check has no bypass, so a local POST has to be signed:
    body='{"entry":[]}'
    sig=$(printf '%s' "$body" | openssl dgst -sha256 -hmac "$WHATSAPP_APP_SECRET" -hex | awk '{print $2}')
    curl -X POST localhost:8000/webhook -H "X-Hub-Signature-256: sha256=$sig" \
         -H 'Content-Type: application/json' -d "$body"
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from google import genai
from google.genai import errors as genai_errors

try:  # convenience for local runs; on Vercel the environment is already set
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# --------------------------------------------------------------------------- #
# Configuration (read once at import - the platform injects env before import)
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("BOT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("whatsapp-support-bot")

WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")
WHATSAPP_API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v21.0")

# Everything the bot itself tunes is BOT_*: a plain GEMINI_* or LOG_LEVEL name
# can already be set in the host environment by unrelated tooling.
MODEL = os.environ.get("BOT_MODEL", "gemini-3.8-flash")
# minimal | low | medium | high - how much the model thinks before answering.
# Keep it low: the reply has to be sent before Meta gives up and retries.
THINKING_LEVEL = os.environ.get("BOT_THINKING_LEVEL", "low")
MAX_TOKENS = int(os.environ.get("BOT_MAX_TOKENS", "1024"))
TIMEOUT_SECONDS = float(os.environ.get("BOT_TIMEOUT_SECONDS", "20"))
# Pipeline test: when set, every message gets this exact reply and Gemini is
# never called (nor is GEMINI_API_KEY required). Unset it to go live.
FIXED_REPLY = os.environ.get("BOT_FIXED_REPLY", "")

REQUIRED_ENV = (
    "WHATSAPP_ACCESS_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_VERIFY_TOKEN",
    "WHATSAPP_APP_SECRET",
    "GEMINI_API_KEY",
)

DEFAULT_SYSTEM_PROMPT = """You are the customer-support assistant for this business, replying inside WhatsApp.

- Reply in the language the customer wrote in.
- Keep it to a few sentences - WhatsApp-sized. No markdown headings, tables or code blocks; *bold* and _italic_ are fine.
- Answer only from what you actually know. Never invent order numbers, prices, policies, stock or delivery dates.
- If you need one detail to proceed (order number, e-mail), ask for exactly that one detail.
- For refunds, complaints or anything you cannot verify, say a colleague will follow up and ask for the best contact detail."""

SYSTEM_PROMPT = os.environ.get("BOT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

# Canned replies for the paths where the model produced nothing usable.
ERROR_REPLY = os.environ.get(
    "BOT_ERROR_REPLY",
    "Sorry, I could not process that just now. Please try again in a moment."
)
# Sent on a 429: the free-tier per-minute or per-day quota is spent.
BUSY_REPLY = os.environ.get(
    "BOT_BUSY_REPLY",
    "I am handling a lot of messages right now. Please try again in a few minutes.",
)
REFUSAL_REPLY = os.environ.get(
    "BOT_REFUSAL_REPLY",
    "I cannot help you with that. Could you ask something else?"
)
UNSUPPORTED_REPLY = os.environ.get(
    "BOT_UNSUPPORTED_REPLY",
    "I can only read text messages at the moment. Could you describe it in writing?",
)

# WhatsApp caps a text body at 4096 characters; stay under it with a margin.
TEXT_CHUNK_LIMIT = 4000


def missing_config() -> list[str]:
    """Names of required environment variables that are unset or empty."""
    return [
        name
        for name in REQUIRED_ENV
        if not os.environ.get(name) and not (FIXED_REPLY and name == "GEMINI_API_KEY")
    ]


# --------------------------------------------------------------------------- #
# Clients (module level so warm invocations reuse the connection pools)
# --------------------------------------------------------------------------- #

_gemini: genai.Client | None = None
_http: httpx.AsyncClient | None = None


def _gemini_client() -> genai.Client:
    global _gemini
    if _gemini is None:
        # The SDK reads GEMINI_API_KEY itself. Two attempts only: the request
        # has to finish inside the function's budget, and the SDK would
        # otherwise retry a 429 several times over.
        _gemini = genai.Client(http_options={"retry_options": {"attempts": 2}})
    return _gemini


def _http_client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    return _http


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if _http is not None:
        await _http.aclose()


app = FastAPI(title="WhatsApp Support Bot", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Conversation memory
# --------------------------------------------------------------------------- #


class ConversationCache:
    """Per-process message dedupe plus the Gemini thread id per customer.

    Gemini stores the transcript itself, so all this keeps is the id of the
    last interaction with each customer. It lives only as long as one warm
    instance, so treat it as an optimisation rather than storage: after a cold
    start the next message opens a new thread, and two concurrent instances do
    not see each other's entries. Point these methods at Redis/Upstash when
    durable threading or cross-instance dedupe matters.
    """

    def __init__(self, max_users: int = 500, ttl: float = 1800.0):
        self._max_users = max_users
        self._ttl = ttl
        self._threads: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._seen: OrderedDict[str, float] = OrderedDict()

    def already_handled(self, message_id: str) -> bool:
        """True if this message id was processed before; records it otherwise."""
        now = time.monotonic()
        seen_at = self._seen.get(message_id)
        if seen_at is not None and now - seen_at < self._ttl:
            return True
        self._seen[message_id] = now
        while len(self._seen) > self._max_users * 4:
            self._seen.popitem(last=False)
        return False

    def thread_id(self, wa_id: str) -> str | None:
        """The last interaction id for this customer, if still fresh."""
        entry = self._threads.get(wa_id)
        if entry is None or time.monotonic() - entry[0] > self._ttl:
            self._threads.pop(wa_id, None)
            return None
        self._threads.move_to_end(wa_id)
        return entry[1]

    def set_thread_id(self, wa_id: str, interaction_id: str) -> None:
        self._threads[wa_id] = (time.monotonic(), interaction_id)
        self._threads.move_to_end(wa_id)
        while len(self._threads) > self._max_users:
            self._threads.popitem(last=False)


_cache = ConversationCache()


# --------------------------------------------------------------------------- #
# Webhook payload parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Incoming:
    message_id: str
    wa_id: str
    kind: str
    text: str | None
    profile_name: str | None


def _message_text(message: dict[str, Any]) -> str | None:
    """The customer's words, for the message types that carry any."""
    kind = message.get("type")
    if kind == "text":
        return (message.get("text") or {}).get("body")
    if kind == "button":  # quick reply on a template message
        return (message.get("button") or {}).get("text")
    if kind == "interactive":
        interactive = message.get("interactive") or {}
        reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return reply.get("title") or reply.get("id")
    return None  # image, audio, document, location, sticker, ...


def incoming_messages(payload: dict[str, Any]) -> list[Incoming]:
    """Flatten Meta's nested webhook body into the messages we should answer.

    Delivery and read receipts arrive under `statuses` instead of `messages`,
    so they fall out here and the caller simply has nothing to do.
    """
    found: list[Incoming] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            names = {
                contact.get("wa_id"): (contact.get("profile") or {}).get("name")
                for contact in value.get("contacts") or []
            }
            for message in value.get("messages") or []:
                wa_id = message.get("from") or ""
                message_id = message.get("id") or ""
                if not wa_id or not message_id:
                    continue
                found.append(
                    Incoming(
                        message_id=message_id,
                        wa_id=wa_id,
                        kind=message.get("type") or "unknown",
                        text=_message_text(message),
                        profile_name=names.get(wa_id),
                    )
                )
    return found


def valid_signature(raw_body: bytes, header: str) -> bool:
    """Check Meta's X-Hub-Signature-256 against the raw request body."""
    if not WHATSAPP_APP_SECRET or not header.startswith("sha256="):
        return False
    expected = hmac.new(
        WHATSAPP_APP_SECRET.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256=") :])


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


class Answer(NamedTuple):
    status: str  # "ok" | "refused" | "busy" | "error"
    text: str
    thread_id: str | None = None  # interaction id to continue from next time


async def _create_interaction(user_text: str, thread_id: str | None):
    """One Interactions API call. store=True is what makes threading work."""
    return await asyncio.wait_for(
        _gemini_client().aio.interactions.create(
            model=MODEL,
            input=user_text,
            system_instruction=SYSTEM_PROMPT,
            previous_interaction_id=thread_id,
            store=True,  # without this there is no thread to continue
            generation_config={
                "max_output_tokens": MAX_TOKENS,
                "thinking_config": {"thinking_level": THINKING_LEVEL},
            },
            timeout=TIMEOUT_SECONDS,
        ),
        timeout=TIMEOUT_SECONDS + 5,
    )


async def ask_gemini(user_text: str, thread_id: str | None) -> Answer:
    """Answer one customer message, continuing their thread when we have its id."""
    try:
        try:
            interaction = await _create_interaction(user_text, thread_id)
        except genai_errors.ClientError as exc:
            # A cached id the API will not accept - expired, or from another
            # deployment. Answer as a new thread instead of failing the message.
            if thread_id is None or exc.code not in (400, 404):
                raise
            log.warning("thread %s rejected (%s), starting a new one", thread_id, exc.code)
            interaction = await _create_interaction(user_text, None)
    except asyncio.TimeoutError:
        log.error("Gemini timed out after %.0fs", TIMEOUT_SECONDS)
        return Answer("error", ERROR_REPLY)
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            # Free tier: the per-minute or per-day quota is spent. Nothing is
            # billed - the request is simply refused until the quota resets.
            log.error("quota exhausted (429): %s", exc.message)
            return Answer("busy", BUSY_REPLY)
        log.error("Gemini rejected the request (%s): %s", exc.code, exc.message)
        return Answer("error", ERROR_REPLY)
    except genai_errors.ServerError as exc:
        log.error("Gemini server error (%s): %s", exc.code, exc.message)
        return Answer("error", ERROR_REPLY)
    except genai_errors.APIError as exc:
        log.error("Gemini API error: %s", exc)
        return Answer("error", ERROR_REPLY)
    except httpx.HTTPError as exc:
        log.error("could not reach Gemini: %s", exc)
        return Answer("error", ERROR_REPLY)

    usage = interaction.usage
    if usage is not None:
        # Watch these numbers to see how close a normal day runs to the
        # free-tier ceiling; AI Studio reports the same totals a day later.
        log.info(
            "tokens in=%s out=%s thought=%s total=%s",
            usage.total_input_tokens,
            usage.total_output_tokens,
            usage.total_thought_tokens,
            usage.total_tokens,
        )

    text = (interaction.output_text or "").strip()

    if interaction.errors:
        for err in interaction.errors:
            log.error("interaction error %s: %s", err.code, err.message)
        if not text:
            return Answer("error", ERROR_REPLY)

    if not text:
        # No output and no error means the model declined to produce anything -
        # a safety block reaches us this way, so answer as a decline.
        log.warning("empty output (status=%s)", interaction.status)
        return Answer("refused", REFUSAL_REPLY)

    return Answer("ok", text, interaction.id)


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #


def _chunks(text: str, limit: int = TEXT_CHUNK_LIMIT) -> Iterator[str]:
    """Split an over-long reply on a line or word boundary."""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut].rstrip()
        text = text[cut:].lstrip()
    if text:
        yield text


async def send_text(to: str, body: str) -> None:
    """Deliver a text reply through the WhatsApp Cloud API."""
    url = (
        f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
        f"/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    client = _http_client()
    for chunk in _chunks(body):
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        }
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error(
                "WhatsApp send rejected (%s): %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return
        except httpx.HTTPError as exc:
            log.error("WhatsApp send failed: %s", exc)
            return


async def handle_message(message: Incoming) -> None:
    if FIXED_REPLY:
        log.info("fixed reply to %s (%s)", message.wa_id, message.kind)
        await send_text(message.wa_id, FIXED_REPLY)
        return

    if message.text is None:
        log.info("unsupported message type %s from %s", message.kind, message.wa_id)
        await send_text(message.wa_id, UNSUPPORTED_REPLY)
        return

    log.info("message from %s (%s)", message.profile_name or "?", message.wa_id)
    answer = await ask_gemini(message.text, _cache.thread_id(message.wa_id))
    if answer.status == "ok" and answer.thread_id:
        _cache.set_thread_id(message.wa_id, answer.thread_id)
    await send_text(message.wa_id, answer.text)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request) -> Response:
    """Meta's one-time subscription handshake."""
    params = request.query_params
    if not WHATSAPP_VERIFY_TOKEN:
        log.error("WHATSAPP_VERIFY_TOKEN is unset; cannot complete the handshake")
        return PlainTextResponse("not configured", status_code=503)
    if params.get("hub.mode") == "subscribe" and hmac.compare_digest(
        params.get("hub.verify_token", ""), WHATSAPP_VERIFY_TOKEN
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    log.warning("handshake rejected: verify token mismatch")
    return PlainTextResponse("verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request) -> Response:
    raw = await request.body()  # the signature covers the bytes, not the parsed JSON
    if not valid_signature(raw, request.headers.get("x-hub-signature-256", "")):
        log.warning("rejected: missing or invalid X-Hub-Signature-256")
        return JSONResponse({"error": "invalid signature"}, status_code=403)

    missing = missing_config()
    if missing:
        # 503 rather than 200: let Meta retry once the deployment is configured.
        log.error("unset environment variables: %s", ", ".join(missing))
        return JSONResponse({"error": "not configured"}, status_code=503)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("rejected: body is not JSON")
        return JSONResponse({"error": "invalid json"}, status_code=400)

    for message in incoming_messages(payload):
        if _cache.already_handled(message.message_id):
            log.info("duplicate delivery of %s, skipped", message.message_id)
            continue
        try:
            await handle_message(message)
        except Exception:  # never fail the webhook over a single message
            log.exception("failed to handle %s", message.message_id)

    # Always 200 once accepted, so Meta stops retrying this delivery.
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health() -> dict[str, Any]:
    missing = missing_config()
    return {
        "status": "ok" if not missing else "misconfigured",
        "model": MODEL,
        "fixed_reply": bool(FIXED_REPLY),
        "thinking_level": THINKING_LEVEL,
        "graph_api_version": WHATSAPP_API_VERSION,
        "missing_env": missing,
    }


STATUS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WhatsApp Support Bot</title>
<style>
  :root {{ color-scheme: light dark; --bg: #f6f7f9; --card: #fff; --fg: #1d2129; --muted: #667085; --ok: #12b76a; --bad: #f04438; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #101214; --card: #1a1d21; --fg: #e6e8eb; --muted: #98a2b3; }} }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }}
  main {{ background: var(--card); padding: 2rem; border-radius: 12px; max-width: 28rem; margin: 1rem; box-shadow: 0 1px 3px rgb(0 0 0 / .12); }}
  h1 {{ margin: 0 0 .5rem; font-size: 1.4rem; }}
  .dot {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; background: {color}; margin-right: .4rem; }}
  p, li {{ color: var(--muted); }}
  code {{ font-size: .9em; }}
</style>
</head>
<body>
<main>
  <h1>WhatsApp Support Bot</h1>
  <p><span class="dot"></span>{status}</p>
  <ul>
    <li>Mode: {mode}</li>
    <li>Webhook: <code>/webhook</code></li>
    <li>Details: <a href="/health">/health</a></li>
  </ul>
</main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    missing = missing_config()
    return STATUS_PAGE.format(
        color="var(--ok)" if not missing else "var(--bad)",
        status="Running" if not missing else f"Missing configuration: {', '.join(missing)}",
        mode="fixed reply (test)" if FIXED_REPLY else f"Gemini ({MODEL})",
    )
