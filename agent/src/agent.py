import asyncio
import logging
import os
import re
import socket
import textwrap
from collections.abc import Awaitable, Callable

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics, openai

import vision
from robot_gate import (
    ESTOP_NOW, FREEZE_WORD_RE, RETRYING_SPEECH, SENT_TEXT, STOP_FAILED_SPEECH, MatchMode, RobotGate, UserTurn,
    is_stop_failure,
)
from station_client import StationClient
from speech_guard import scrub_transcription, scrub_trigger_words
from harness_client import HarnessClient, LocalRunner, SshRunner
from station_monitor import HarnessMonitor, StationMonitor

logger = logging.getLogger("agent")

# Secrets come from the environment only (secrets.sh); no .env loading here.


def _make_llm() -> llm.LLM:
    """LiveKit Inference by default; AGENT_LLM=omlx uses an OpenAI-compatible server instead.

    That server comes from the environment only: OMLX_BASE_URL (e.g. http://<host>:<port>/v1), OMLX_MODEL, OMLX_API_KEY.
    parallel_tool_calls=False is defense in depth only (the code gate is what enforces confirmation).
    Verified in installed source: inference.LLM(extra_kwargs=...) and openai.LLM(parallel_tool_calls=...).
    """
    if os.environ.get("AGENT_LLM") == "omlx":
        missing = [k for k in ("OMLX_BASE_URL", "OMLX_MODEL", "OMLX_API_KEY") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"AGENT_LLM=omlx needs {', '.join(missing)} in the environment")
        return openai.LLM(
            model=os.environ["OMLX_MODEL"],
            base_url=os.environ["OMLX_BASE_URL"],
            api_key=os.environ["OMLX_API_KEY"],
            parallel_tool_calls=False,
        )
    return inference.LLM(model="google/gemma-4-31b-it", extra_kwargs={"parallel_tool_calls": False})


def latest_user_turn(session: AgentSession) -> UserTurn | None:
    """The latest user chat message in the session history, with its ordinal. None if there is none."""
    history = getattr(session, "history", None)
    if history is None:
        return None
    users = [i for i in session.history.items if i.type == "message" and i.role == "user"]
    if not users:
        return None
    last = users[-1]
    prev = ""
    for item in session.history.items:
        if item is last:
            break
        if item.type == "message" and item.role == "assistant":
            prev = item.text_content or ""
    return UserTurn(id=last.id, text=last.text_content or "", count=len(users), last_assistant=prev)


def say_with_retry(session, text: str):
    """session.say(text); if that speech ends interrupted (the human's own speech cut it off), say it once more.

    Never allow_interruptions=False: LiveKit then discards mic audio and the emergency word would go unheard.
    Test fakes may return None, so the handle is probed with hasattr. May raise like session.say.
    """
    handle = session.say(text)
    if hasattr(handle, "add_done_callback") and hasattr(handle, "interrupted"):

        def _again(h) -> None:
            if not h.interrupted:
                return
            try:
                session.say(text)  # once: the retry is not itself retried
            except Exception:
                logger.exception("could not re-speak %r", text)

        try:
            handle.add_done_callback(_again)
        except Exception:
            logger.exception("could not watch the speech handle")
    return handle


def speak_alarm(session, interrupt: bool = False, text: str = STOP_FAILED_SPEECH) -> None:
    """Say the stop-failed line (or `text`, e.g. naming the program that holds the arms). Never raises."""
    if interrupt and hasattr(session, "interrupt"):
        try:
            session.interrupt()  # verified: AgentSession.interrupt(*, force=False) -> Future
        except Exception:
            logger.exception("interrupt() failed before the stop-failed line")
    try:
        say_with_retry(session, text)
    except Exception:
        logger.exception("could not speak the stop-failed line")


def make_state_handler(gate: RobotGate):
    """Handler for session.on("agent_state_changed"): restart a pending proposal's TTL when the agent stops speaking."""

    def _on_state(ev) -> None:
        if getattr(ev, "old_state", None) == "speaking" and getattr(ev, "new_state", None) != "speaking":
            gate.restart_pending_clock()

    return _on_state


def make_transcript_handler(gate: RobotGate, session):
    """Handler for session.on("user_input_transcribed"): fast-path stop, speaking the alarm if it fails."""

    def _on_transcript(ev) -> None:
        task = gate.fast_stop(ev.transcript, is_final=bool(getattr(ev, "is_final", True)))
        if task is not None:
            task.add_done_callback(_speak_if_stop_failed(session, gate))

    return _on_transcript


# Chosen from a live check on 2026-11-08 (one masked station frame, the agent's prompts; see the build notes'
# vlm-eval-2026-11-08.md). gemini-3.6-flash: as accurate and fast as 3.5-flash (1.6-2.2 s) at half the price.
# gemini-3.8-flash: the sharpest Flash (saw a half-hidden ball), 2.8-4.6 s, same price: the second try.
# Neither id is in the installed SDK's Literal list yet; LiveKit Inference accepted both with images.
DEFAULT_VISION_MODEL = "google/gemini-3.6-flash"
DEFAULT_VISION_FALLBACK_MODEL = "google/gemini-3.8-flash"
VISION_TIMEOUT_S = 12.0  # the whole look, both tries (the tool's own wait_for)
VISION_PRIMARY_TRY_S = 6.0  # the first model's share; the second try gets what is left of VISION_TIMEOUT_S
KEYTERMS = [
    "LiveKit",
    "Dream",
    "freeze",
    "halt",
    # no "stop" or "cut": neither is a trigger, and boosting "cut" would turn "cup" into one
    "golf ball",
    "tower",
    "hub",
    "cable",
    "peg",
    "block",
    "bin",
    "black ball",
]

VisionFn = Callable[[list[bytes], str], Awaitable[str]]
CameraFn = Callable[[str], Awaitable[bytes]]


def vision_fallback_model() -> str | None:
    """The second-try vision model: VISION_FALLBACK_MODEL (off/none/empty disables it), else the default."""
    m = os.environ.get("VISION_FALLBACK_MODEL")
    if m is None:
        return DEFAULT_VISION_FALLBACK_MODEL
    return None if m.strip().lower() in ("", "off", "none") else m.strip()


async def ask_vision_llm(images: list[bytes], prompt: str, model: str | None = None) -> str:
    """Ask the vision model about already privacy-processed JPEGs (LiveKit Inference).

    An explicit `model` gets exactly one call (eval scripts). Otherwise the primary (VISION_MODEL, else
    DEFAULT_VISION_MODEL) gets VISION_PRIMARY_TRY_S; if it fails, times out or answers empty, the fallback model gets
    one try in the time the caller has left."""
    if model:
        return await _ask_vision_once(model, images, prompt)
    primary = os.environ.get("VISION_MODEL") or DEFAULT_VISION_MODEL
    fallback = vision_fallback_model()
    if not fallback or fallback == primary:
        return await _ask_vision_once(primary, images, prompt)
    try:
        text = await asyncio.wait_for(_ask_vision_once(primary, images, prompt), VISION_PRIMARY_TRY_S)
        if text:
            return text
        logger.warning("vision model %s gave no answer; trying %s", primary, fallback)
    except Exception as exc:  # incl. TimeoutError; a cancel from the caller's own timeout still propagates
        logger.warning("vision model %s failed (%s: %s); trying %s", primary, type(exc).__name__, exc, fallback)
    return await _ask_vision_once(fallback, images, prompt)


async def _ask_vision_once(model: str, images: list[bytes], prompt: str) -> str:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=VISION_SYSTEM)
    content: list = [prompt] + [
        llm.ImageContent(image=vision.data_url(j), inference_detail="low") for j in images
    ]
    chat_ctx.add_message(role="user", content=content)
    vllm = inference.LLM(model=model)
    try:
        resp = await vllm.chat(chat_ctx=chat_ctx).collect()
    finally:
        await vllm.aclose()
    return (resp.text or "").strip()


VISION_SYSTEM = (
    "You describe camera images for a robot operator in at most two short plain spoken sentences, no markdown. "
    "Say what you can see and what you can't. Give a number only if you can actually count it in the image; "
    "otherwise say 'I can't tell' and why. Never invent counts, heights or ports. "
    "Describe only table objects: cables and the hub, blocks, pegs, a tower, golf balls, a black ball and the bin. "
    "Never describe people, faces or clothing, and never read laptop or phone screens or any other on-screen text. "
    "Parts of the image are intentionally masked or blurred for privacy; ignore them."
)

# Prompts shared by the tools and smoke/vision_eval.py.
PROMPTS = {
    "look_at_table": "This is the overhead view of the table. Answer only about table objects. Question: {question}",
    "count_golf_balls": (
        "Count the golf balls visible on the table. Give the count only if you can count them clearly; "
        "if some may be hidden or out of view, say how many you can see and that it may be more."
    ),
    "check_tower": (
        "Is a tower of stacked blocks standing? If yes, estimate its height in inches by comparing it with a "
        "visible reference such as the block size, and say it is an estimate. If it has fallen or you can't "
        "tell, say so."
    ),
    "check_cables": (
        "Images are the overhead camera, then the left wrist camera, then the right wrist camera. "
        "Which hub ports still have a cable plugged in? Name ports only if clearly visible and say which "
        "you can't see."
    ),
    "check_progress": (
        "This is the hill climb progress check on the overhead view. Report briefly, only what you can see: "
        "1) how many hub cables look plugged in versus unplugged, or can't tell; "
        "2) whether a tower is standing and roughly how tall in inches (an estimate, using the block size); "
        "3) how many golf balls are in the bin versus on the table; "
        "4) whether a black ball is in the bin. Give a number only if you can count it, otherwise say can't tell."
    ),
}

INSTRUCTIONS = textwrap.dedent(
    """\
    You are a friendly robot-arm operator at a creative hackathon demo. You talk with a human and control one robot arm that runs a Dream-1 policy.

    # Output rules

    - Voice only: plain spoken sentences. No markdown, lists, code, or emojis.
    - Keep replies to one or two short sentences. Ask one question at a time.
    - Do not read out tool names, parameters, or raw data. Summarize status in plain words.

    # Safety and confirmation (strict)

    - Do not bring up the emergency word on your own. If the human asks how to stop the arm, say the emergency word on their screen holds it (it takes a second or two), or they can say to end the run and then yes; at the station, Ctrl-C in the harness window ends the run. There is no physical e-stop. Keep confirmation questions short. Never say the emergency word yourself, and never start a sentence with stop or cut; say things like \"Done, the arm is holding\" instead of \"Okay, stop sent\".
    - Never open a reply with a yes-word (Sure, Yes, Yeah, Okay, Go ahead), because the human's own yes must not be confused with your words.
    - To move the arm: first call the matching propose tool (new or changed task, or go home). These do not move anything.
    - After proposing, say what you are about to do and ask for a clear spoken yes, then END your reply and wait. Never call confirm_pending_action in the same reply as a propose tool.
    - Only after the human's NEXT message is a clear yes, call confirm_pending_action. Code checks this and refuses otherwise.
    - If the human says no or changes their mind, call cancel_pending_action.
    - When proposing to go home, warn that once homing starts voice commands do not work until the move finishes.
    - If the human says stop or cut, or sounds alarmed, call stop_robot immediately, with no questions. The word wait is not a stop word. If the human says the emergency word anywhere, or asks the arm to hold still or stop moving, in any words, call stop_robot immediately; a repeat is harmless. It joins any stop already in flight and reports the real outcome. Only say the arms are holding when that tool or the status line says so.
    - If stop_robot reports it was not confirmed, say so plainly in one short sentence (the system is already retrying and speaks the outcome itself). Never tell the human to press a button; there is no physical e-stop.
    - If a tool says it was a dry run, say plainly that the arm did not move. That is normal for a dry run (the station is not live): do not suggest anything is wrong or offer to check the status because of it.
    - Pass the human's task to propose_task word for word as they said it. Drop only a leading label (such as \"Object stacking:\") and filler words. Never shorten, merge, reorder or paraphrase it: the robot's model responds to exact wording. If they gave the task over several turns, join their words in order.
    - If no model is configured, say so; never invent one. Never claim the arm did something a tool did not confirm.
    - Offer to check robot status when useful.
    - A system message at the end of the conversation, "Station status (...)", is the live robot state. For "how is it going" or "what is the robot doing", answer from it without a tool call, in plain words. If it says no fresh status or UNREACHABLE, say you cannot see the station right now. Call get_robot_status only to force a fresh read.
    - To change the task, always use propose_task, even while a run is active: the tool ends the run first and the arms hold still. Then ask for a spoken yes to start the new task; nothing moves until then.

    # Camera tools (view only)

    - look_at_table, count_golf_balls, check_tower, check_cables, and check_progress only look at camera snapshots. They never move or command the arm and need no confirmation.
    - Report only what the tool says. If it says it cannot tell or the camera is unavailable, say so plainly; never invent counts, heights, or ports. Keep camera answers to two short sentences.
    - If asked what you can do: say you can look at the table and answer questions about it, robot moves need a spoken confirmation first, and the emergency word on your screen holds the arms. Never say the emergency word when answering this; refer to the emergency word on your screen.
    - If the human asks what you can see, what is on the table, or anything about the table, call look_at_table right away with their question. Never ask what to look for first, and never answer it from the station status: the cameras work even when the status is old.
    - Say nothing before calling a camera tool: the tool itself says \"Let me look.\" No other filler.
    - If a camera is unavailable, say so once and stop there. Never promise to tell the human later, to notify them, or to keep checking: you cannot speak unprompted.
    """
)


_CAMERA_SECTION = INSTRUCTIONS[INSTRUCTIONS.index("# Camera tools"):]

HARNESS_INSTRUCTIONS = textwrap.dedent(
    """\
    You are a friendly robot-arm operator at a creative hackathon demo. You talk with a human and run a robot hill climb: a program on the station drives two arms through a timed match by itself. The human may also call it the relay, the run or the match.

    # Output rules

    - Voice only: plain spoken sentences. No markdown, lists, code, or emojis.
    - Keep replies to one or two short sentences. Ask one question at a time.
    - Do not read out tool names, parameters, or raw data. Summarize status in plain words.

    # The three commands

    - Start the hill climb: call propose_relay_start. It does not start anything.
    - End the hill climb: call propose_end_run. It does not end anything. Ending sends the arms home.
    - Hold the arms: call freeze_robot. Safe at any time, never moves the arms; a repeat is harmless.

    # Safety and confirmation (strict)

    - Do not bring up the emergency word on your own. If the human asks how to stop the arms, say the emergency word on their screen holds them (it takes a second or two), or they can say to end the run and then yes; at the station, Ctrl-C in the harness window ends the run. There is no physical e-stop. Keep confirmation questions short.
    - Never say the emergency words yourself, and never repeat them if the human says them. Refer to \"the emergency word on your screen\". Say \"the arms are holding\" rather than naming the word.
    - Never open a reply with a yes-word (Sure, Yes, Yeah, Okay, Go ahead), because the human's own yes must not be confused with your words.
    - After a propose tool, say what you are about to do and ask for a clear spoken yes, then END your reply and wait. Never call confirm_pending_action in the same reply as a propose tool.
    - Only after the human's NEXT message is a clear yes, call confirm_pending_action. Code checks this and refuses otherwise.
    - If the human says no or changes their mind, call cancel_pending_action.
    - If the human asks the arms to hold still, stop moving, or not move, in any words, call freeze_robot immediately, with no questions. If the human says the emergency word anywhere, call freeze_robot too. It joins any stop already in flight and reports the real outcome. Only say the arms are holding when that tool or the status line says so. When the tool says the run is over, say the run is over and the arms are parked and switched off, not holding.
    - If the human asks to end or stop the hill climb (or the relay, the run or the match), or says we're done, call propose_end_run (then a clear yes is needed). If they sound alarmed, call freeze_robot first.
    - If freeze_robot reports it was not confirmed, or says it was already handled, say nothing more or one short plain sentence: the system retries and speaks the outcome itself. Never tell the human to press a button; there is no physical e-stop. If a tool names another program holding the arms, say so and that you cannot stop it by voice.
    - If a tool says it was a dry run, say plainly that the arms did not move. That is normal for a dry run: do not suggest anything is wrong because of it. If a tool says the harness refused, say so and why.
    - Never claim the hill climb did something a tool did not confirm.
    - After you report that the arms are holding, say nothing more about starting. Never ask whether the human is ready to get started or offer to start; wait for them to ask.
    - Never promise to notify the human later.

    # Match workflow

    - Before the match: offer to check the cameras and status. When the human is ready, propose the hill climb start.
    - Once the match is running: you may report scoring and progress when asked. Do not start or change anything by yourself.
    - To finish a match, the human says the emergency word on their screen first so the arms hold, the referee scores the table, and only then they ask to end the run and give a clear yes. Ending runs the home move and park, and that can knock over a standing tower.
    - If the human asks to end the run while the status does not say the arms are holding, suggest the emergency word first, and explain the home move and park can knock over the tower. If they insist, you may still call propose_end_run.
    - A system message at the end of the conversation, "Station status (...)", is the live state. Answer \"how is it going\" from it without a tool call. If it says no fresh status or UNREACHABLE, say you cannot see the station right now. Call get_robot_status only to force a fresh read.
    - Match mode: once the hill climb is running you are silent unless the human asks how it is going, asks to end the run, or says the emergency word. Answer a status question in one short sentence, nothing else. Never comment on the match, never chat, never offer anything.

    """
) + _CAMERA_SECTION


class BaseAssistant(Agent):
    def __init__(
        self,
        gate: RobotGate | None = None,
        camera: CameraFn | None = None,
        vision_fn: VisionFn | None = None,
        monitor: StationMonitor | None = None,
        instructions: str = INSTRUCTIONS,
    ) -> None:
        super().__init__(
            llm=_make_llm(),
            instructions=instructions,
        )
        # Code-owned pending-action state: the model can only propose; execution is gated in code.
        self.gate = gate or RobotGate(StationClient())
        # View-only: these never touch the gate or the station. Injectable for tests.
        self._camera: CameraFn = camera or vision.get_blurred_jpeg
        self._vision: VisionFn = vision_fn or ask_vision_llm
        self.monitor = monitor

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Give each LLM call the live station status, in a copy of the context so history never stores it.

        A user message the fast path already froze for gets no LLM turn: one utterance, one freeze, one spoken outcome.
        """
        items = chat_ctx.items
        last = items[-1] if items else None
        if last is not None and last.type == "message" and last.role == "user":
            if self.gate.fast_owned(last.text_content or "", getattr(last, "id", None)) is not None:
                return None
        if self.monitor is not None:
            chat_ctx = chat_ctx.copy()  # shares the messages, never changes the caller's list
            status = llm.ChatMessage(role="system", content=[self.monitor.describe()])
            # Right after the instructions (as Portal's voice agent does), so the request folds it into the system
            # prompt: some model chat templates (Gemma's among them) reject a system message after the first turn.
            first = next((i for i, item in enumerate(chat_ctx.items)
                          if item.type == "message" and item.role == "system"), None)
            chat_ctx.items.insert(0 if first is None else first + 1, status)
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    def tts_node(self, text, model_settings):
        """Echo scrubber: never speak a freeze word (the mic fast path would hear it)."""
        return Agent.default.tts_node(self, scrub_trigger_words(text), model_settings)

    def transcription_node(self, text, model_settings):
        return Agent.default.transcription_node(self, scrub_transcription(text), model_settings)

    async def _look(self, context, roles: tuple[str, ...], prompt: str) -> str:
        try:
            jpegs = list(await asyncio.gather(*[self._camera(r) for r in roles]))
        except vision.CameraDown as exc:
            logger.warning("camera unavailable: %s", exc)
            return vision.down_message() + " Say exactly that, in one line."
        try:
            context.session.say("Let me look.")  # interruptible filler for the slow vision call
        except Exception:
            logger.debug("could not say filler", exc_info=True)
        try:
            answer = await asyncio.wait_for(self._vision(jpegs, prompt), VISION_TIMEOUT_S)
        except Exception:
            logger.exception("vision model call failed")
            return "I couldn't get an answer from the vision model. Say you can't tell."
        if not answer:
            return "The vision model gave no answer. Say you can't tell."
        # Explicit instruction: in the first live test the LLM once dropped a bare answer and said "Of course."
        return f"What the camera shows (tell the human this now, briefly, in your own words): {answer}"

    @function_tool
    async def look_at_table(self, context: RunContext, question: str) -> str:
        """Look at the overhead camera and answer a question about the table. View only, never moves the arm.

        Args:
            question: What the human wants to know about the table, in plain words.
        """
        return await self._look(context, ("overhead",), PROMPTS["look_at_table"].format(question=question))

    @function_tool
    async def count_golf_balls(self, context: RunContext) -> str:
        """Count the golf balls visible on the table in the overhead camera. View only."""
        return await self._look(context, ("overhead",), PROMPTS["count_golf_balls"])

    @function_tool
    async def check_tower(self, context: RunContext) -> str:
        """Check whether a tower of blocks is standing and roughly how tall it is in inches. View only."""
        return await self._look(context, ("overhead",), PROMPTS["check_tower"])

    @function_tool
    async def check_cables(self, context: RunContext) -> str:
        """Check which hub ports still have cables plugged in, using all three cameras. View only."""
        return await self._look(context, vision.ROLES, PROMPTS["check_cables"])

    @function_tool
    async def check_progress(self, context: RunContext) -> str:
        """Report hill climb progress from the overhead camera: cables plugged in, tower standing and height, golf balls in the bin versus on the table, black ball in the bin. View only."""
        return await self._look(context, ("overhead",), PROMPTS["check_progress"])

    @function_tool
    async def get_robot_status(self) -> str:
        """Read the robot station's current state (idle, running, homing, error), task and whether the arms respond. Safe, read-only."""
        return await self.gate.status()

    @function_tool
    async def confirm_pending_action(self, context: RunContext) -> str:
        """Execute the one pending proposal. Call ONLY in a reply to the human's NEXT message after the proposal, and only if that message is a clear yes. Never in the same reply as a propose tool."""
        session = context.session
        # Never disallow interruptions here: LiveKit then discards incoming audio (discard_audio_if_uninterruptible),
        # so the emergency word said during this reply would go unheard. The gate's re-send is shielded instead.
        self.gate.on_resend_failed = lambda: self.gate.alarm_due() and speak_alarm(session)
        text = await self.gate.confirm(latest_user_turn(session))
        if is_stop_failure(text) and self.gate.alarm_due():  # a fenced re-stop failed
            speak_alarm(session)
        return text

    @function_tool
    async def cancel_pending_action(self) -> str:
        """Cancel the pending proposal, when the human says no or changes their mind."""
        return self.gate.cancel()


class StationAssistant(BaseAssistant):
    """Station (/start, /home, /stop) mode."""

    @function_tool
    async def propose_task(self, context: RunContext, prompt: str) -> str:
        """Propose a new or changed task for the robot arm. If a run is active it is ended first and the arms hold still; the new task starts only after a spoken yes. Afterwards ask the human for a spoken yes and wait for their next message.

        Args:
            prompt: The human's task word for word as they said it, for example "pick up the red block". Drop only a leading label (such as "Object stacking:") and filler words; never shorten, merge or paraphrase it, because the robot's model responds to exact wording. Join a task given over several turns in order.
        """
        text = await self.gate.request_task(prompt, latest_user_turn(context.session))
        if is_stop_failure(text):  # ending the run failed
            speak_alarm(context.session)
        return text

    @function_tool
    async def propose_go_home(self, context: RunContext) -> str:
        """Propose sending the arms to their home pose (only valid when no task is running). Does NOT move the arm. Ask for a spoken yes and wait for their next message."""
        return self.gate.propose_home(latest_user_turn(context.session))

    @function_tool
    async def stop_robot(self, context: RunContext) -> str:
        """Stop the robot now. No confirmation needed. Call at once if the human says stop or cut, or sounds alarmed."""
        return await _stop_with_alarm(self.gate, context.session)


class HarnessAssistant(BaseAssistant):
    """Harness hill climb mode: start/end through the confirm gate, freeze never gated."""

    def __init__(
        self,
        gate: RobotGate,
        camera: CameraFn | None = None,
        vision_fn: VisionFn | None = None,
        monitor: StationMonitor | None = None,
    ) -> None:
        super().__init__(gate, camera, vision_fn, monitor, instructions=HARNESS_INSTRUCTIONS)

    @function_tool
    async def propose_relay_start(self, context: RunContext) -> str:
        """Propose starting the hill climb (the arms begin the match by themselves). Does NOT start anything. Ask for a spoken yes and wait for their next message."""
        return self.gate.propose_relay_start(latest_user_turn(context.session))

    @function_tool
    async def propose_end_run(self, context: RunContext) -> str:
        """Propose ending the hill climb run (the arms go home). Does NOT end anything. Ask for a spoken yes and wait for their next message."""
        return self.gate.propose_end(latest_user_turn(context.session))

    @function_tool
    async def freeze_robot(self, context: RunContext) -> str:
        """Hold the arms where they are, now. No confirmation. Call at once if the human asks the arms to hold still, stop moving or not move, or sounds alarmed. Never moves the arms; a repeat is harmless."""
        owned = None
        turn = latest_user_turn(context.session)
        if turn is not None:
            owned = self.gate.fast_owned(turn.text, turn.id)
        if owned is not None:  # the fast path already froze for this very utterance and spoke the outcome
            try:
                _, text = await asyncio.shield(owned)
            except Exception:  # noqa: BLE001
                text = ESTOP_NOW
            return f"Already handled by the emergency word, and the outcome was already said aloud: {text} Say nothing more."
        if self.gate.match.active:
            if turn is None or not FREEZE_WORD_RE.search(turn.text):
                # Match mode: a bystander's "hold on" or "stop" must not hold the run. Only the emergency word does.
                return "Not done: in match mode only the emergency word holds the arms. Say nothing."
        return await _stop_with_alarm(self.gate, context.session)

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Match mode: stay silent unless the human asks for status, an end, or says the emergency word."""
        match = self.gate.match
        if not match.active:
            return await super().llm_node(chat_ctx, tools, model_settings)
        items = chat_ctx.items
        last = items[-1] if items else None
        if last is not None and last.type == "function_call_output":
            return await super().llm_node(chat_ctx, tools, model_settings)  # the follow-up to a tool we allowed
        user = next((i for i in reversed(items) if i.type == "message" and i.role == "user"), None)
        text = (user.text_content or "") if user is not None else ""
        if FREEZE_WORD_RE.search(text):
            allowed = {"freeze_robot", "propose_end_run", "confirm_pending_action", "cancel_pending_action"}
        elif self.gate.pending is not None or END_INTENT_RE.search(text):
            allowed = {"propose_end_run", "confirm_pending_action", "cancel_pending_action"}
        elif STATUS_QUESTION_RE.search(text):
            ctx = chat_ctx.copy()
            ctx.add_message(role="system", content=[MATCH_STATUS_NOTE])
            return await super().llm_node(ctx, [], model_settings)
        else:
            return None  # silent
        return await super().llm_node(chat_ctx, [t for t in tools if getattr(t, "id", None) in allowed], model_settings)


Assistant = StationAssistant


HOLDING_SPEECH = "Holding."
END_INTENT_RE = re.compile(
    r"\b(end|stop|finish|quit|wrap up|shut down)\b.*\b(run|relay|hill climb|match|round|session|it)\b|\b(we'?re|all) done\b|\bthat'?s it\b",
    re.IGNORECASE,
)
STATUS_QUESTION_RE = re.compile(
    r"\b(how('?s| is) (it|the (hill climb|relay|run|match)) going|status|score|scoring|what('?s| is| are) (the )?(robot|arms?|it|we|you) doing|"
    r"are (we|the arms) (holding|running|moving)|is it (running|holding|done|over)|what stage|which stage|"
    r"how (long|much time))\b",
    re.IGNORECASE,
)
MATCH_STATUS_NOTE = (
    "Match mode. Answer the human's status question in one short plain sentence from the station status line. "
    "No questions, no offers, no commentary, and never use the emergency word."
)


def _speak_if_stop_failed(session: AgentSession, gate: RobotGate | None = None):
    def _cb(task) -> None:
        if task.cancelled() or task.exception() is not None or not task.result()[0]:
            if gate is None or gate.alarm_due():
                failed = None if task.cancelled() or task.exception() is not None else task.result()[1]
                speak_alarm(session, interrupt=True, text=failed or STOP_FAILED_SPEECH)
            return
        ok, text = task.result()
        try:  # fast path has no LLM turn: speak a fixed code-owned line
            say_with_retry(session, HOLDING_SPEECH if text == SENT_TEXT else text)
        except Exception:
            logger.exception("could not speak the holding line")

    return _cb


async def _stop_with_alarm(gate: RobotGate, session) -> str:
    """Run the stop as its own task (held in gate._bg) so an interrupted tool call still speaks the outcome."""
    task = asyncio.create_task(gate.stop_result())
    gate._bg.add(task)
    task.add_done_callback(gate._bg.discard)
    state = {"interrupted": False}

    def _done(t) -> None:
        if t.cancelled() or t.exception() is not None or not t.result()[0]:
            if gate.alarm_due():
                failed = None if t.cancelled() or t.exception() is not None else t.result()[1]
                speak_alarm(session, text=failed or STOP_FAILED_SPEECH)
        elif state["interrupted"]:  # nobody is left to read the tool result aloud
            text = t.result()[1]
            try:
                say_with_retry(session, HOLDING_SPEECH if text == SENT_TEXT else text)
            except Exception:
                logger.exception("could not speak the holding line")

    task.add_done_callback(_done)
    try:
        return (await asyncio.shield(task))[1]
    except asyncio.CancelledError:
        state["interrupted"] = True
        raise


def build_backend(env) -> tuple[RobotGate, type[BaseAssistant], StationMonitor]:
    """(gate, assistant class, monitor) for ROBOT_BACKEND: station (default), harness (ssh) or fake (local helper).

    Constructs only; nothing connects or starts until the session does.
    """
    backend = env.get("ROBOT_BACKEND") or "station"
    if backend == "station":
        gate = RobotGate(StationClient())
        return gate, StationAssistant, StationMonitor(gate.client)
    if backend not in ("harness", "fake"):
        raise ValueError(f"unknown ROBOT_BACKEND {backend!r} (station, harness or fake)")
    live = env.get("HARNESS_LIVE") == "1"
    if backend == "fake":
        # LocalRunner can only reach the local fake harness, so it is safe to run it "live".
        if not (env.get("HARNESS_REMOTE_HOME") and env.get("HARNESS_FAKE_CMD")):
            raise ValueError("ROBOT_BACKEND=fake needs HARNESS_REMOTE_HOME and HARNESS_FAKE_CMD (use ops/voice-agent.sh --backend fake)")
        live = True
    runner = SshRunner(env.get("HARNESS_HOST", "nuck-3")) if backend == "harness" else LocalRunner()
    client = HarnessClient(
        runner,
        live=live,
        harness_dir=env.get("HARNESS_DIR", "~/team-harness-voice"),
        calibration=env.get("HARNESS_CALIBRATION", "~/team-harness/harness/calibration.yaml"),
        checkpoint=env.get("DREAM_CHECKPOINT") or None,  # none: the harness's per-stage default models
        run_args=env.get("HARNESS_RUN_ARGS", ""),
    )
    gate = RobotGate(client)
    gate.match = MatchMode(enabled=env.get("MATCH_MODE", "1") != "0")
    monitor = HarnessMonitor(client, match=gate.match)
    gate.fresh_status = monitor.fresh_status
    gate.on_pending = monitor.arm_pending_deadline
    gate.on_settled = monitor.settle_pending
    monitor.retry_freeze = gate.stop_result  # an unconfirmed hold at the deadline triggers the same retry
    return gate, HarnessAssistant, monitor


def _say_line(session):
    def _say(text: str) -> None:
        # The monitored speech path of the fast-path alarms: re-speaks once if the human talks over it, never mutes the
        # mic. May raise if the session is closed (the monitor catches and tries again on a later poll).
        say_with_retry(session, text)

    return _say


STATION_GREETING = (
    "Greet the user in one short sentence as the robot arm operator and ask what they would like the arm to do. "
    "Do not mention safety in the greeting."
)
HARNESS_GREETING = "Hi there, I'm the operator for the hill climb."  # said word for word (session.say), no LLM


def agent_name(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("AGENT_NAME") or f"pah2026-robot-{socket.gethostname()}"


def greeting_instructions(assistant_cls, env=None) -> str | None:
    """LLM instructions for the station greeting; None for the hill climb, which says HARNESS_GREETING word for word.
    (The launcher's banner still prints the backend and whether it is live, a dry run or a local test.)"""
    return None if assistant_cls is HarnessAssistant else STATION_GREETING


server = AgentServer()


@server.rtc_session(agent_name=agent_name())
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        # Keyterms bias the STT toward distinctive words it would otherwise misspell.
        # List your own names, brands, and jargon in `keyterms`. Detection additionally
        # extracts terms from the live conversation, such as a caller's name, and applies
        # them once the transcript corroborates the spelling.
        # See more at https://docs.livekit.io/agents/models/stt/keyterms/
        stt_context_options=STTContextOptions(
            keyterms=KEYTERMS,
            keyterm_detection={"enabled": True},
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"),
        turn_handling=TurnHandlingOptions(
            # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
            # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
            # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
            # AgentSession supplies the required VAD automatically.
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=inference.TurnDetector(),
            # Adaptive interruptions use the turn detector to tell a real interruption from a
            # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
            # This model is on by default on LiveKit Cloud and in dev mode. To use the model elsewhere,
            # uncomment the `interruption` line. Self-hosted usage past the monthly allowance is billed.
            # See https://docs.livekit.io/deploy/admin/quotas-and-limits/#adaptive-interruption-handling for more information.
            # interruption={"mode": "adaptive"},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=True,
    )

    gate, assistant_cls, monitor = build_backend(os.environ)
    if isinstance(monitor, HarnessMonitor):
        monitor.say = _say_line(session)
    gate.on_retry = lambda: say_with_retry(session, RETRYING_SPEECH)
    monitor.start()
    ctx.add_shutdown_callback(monitor.aclose)
    _warm = asyncio.create_task(vision.warmup())  # keep a reference; loads the face blurrer off-loop

    # Fast path that skips the LLM: "freeze" or "halt" among the first two words of a FINAL transcript (never interim, negation-guarded) fires /stop right away.
    # Event name and fields (transcript, is_final) verified in livekit/agents/voice/events.py (1.8.10).
    session.on("user_input_transcribed", make_transcript_handler(gate, session))
    # The confirm TTL counts from when the agent has finished asking (AgentStateChangedEvent, events.py 1.8.10).
    session.on("agent_state_changed", make_state_handler(gate))

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=assistant_cls(gate, monitor=monitor),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
            ),
        ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()

    # Speak first: greet the user before they say anything
    instructions = greeting_instructions(assistant_cls)
    if instructions is None:
        session.say(HARNESS_GREETING)  # exactly this line
    else:
        await session.generate_reply(instructions=instructions)


if __name__ == "__main__":
    cli.run_app(server)
