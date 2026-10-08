"""Textual TUI Application for Jotter digests, search, inspection, and status."""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    OptionList,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from llm.assistant.search import AssistantSearchRunner
from llm.digest.cache import ActivityCache
from llm.digest.engine import DigestEngine
from llm.digest.schedule import Slot
from llm.settings import load_config
from llm.tui.llm_stream import (
    AnswerToken,
    PromptEvalFinished,
    PromptEvalProgress,
    ReasoningToken,
    ToolCallEvent,
)


class JotterApp(App[None]):
    """Jotter: Executive Assistant & Daily Digest TUI."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #digest-container {
        layout: horizontal;
        height: 1fr;
    }
    #digest-sidebar {
        width: 30;
        border-right: solid $primary;
    }
    #digest-view {
        width: 1fr;
        padding: 1 2;
    }
    #search-box {
        dock: bottom;
        margin: 1;
    }
    #chat-scroll {
        height: 1fr;
        padding: 1 2;
    }
    .status-badge {
        margin: 1;
        padding: 0 1;
    }
    #live-progress {
        margin: 1 2;
    }
    #reasoning-scroll {
        height: 12;
        border: solid $accent;
        padding: 1;
        background: $surface;
    }
    #tool-trace-scroll {
        height: 8;
        border: solid $secondary;
        padding: 1;
    }
    """

    TITLE = "Jotter: Assistant & Daily Digest"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "switch_tab('tab-digests')", "Digests"),
        ("s", "switch_tab('tab-search')", "Search/Ask"),
        ("i", "switch_tab('tab-inspect')", "Inspection"),
        ("c", "switch_tab('tab-status')", "Status"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_config()
        self.cache = ActivityCache(self.cfg.digest.cache_db_path)
        self.engine = DigestEngine(self.cfg)
        self.search_runner = AssistantSearchRunner(self.cfg)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="tabs"):
            # TAB 1: DIGESTS
            with TabPane("Daily Digests", id="tab-digests"), Horizontal(id="digest-container"):
                with Vertical(id="digest-sidebar"):
                    yield Label("[bold]Saved Digests[/bold]")
                    yield OptionList(id="digest-list")
                    yield Button("Run Morning Digest", id="btn-run-morning", variant="primary")
                    yield Button("Run Afternoon Digest", id="btn-run-afternoon", variant="success")
                with VerticalScroll(id="digest-view"):
                    yield Markdown(id="digest-markdown")

            # TAB 2: SEARCH / ASK
            with TabPane("Search & Ask", id="tab-search"):
                with VerticalScroll(id="chat-scroll"):
                    yield Markdown(
                        "### Ask Jotter anything\n"
                        "Search across Discourse, GitHub, Mattermost, Matrix, Google, "
                        "Jira, and ~/dev/craft/ code.",
                        id="chat-markdown",
                    )
                yield Input(placeholder="Ask a question or search for a thread/file...", id="search-input")

            # TAB 3: LIVE INSPECTION
            with TabPane("Live Inspection", id="tab-inspect"):
                yield Label("[bold]Prompt Evaluation Progress[/bold]")
                yield ProgressBar(id="eval-progress-bar", total=100.0, show_eta=True)
                yield Label("Idle", id="eval-speed-label")
                yield Label("\n[bold]Reasoning Stream (<think>):[/bold]")
                with VerticalScroll(id="reasoning-scroll"):
                    yield Static("", id="reasoning-text")
                yield Label("\n[bold]Tool Execution Trace:[/bold]")
                with VerticalScroll(id="tool-trace-scroll"):
                    yield Static("", id="tool-trace-text")

            # TAB 4: SETTINGS & STATUS
            with TabPane("Settings & Status", id="tab-status"), VerticalScroll(id="status-container"):
                yield Label("[bold cyan]Service Connectivity & Configuration[/bold cyan]\n")
                yield Static(id="status-details")
                yield Button("Test Connections & Reload", id="btn-refresh-status", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        self.refresh_digest_list()
        self.refresh_status_tab()

    def refresh_digest_list(self) -> None:
        """Populate the historical digest sidebar."""
        option_list = self.query_one("#digest-list", OptionList)
        option_list.clear_options()
        digests = self.cache.list_digests(limit=20)
        if not digests:
            option_list.add_option("No digests yet")
            return
        for d in digests:
            option_list.add_option(f"{d['date']} ({d['slot']})")

    def refresh_status_tab(self) -> None:
        """Update the connection status badges."""
        status_view = self.query_one("#status-details", Static)
        dc_auth = "Yes" if self.cfg.discourse.is_authenticated() else "No"
        mm_auth = "Yes" if self.cfg.mattermost.is_authenticated() else "No"
        mx_auth = "Yes" if self.cfg.matrix.is_authenticated() else "No"
        jira_auth = "Yes" if self.cfg.jira.is_authenticated() else "No"
        gw_auth = "Yes" if self.cfg.google.is_authenticated() else "No"

        lines = [
            f"[bold]Local LLM Client URL:[/bold] {self.cfg.client_url}",
            f"[bold]Embedding Server URL:[/bold] {self.cfg.embed_url} (enabled: {self.cfg.embed.enabled})",
            "",
            f"[bold]Discourse:[/bold] {self.cfg.discourse.url} (auth: {dc_auth})",
            f"[bold]Mattermost:[/bold] {self.cfg.mattermost.url} (auth: {mm_auth})",
            f"[bold]Matrix:[/bold] {self.cfg.matrix.homeserver} (auth: {mx_auth})",
            f"[bold]Jira:[/bold] {self.cfg.jira.url} (auth: {jira_auth})",
            f"[bold]Google Workspace:[/bold] (auth: {gw_auth})",
            f"[bold]Craft Inspector Directory:[/bold] {self.cfg.digest.craft_dir}",
            f"[bold]Cache DB Path:[/bold] {self.cfg.digest.cache_db_path}",
        ]
        status_view.update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run-morning":
            self.run_background_digest(Slot.MORNING)
        elif event.button.id == "btn-run-afternoon":
            self.run_background_digest(Slot.AFTERNOON)
        elif event.button.id == "btn-refresh-status":
            self.cfg = load_config()
            self.refresh_status_tab()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        opt_text = str(event.option.prompt)
        if " (" in opt_text:
            date, rest = opt_text.split(" (", 1)
            slot = rest.rstrip(")")
            content = self.cache.get_digest(date, slot)
            if content:
                self.query_one("#digest-markdown", Markdown).update(content)

    @work(exclusive=True, thread=True)
    def run_background_digest(self, slot: Slot) -> None:
        """Run digest summarization in background thread."""
        self.call_from_thread(self.notify, f"Generating {slot.value} digest...")
        try:
            summary = self.engine.run_digest(slot=slot)
            self.call_from_thread(self.query_one("#digest-markdown", Markdown).update, summary)
            self.call_from_thread(self.refresh_digest_list)
            self.call_from_thread(self.notify, f"{slot.value.capitalize()} digest completed!")
        except Exception as exc:
            self.call_from_thread(self.notify, f"Error generating digest: {exc}", severity="error")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""
        self.run_search_stream(query)

    @work(exclusive=True)
    async def run_search_stream(self, query: str) -> None:
        """Stream interactive search query results and live tokens."""
        chat_md = self.query_one("#chat-markdown", Markdown)
        reasoning_widget = self.query_one("#reasoning-text", Static)
        tool_widget = self.query_one("#tool-trace-text", Static)
        prog_bar = self.query_one("#eval-progress-bar", ProgressBar)
        speed_lbl = self.query_one("#eval-speed-label", Label)

        reasoning_text = ""
        answer_text = f"**User:** {query}\n\n**Jotter:** "
        await chat_md.update(answer_text)

        async for ev in self.search_runner.search_and_stream(query):
            if isinstance(ev, ToolCallEvent):
                tool_widget.update(f"Tool [{ev.tool_name}] -> {ev.status} ({ev.result or ''})")
            elif isinstance(ev, PromptEvalProgress):
                prog_bar.progress = ev.progress_pct
                status_msg = (
                    f"Evaluating {ev.tokens} prompt tokens: "
                    f"{ev.elapsed_sec:.1f}s / {ev.estimated_total_sec:.1f}s"
                )
                speed_lbl.update(status_msg)
            elif isinstance(ev, PromptEvalFinished):
                prog_bar.progress = 100.0
                speed_lbl.update(
                    f"Evaluated {ev.tokens} tokens in {ev.actual_eval_sec:.2f}s ({ev.actual_tps:.1f} tps)"
                )
            elif isinstance(ev, ReasoningToken):
                reasoning_text += ev.token
                reasoning_widget.update(reasoning_text)
            elif isinstance(ev, AnswerToken):
                answer_text += ev.token
                await chat_md.update(answer_text)


def run_tui() -> None:
    """Launch the Jotter Textual application."""
    app = JotterApp()
    app.run()
