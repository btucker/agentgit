"""File watcher for agentgit incremental updates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from watchdog.observers import Observer

from agentgit import parse_transcript
from agentgit.git_builder import GitRepoBuilder


class TranscriptWatcher:
    """Watches transcript files and triggers incremental builds."""

    def __init__(
        self,
        transcript_path: Path,
        output_dir: Path,
        author_name: str = "Agent",
        author_email: str = "agent@local",
        source_repo: Path | None = None,
        on_update: Callable[[int], None] | None = None,
    ):
        """Initialize the watcher.

        Args:
            transcript_path: Path to the transcript file to watch.
            output_dir: Directory for the git repo.
            author_name: Name for git commits.
            author_email: Email for git commits.
            source_repo: Optional source repository to interleave commits from.
            on_update: Optional callback called with number of new commits after each update.
        """
        self.transcript_path = transcript_path
        self.output_dir = output_dir
        self.author_name = author_name
        self.author_email = author_email
        self.source_repo = source_repo
        self.on_update = on_update
        self._observer: Observer | None = None
        self._last_mtime: float = 0

    def _do_incremental_build(self) -> int:
        """Perform an incremental build and return number of new commits."""
        try:
            transcript = parse_transcript(self.transcript_path)

            builder = GitRepoBuilder(output_dir=self.output_dir)

            # Get commit count before
            from git import Repo
            from git.exc import InvalidGitRepositoryError

            try:
                repo = Repo(self.output_dir)
                commits_before = len(list(repo.iter_commits()))
            except InvalidGitRepositoryError:
                commits_before = 0

            repo, _, _ = builder.build(
                operations=transcript.operations,
                author_name=self.author_name,
                author_email=self.author_email,
                source_repo=self.source_repo,
                incremental=True,
            )

            commits_after = len(list(repo.iter_commits()))
            return commits_after - commits_before

        except Exception as e:
            # Log error but don't crash the watcher
            import sys

            print(f"Error during incremental build: {e}", file=sys.stderr)
            return 0

    def _handle_change(self) -> None:
        """Handle a file change event."""
        # Check if file actually changed (debounce)
        try:
            current_mtime = self.transcript_path.stat().st_mtime
            if current_mtime <= self._last_mtime:
                return
            self._last_mtime = current_mtime
        except FileNotFoundError:
            return

        new_commits = self._do_incremental_build()

        if self.on_update and new_commits > 0:
            self.on_update(new_commits)

    def start(self) -> None:
        """Start watching for changes."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as e:
            raise ImportError(
                "watchdog is required for watch mode. "
                "Install with: pip install agentgit[watch]"
            ) from e

        # Do initial build
        self._last_mtime = self.transcript_path.stat().st_mtime
        self._do_incremental_build()

        # Set up file watcher
        class Handler(FileSystemEventHandler):
            def __init__(handler_self, watcher: TranscriptWatcher):
                handler_self.watcher = watcher

            def on_modified(handler_self, event) -> None:
                if event.is_directory:
                    return
                if Path(event.src_path) == handler_self.watcher.transcript_path:
                    handler_self.watcher._handle_change()

            def on_created(handler_self, event) -> None:
                if event.is_directory:
                    return
                if Path(event.src_path) == handler_self.watcher.transcript_path:
                    handler_self.watcher._handle_change()

        self._observer = Observer()
        handler = Handler(self)

        # Watch the directory containing the transcript
        watch_dir = str(self.transcript_path.parent)
        self._observer.schedule(handler, watch_dir, recursive=False)
        self._observer.start()

    def stop(self) -> None:
        """Stop watching for changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def run_forever(self) -> None:
        """Run the watcher until interrupted."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def watch_transcript(
    transcript_path: Path,
    output_dir: Path,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    source_repo: Path | None = None,
    on_update: Callable[[int], None] | None = None,
) -> TranscriptWatcher:
    """Create and start a transcript watcher.

    Args:
        transcript_path: Path to the transcript file to watch.
        output_dir: Directory for the git repo.
        author_name: Name for git commits.
        author_email: Email for git commits.
        source_repo: Optional source repository to interleave commits from.
        on_update: Optional callback called with number of new commits after each update.

    Returns:
        A running TranscriptWatcher instance.
    """
    watcher = TranscriptWatcher(
        transcript_path=transcript_path,
        output_dir=output_dir,
        author_name=author_name,
        author_email=author_email,
        source_repo=source_repo,
        on_update=on_update,
    )
    watcher.start()
    return watcher
