"""URL resolution for fetching remote transcripts and agentgit repos.

Supports:
- HTTP/HTTPS URLs to JSONL transcript files
- Git repository URLs (existing agentgit repos)
- file:// URIs
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

# Regex patterns for URL detection
HTTP_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
FILE_URI_PATTERN = re.compile(r"^file://", re.IGNORECASE)
GIT_URL_PATTERNS = [
    re.compile(r"^git@"),  # git@github.com:user/repo.git
    re.compile(r"^ssh://"),  # ssh://git@...
    re.compile(r"\.git$"),  # https://github.com/user/repo.git
    re.compile(r"^git://"),  # git:// protocol
]


class URLResolverError(Exception):
    """Error resolving or fetching URL content."""

    pass


@dataclass
class ResolvedSource:
    """Result of resolving a transcript source."""

    path: Path
    source_type: Literal["local", "url", "git_repo"]
    original_url: str | None = None
    is_temporary: bool = False
    temp_dir: Path | None = None  # For git repos, the temp dir containing the clone

    def cleanup(self) -> None:
        """Clean up temporary files if any."""
        if self.is_temporary:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            elif self.path.exists() and self.path.is_file():
                self.path.unlink(missing_ok=True)


def is_url(arg: str) -> bool:
    """Check if the argument looks like a URL.

    Args:
        arg: The argument to check.

    Returns:
        True if the argument is a URL (http://, https://, file://, git URL).
    """
    if HTTP_URL_PATTERN.match(arg):
        return True
    if FILE_URI_PATTERN.match(arg):
        return True
    for pattern in GIT_URL_PATTERNS:
        if pattern.search(arg):
            return True
    return False


def is_git_url(url: str) -> bool:
    """Check if the URL points to a git repository.

    Args:
        url: The URL to check.

    Returns:
        True if the URL appears to be a git repository URL.
    """
    # Explicit git protocols
    if url.startswith("git@") or url.startswith("git://") or url.startswith("ssh://"):
        return True

    # .git suffix
    if url.endswith(".git"):
        return True

    # GitHub/GitLab/Bitbucket URLs without .git
    # Pattern: https://github.com/user/repo (exactly 2 path segments)
    git_hosts = ["github.com", "gitlab.com", "bitbucket.org"]
    for host in git_hosts:
        if host in url:
            # Check if it looks like a repo URL (host/user/repo)
            # Not a file URL (host/user/repo/blob/main/file.jsonl)
            path_match = re.search(rf"{host}/([^/]+)/([^/]+)(?:/|$)", url)
            if path_match:
                # If there's more path after user/repo, it might be a file
                remaining = url.split(path_match.group(0))[-1] if path_match.group(0) in url else ""
                # Common file paths in git repos
                if any(x in remaining.lower() for x in ["blob/", "raw/", "tree/", "-/blob", "-/raw"]):
                    return False
                # If no additional path or just trailing slash, it's a repo
                if not remaining or remaining == "/":
                    return True

    return False


def fetch_url(url: str, timeout: float = 60.0) -> bytes:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The response content as bytes.

    Raises:
        URLResolverError: If the fetch fails.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.TimeoutException as e:
        raise URLResolverError(f"Timeout fetching URL: {url}") from e
    except httpx.HTTPStatusError as e:
        raise URLResolverError(
            f"HTTP error {e.response.status_code} fetching URL: {url}"
        ) from e
    except httpx.RequestError as e:
        raise URLResolverError(f"Network error fetching URL: {url} - {e}") from e


def clone_git_repo(url: str, shallow: bool = True) -> tuple[Path, Path]:
    """Clone a git repository to a temporary directory.

    Args:
        url: The git repository URL.
        shallow: If True, do a shallow clone (--depth=1).

    Returns:
        Tuple of (repo_path, temp_dir). temp_dir is the parent that should be cleaned up.

    Raises:
        URLResolverError: If the clone fails.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="agentgit-clone-"))
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "") or "repo"
    repo_path = temp_dir / repo_name

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([url, str(repo_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for clone
        )
        if result.returncode != 0:
            shutil.rmtree(temp_dir)
            raise URLResolverError(f"Git clone failed: {result.stderr}")
        return repo_path, temp_dir
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir)
        raise URLResolverError(f"Git clone timed out: {url}")
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise URLResolverError(f"Git clone error: {e}") from e


def resolve_file_uri(uri: str) -> Path:
    """Convert a file:// URI to a local Path.

    Args:
        uri: The file:// URI.

    Returns:
        Local Path object.

    Raises:
        URLResolverError: If the path doesn't exist.
    """
    # Strip file:// prefix
    path_str = uri[7:]  # Remove 'file://'

    # Handle file:///path (3 slashes = absolute) vs file://host/path
    if path_str.startswith("/"):
        path = Path(path_str)
    else:
        # file://host/path - just use /path part
        path = Path("/" + path_str.split("/", 1)[-1])

    if not path.exists():
        raise URLResolverError(f"File not found: {path}")

    return path


def resolve_transcript_source(arg: str) -> ResolvedSource:
    """Resolve a transcript argument to a local path.

    Handles:
    - Local file paths (returned as-is)
    - HTTP/HTTPS URLs to JSONL files (fetched to temp file)
    - file:// URIs (converted to local path)
    - Git repository URLs (cloned to temp directory)

    Args:
        arg: The transcript argument (path or URL).

    Returns:
        ResolvedSource with path and metadata.

    Raises:
        URLResolverError: If resolution fails.
        FileNotFoundError: If local path doesn't exist.
    """
    # Check if it's a file:// URI
    if FILE_URI_PATTERN.match(arg):
        path = resolve_file_uri(arg)
        return ResolvedSource(
            path=path,
            source_type="local",
            original_url=arg,
            is_temporary=False,
        )

    # Check if it's an HTTP/HTTPS URL
    if HTTP_URL_PATTERN.match(arg):
        # Determine if this is a git repo or a file
        if is_git_url(arg):
            repo_path, temp_dir = clone_git_repo(arg)
            return ResolvedSource(
                path=repo_path,
                source_type="git_repo",
                original_url=arg,
                is_temporary=True,
                temp_dir=temp_dir,
            )
        else:
            # Fetch as file
            content = fetch_url(arg)

            # Determine filename from URL
            url_path = arg.split("?")[0]  # Remove query params
            filename = url_path.rstrip("/").split("/")[-1] or "transcript.jsonl"
            if not filename.endswith(".jsonl"):
                filename += ".jsonl"

            # Write to temp file
            temp_file = Path(tempfile.mktemp(suffix=f"-{filename}", prefix="agentgit-"))
            temp_file.write_bytes(content)

            return ResolvedSource(
                path=temp_file,
                source_type="url",
                original_url=arg,
                is_temporary=True,
            )

    # Check if it's a git URL (git@, ssh://, git://)
    for pattern in GIT_URL_PATTERNS:
        if pattern.search(arg):
            repo_path, temp_dir = clone_git_repo(arg)
            return ResolvedSource(
                path=repo_path,
                source_type="git_repo",
                original_url=arg,
                is_temporary=True,
                temp_dir=temp_dir,
            )

    # Treat as local path
    path = Path(arg)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {arg}")

    return ResolvedSource(
        path=path,
        source_type="local",
        is_temporary=False,
    )


def find_transcripts_in_repo(repo_path: Path) -> list[Path]:
    """Find transcript files in a cloned agentgit or source repo.

    Searches for:
    - .jsonl files that look like transcripts
    - Files in standard locations

    Args:
        repo_path: Path to the cloned repository.

    Returns:
        List of transcript file paths found.
    """
    transcripts = []

    # Look for JSONL files
    for jsonl_file in repo_path.rglob("*.jsonl"):
        # Skip node_modules, .git, etc.
        if any(part.startswith(".") or part == "node_modules" for part in jsonl_file.parts):
            continue
        transcripts.append(jsonl_file)

    # Sort by modification time (most recent first)
    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return transcripts
