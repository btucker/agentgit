"""Tests for agentgit.utils module."""

import pytest

from agentgit.utils import extract_deleted_paths


class TestExtractDeletedPaths:
    """Tests for extract_deleted_paths function."""

    def test_simple_rm(self):
        """Should extract paths from simple rm commands."""
        paths = extract_deleted_paths("rm file.txt")
        assert paths == ["file.txt"]

    def test_with_flags(self):
        """Should extract paths from rm with flags."""
        paths = extract_deleted_paths("rm -rf /path/to/dir")
        assert paths == ["/path/to/dir"]

    def test_multiple_paths(self):
        """Should extract multiple paths."""
        paths = extract_deleted_paths("rm file1.txt file2.txt")
        assert paths == ["file1.txt", "file2.txt"]

    def test_quoted_paths(self):
        """Should handle quoted paths."""
        paths = extract_deleted_paths('rm "path with spaces/file.txt"')
        assert paths == ["path with spaces/file.txt"]

    def test_single_quoted_paths(self):
        """Should handle single-quoted paths."""
        paths = extract_deleted_paths("rm 'path with spaces/file.txt'")
        assert paths == ["path with spaces/file.txt"]

    def test_non_rm_command(self):
        """Should return empty for non-rm commands."""
        paths = extract_deleted_paths("ls -la")
        assert paths == []

    def test_rm_with_multiple_flags(self):
        """Should handle rm with multiple flags."""
        paths = extract_deleted_paths("rm -r -f /path/to/dir")
        assert paths == ["/path/to/dir"]
