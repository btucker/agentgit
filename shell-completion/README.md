# Shell Completion for agentgit / agit

Shell completion enables tab completion for agentgit (and its alias `agit`), including full git completion for passthrough commands.

## Quick Installation (Recommended)

The easiest way to install completion is using the built-in `completion` command:

### Bash

```bash
# Install for current session
source <(agentgit completion bash)

# Install permanently
echo 'source <(agentgit completion bash)' >> ~/.bashrc
source ~/.bashrc
```

### Zsh

```bash
# Install for current session
source <(agentgit completion zsh)

# Install permanently
echo 'source <(agentgit completion zsh)' >> ~/.zshrc
source ~/.zshrc
```

## Alternative: Manual Installation

If you prefer to use the files in this directory:

### Bash

```bash
# Option 1: Source directly
source /path/to/agentgit/shell-completion/agentgit.bash

# Option 2: Copy to bash completion directory
mkdir -p ~/.bash_completion.d
cp shell-completion/agentgit.bash ~/.bash_completion.d/
echo 'source ~/.bash_completion.d/agentgit.bash' >> ~/.bashrc
```

### Zsh

```bash
# Option 1: Add to fpath (before compinit)
fpath=(/path/to/agentgit/shell-completion $fpath)

# Option 2: Copy to zsh completions directory
cp shell-completion/agentgit.zsh ~/.zsh/completion/_agentgit
```

## Features

- Completes agentgit-specific commands: `process`, `sessions`, `discover`
- Delegates to git's native completion for git passthrough commands
- Supports all git command completion including:
  - Git commands (log, diff, show, etc.)
  - Branch names
  - File paths in the repository
  - Git options and flags
  - Refs, tags, remotes, etc.

## Examples

```bash
agentgit <TAB>          # Shows agentgit commands + git commands
agentgit log <TAB>      # Uses git completion for branch names, refs, etc.
agentgit diff <TAB>     # Uses git completion for files, commits, etc.
agentgit checkout <TAB> # Uses git completion for branches
```
