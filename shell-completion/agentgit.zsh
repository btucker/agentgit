#compdef agentgit agit

# Zsh completion for agentgit and agit
# This delegates to git's existing completion for git passthrough commands

_agentgit() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    # If we're at the first argument position, show agentgit + git commands
    if (( CURRENT == 2 )); then
        local agentgit_commands=(
            'process:Process agent transcripts into git repositories'
            'sessions:Manage and list web sessions'
            'discover:Discover transcripts for the current project'
            'repo:Print path to agentgit repository'
            'completion:Output shell completion script'
        )
        local git_commands
        git_commands=(${(f)"$(git help -a 2>/dev/null | awk '/^  [a-z]/ {print $1}')"})

        _describe -t agentgit-commands 'agentgit commands' agentgit_commands
        _describe -t git-commands 'git commands' git_commands
        return 0
    fi

    # For positions after the first argument, provide git-aware completions
    # Find the agentgit repo path using repo ID
    local agentgit_repo=""
    local git_root=$(git rev-parse --show-toplevel 2>/dev/null)

    if [[ -n "$git_root" ]]; then
        local repo_id=$(git -C "$git_root" rev-list --max-parents=0 HEAD 2>/dev/null | head -1 | cut -c1-12)
        if [[ -n "$repo_id" ]]; then
            agentgit_repo="$HOME/.agentgit/projects/$repo_id"
        fi
    fi

    # If repo exists, provide git-aware completions
    if [[ -d "$agentgit_repo/.git" ]]; then
        local -a branches
        branches=(${(f)"$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ refs/remotes/ refs/tags/ 2>/dev/null)"})
        _describe -t branches 'git branches' branches
    else
        _files
    fi
}

# Manually register the completion (needed when sourcing directly)
compdef _agentgit agit
compdef _agentgit agentgit
