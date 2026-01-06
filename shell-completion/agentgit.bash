# Bash completion for agentgit
# This delegates to git's existing completion for git passthrough commands

# Wrapper function that sets up git context for agentgit repo
_agentgit_git_wrapper() {
    # Find the agentgit repo path
    local agentgit_repo=""
    local project_name=""
    local git_root=$(git rev-parse --show-toplevel 2>/dev/null)

    if [ -n "$git_root" ]; then
        # Get repo ID (first 12 chars of root commit)
        local repo_id=$(git -C "$git_root" rev-list --max-parents=0 HEAD 2>/dev/null | head -1 | cut -c1-12)
        if [ -n "$repo_id" ]; then
            agentgit_repo="$HOME/.agentgit/projects/$repo_id"
        fi
    fi

    # If agentgit repo exists, provide completions
    if [ -d "$agentgit_repo/.git" ]; then
        # Check if git completion is available
        if declare -f __git_main &>/dev/null; then
            # Use git's native completion
            local saved_pwd="$PWD"
            local saved_git_dir="$__git_dir"

            cd "$agentgit_repo" 2>/dev/null
            __git_dir="$agentgit_repo/.git"

            __git_main

            __git_dir="$saved_git_dir"
            cd "$saved_pwd" 2>/dev/null
        else
            # Fallback: provide basic branch/ref completion
            local git_cmd="${words[1]}"
            case "$git_cmd" in
                log|show|diff|checkout|switch|rebase|merge|cherry-pick)
                    # Complete with branches and refs
                    local refs=$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ refs/remotes/ refs/tags/ 2>/dev/null)
                    COMPREPLY=( $(compgen -W "$refs" -- "$cur") )
                    ;;
                branch)
                    local branches=$(git -C "$agentgit_repo" for-each-ref --format='%(refname:short)' refs/heads/ 2>/dev/null)
                    COMPREPLY=( $(compgen -W "$branches" -- "$cur") )
                    ;;
                *)
                    # Default to file completion
                    _filedir
                    ;;
            esac
        fi
    fi
}

_agentgit() {
    local cur prev words cword
    _init_completion || return

    # Known agentgit commands
    local agentgit_commands="process sessions discover"

    # If we're completing the first argument
    if [ $cword -eq 1 ]; then
        # Offer both agentgit commands and git commands
        local git_commands=$(git help -a 2>/dev/null | awk '/^  [a-z]/ {print $1}')
        COMPREPLY=( $(compgen -W "$agentgit_commands $git_commands" -- "$cur") )
        return 0
    fi

    # If the first argument is an agentgit command, use default completion
    case "${words[1]}" in
        process|sessions|discover)
            # Use default file completion
            _filedir
            return 0
            ;;
        *)
            # For git passthrough commands, use wrapper that sets git context
            # Replace command name with git in the completion arrays
            local saved_word0="${words[0]}"
            local saved_comp_word0="${COMP_WORDS[0]}"

            words[0]="git"
            COMP_WORDS[0]="git"

            # Call our wrapper that handles the agentgit repo context
            _agentgit_git_wrapper

            # Restore command name
            COMP_WORDS[0]="$saved_comp_word0"
            words[0]="$saved_word0"
            return 0
            ;;
    esac
}

complete -F _agentgit agentgit
complete -F _agentgit agit
