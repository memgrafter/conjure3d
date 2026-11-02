#!/usr/bin/env bash
# Repository scans before a release (run from anywhere; scans this repository's tracked files and git history).
#   bash tools/scan.sh                       # all checks; exit 1 if any check finds something
#   SCAN_WORDS_FILE=private-words.txt bash tools/scan.sh
# 1. secrets: gitleaks over the working tree and over every commit (needs gitleaks on PATH)
# 2. personal and machine data: e-mail addresses, home folders, IPv4 addresses, token and key patterns, and the words
#    in SCAN_WORDS_FILE (one case-insensitive regular expression per line: your names, host names, user names...).
#    Keep that file outside the repository. Commit authors and messages are checked too.
# 3. binary files: lists every tracked file that is not text; images and GLBs are reported with their metadata
#    (exiftool if present)
# 4. safety: rm with globs or variables in shell scripts, downloads piped into a shell, Python run without -I in docs
#    and scripts
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
note() { echo "== $*"; }

note "1. secrets (gitleaks)"
if command -v gitleaks > /dev/null; then
  gitleaks version
  gitleaks dir --no-banner --redact . || fail=1
  gitleaks git --no-banner --redact . || fail=1
else
  echo "gitleaks not found: install it (https://github.com/gitleaks/gitleaks) and run again"; fail=1
fi

note "2. personal and machine data"
FILES=$(git ls-files | grep -v -E '(^|/)bun\.lock$|\.sha256$|^web/package\.json$')
# "c" = case-sensitive, "i" = case-insensitive
PATTERNS=(
  'i [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'                   # e-mail addresses (Python decorator lines skipped)
  'c /(home|Users)/[A-Za-z0-9_.-]+'                                    # home folders
  'c (^|[^=.0-9| ]|[^|] )([0-9]{1,3}\.){3}[0-9]{1,3}([^.0-9]|$)'      # IPv4 addresses (not version pins or table cells)
  'c (ghp|gho|ghu|ghs|github_pat|hf|sk|xox[abp])_[A-Za-z0-9]{16,}'      # common token formats
  'c -----BEGIN [A-Z ]*PRIVATE KEY-----'
  'i (api[_-]?key|secret|password|token)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{8,}'
)
if [ -n "${SCAN_WORDS_FILE:-}" ]; then
  while IFS= read -r w; do [ -n "$w" ] && PATTERNS+=("i $w"); done < "$SCAN_WORDS_FILE"
fi
for entry in "${PATTERNS[@]}"; do
  p=${entry#? }; case_flag=; [ "${entry%% *}" = i ] && case_flag=-i
  # shellcheck disable=SC2086
  hits=$(grep -n $case_flag -E -e "$p" $FILES 2>/dev/null | grep -v -E '@pytest\.|example\.com|github\.com/[A-Za-z0-9_-]+/|@git\+https|^[^:]+:[0-9]+:[[:space:]]*@[A-Za-z_][A-Za-z0-9_.]*\(' || true)
  # history: every commit's message and diff (authors are listed below for a human check)
  hist=$(git log --all -p --format='%s%n%b' | grep $case_flag -E -e "$p" | grep -v -E '@git\+https|github\.com/|@pytest\.|^[+ -]?[[:space:]]*@[A-Za-z_][A-Za-z0-9_.]*\(' | sort -u || true)
  if [ -n "$hits" ] || [ -n "$hist" ]; then echo "pattern: $p"; echo "$hits"; echo "$hist" | head -20; fail=1; fi
done
echo "commit authors and committers (check by hand): $(git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | tr '\n' ' ')"

note "3. binary files"
while IFS= read -r f; do
  if [ -n "$(grep -I -l . "$f" 2>/dev/null)" ] || [ ! -s "$f" ]; then continue; fi
  echo "binary: $f ($(wc -c < "$f" | tr -d ' ') bytes)"
  if command -v exiftool > /dev/null; then exiftool -s -G "$f" | grep -v -E '^\[(File|ExifTool)\]'; fi
done < <(git ls-files)

note "4. safety"
SH=$(git ls-files '*.sh')
# shellcheck disable=SC2086
grep -n -E '(^|[;&|[:space:]])rm[[:space:]][^#]*(\*|\$)' $SH && fail=1
# shellcheck disable=SC2086
grep -n -E '(curl|wget)[^|]*\|[[:space:]]*(ba|z)?sh' $(git ls-files) && fail=1
# shellcheck disable=SC2086
grep -n -E '(^|[[:space:]])(python3?|/bin/python)[[:space:]]+[^-[:space:]][^[:space:]]*\.py' $(git ls-files '*.sh' '*.md' '*.py') && fail=1
[ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOMETHING TO LOOK AT (above)"
exit $fail
