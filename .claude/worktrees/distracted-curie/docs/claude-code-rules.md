# Claude Code Development Rules — MRG LongPlay Studio V5.5

## MANDATORY: Read Before Every Story

Before implementing ANY story, Claude Code MUST:
1. Read `docs/prd.md` — understand full project context
2. Read `docs/brief.md` — understand vision and tech philosophy
3. Read this file — follow all rules below
4. Read the specific story's acceptance criteria
5. Check dependencies — ensure prerequisite stories are complete

---

## Story Implementation Workflow

### Step 1: Create Feature Branch
```bash
git checkout main && git pull
git checkout -b story/{epic}.{story}-{short-description}
# Example: story/1.1-compile-rust-backend
```

### Step 2: Implement
- Follow Technical Preferences in prd.md
- Rust for DSP/audio, Python for GUI/orchestration
- Keep existing APIs — do NOT break working features

### Step 3: Self-Test (MANDATORY before commit)

Run ALL applicable checks:

```bash
# Python syntax check
python3 -c "
import ast, os
errors = []
for root, dirs, files in os.walk('.'):
    if 'node_modules' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh: ast.parse(fh.read())
            except SyntaxError as e: errors.append(f'{path}: {e}')
print('PASS: All Python files syntax OK' if not errors else '\n'.join(errors))
"

# Python imports check (verify no circular imports)
python3 -c "
import sys; sys.path.insert(0, '.')
try:
    from modules.master import MasterChain, NATIVE_BACKEND
    print(f'PASS: MasterChain imported (backend={\"Rust\" if NATIVE_BACKEND else \"Python\"})')
except Exception as e:
    print(f'FAIL: {e}')
"

# Rust compilation check (if Rust files changed)
cd rust && cargo check 2>&1 | tail -5

# Rust tests (if Rust files changed)
cd rust && cargo test 2>&1 | tail -10

# Python tests (if test files exist)
python3 -m pytest tests/ -v 2>&1 || echo "No tests directory yet"
```

### Step 4: Verify Acceptance Criteria

For each acceptance criterion in the story, verify it explicitly:
```
AC-1: [criterion text] → PASS/FAIL (evidence)
AC-2: [criterion text] → PASS/FAIL (evidence)
...
```

### Step 5: Commit with Verification Report

```bash
git add -A
git commit -m "feat(scope): description

Story X.Y: [Title]
Status: COMPLETE

Acceptance Criteria:
- [x] AC-1: description — verified by [test/manual]
- [x] AC-2: description — verified by [test/manual]

Tests: [passed/added X new tests]
Breaking Changes: [none/description]
"
```

### Step 6: Push for Review

```bash
git push origin story/{epic}.{story}-{short-description}
```

Then tell the user:
> "Story X.Y complete. Push to branch `story/X.Y-name`.
> Ready for QA review in Claude.ai."

---

## QA Review Process (Done in Claude.ai, NOT Claude Code)

The user will open Claude.ai and request BMAD QA review:

```
เรียก BMAD QA Team ตรวจ Story X.Y
Repo: [url]
Branch: story/X.Y-name
ตรวจตาม acceptance criteria ใน docs/prd.md
```

### QA Agent (Quinn) Checklist

#### Code Quality
- [ ] No syntax errors in any Python file
- [ ] No compilation errors in Rust (if applicable)
- [ ] No bare `except:` blocks (must specify exception type)
- [ ] No hardcoded values that should be configurable
- [ ] No TODO/FIXME left unresolved in new code
- [ ] Type hints on all new Python functions
- [ ] Docstrings on all new classes and public methods
- [ ] Rust code passes `cargo clippy` without warnings

#### Functionality
- [ ] All acceptance criteria from PRD verified
- [ ] Existing features still work (no regression)
- [ ] Error handling: invalid inputs don't crash
- [ ] Edge cases: empty files, Unicode paths, large files

#### Architecture
- [ ] Follows 3-layer architecture (Rust engine → Python app → PyQt6 GUI)
- [ ] New code in correct location per refactored structure
- [ ] No circular imports
- [ ] Rust backend and Python fallback both work

#### Audio Quality (for mastering stories)
- [ ] True Peak ≤ ceiling (verify with `ffmpeg -af loudnorm=print_format=json`)
- [ ] LUFS within ±0.5 LU of target
- [ ] No audio artifacts (clipping, clicks, silence gaps)
- [ ] Stereo field preserved (no unexpected mono collapse)

#### UI (for GUI stories)
- [ ] PyQt6 AND PySide6 compatible (try/except imports)
- [ ] Dark theme consistent with Waves/SSL aesthetic
- [ ] No layout breaks at different window sizes
- [ ] All buttons/controls functional (no dead UI)

---

## Definition of Done (Global)

A story is DONE when:
- [ ] All acceptance criteria met and verified
- [ ] Self-tests pass (Step 3 above)
- [ ] Committed with verification report
- [ ] Pushed to feature branch
- [ ] QA review passed in Claude.ai
- [ ] Merged to main (by user)

---

## Critical Rules

### NEVER Do
- Never delete existing working code without replacement
- Never change module API signatures without updating all callers
- Never commit with failing tests
- Never use `except:` (bare) — always specify exception type
- Never hardcode sample rates (read from input file)
- Never hardcode file paths (use os.path / pathlib)
- Never import PyQt6 without PySide6 fallback in new files

### ALWAYS Do
- Always read prd.md before starting a story
- Always create feature branch (never commit directly to main)
- Always run syntax check before committing
- Always verify the Python fallback works when changing Rust code
- Always use conventional commits (feat:, fix:, refactor:, docs:, test:)
- Always preserve backward compatibility with existing project files
