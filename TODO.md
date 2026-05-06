# TODO

## Goal
Externalize the hardcoded model presets into an editable `endpoints.json` file with an interactive Add/Delete/Rename menu in the CLI.

## Tasks

### 1. Create `endpoints.json` with defaults
- [x] Create `endpoints.json` containing all 12 current presets from `main.py`
- [x] Add URL format validator (must start with `http://` or `https://`)

### 2. Refactor `main.py` preset loading
- [x] Replace the hardcoded `PRESETS` dict with a function that reads `endpoints.json`
- [x] Fall back to embedded defaults if the file is missing or malformed
- [x] Auto-migrate: on first run, if file doesn't exist, write defaults and notify user

### 3. Interactive endpoint editor
- [x] Add `[A] Add`, `[D] Delete`, `[R] Rename` options to the model selection menu
- [x] **Add**: prompt for name, URL (with validation), model → append → save
- [x] **Delete**: prompt for key to remove → compact keys (1, 2, 3...) → save
- [x] **Rename**: prompt for key + new name → save
- [x] Re-render the menu after each edit so user sees changes immediately

### 4. Cleanup
- [x] Remove the now-unused hardcoded `PRESETS` dict from `main.py`
- [x] Add `endpoints.json` to `.gitignore`
- [x] Update `README.md` with endpoint management docs

## Notes
- Flat structure only: one URL per preset (no per-role URLs in the editor)
- Basic URL validation (format only), no connectivity check
- File format: `{"presets": {"1": {"name": "...", "url": "...", "model": "..."}, ...}}`
