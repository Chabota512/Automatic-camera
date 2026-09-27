# Target-reference tracking plan

## Goal

When the operator clicks a presenter or object, the application will keep a
session-only reference for that target. OpenCV tracking, AI detections, motion
history, and appearance matching will work together instead of relying on a
fixed pixel region.

The first human feature is **face detection**, not face recognition. The
application will not identify people by name or maintain a biometric identity
database.

## Implementation status

- [x] **Milestone 1 — Session-only target reference.** Implemented in
  `ea0c8d8`, including capture, refresh, clear, and generic/human target
  metadata.
- [x] **Milestone 2 — Hybrid matching and safe camera decisions.** Implemented
  in `ea0c8d8` and the current tracking flow, including appearance gates,
  temporal/disagreement scoring, safe-box validation, tracker reinitialization,
  and diagnostics.
- [x] **Milestone 3 — Human tracking toggle and face detection.** Implemented
  locally in the current working tree; dependency-backed tests and a Windows
  smoke test remain part of milestone 5.
- [x] **Milestone 4 — AI reference context.** Provider adapters now receive the
  live frame plus an optional session-only reference crop, with the existing
  local validation and independent provider failure handling preserved.
- [x] **Milestone 5 — Verification and documentation.** The automated suite,
  syntax checks, README usage, privacy notes, and
  `WINDOWS_SMOKE_TEST.md` checklist are complete. The hands-on Windows run is
  now the operator acceptance step because this workspace is Linux-based.

## Milestones

### 1. Session-only target reference

- Capture the selected target crop when the operator clicks the source view.
- Store the reference in memory for the current source/session only.
- Track the target type as generic or human.
- Add clear-target and refresh-reference behavior.
- Keep camera images out of the repository and do not persist them by default.

### 2. Hybrid matching and safe camera decisions

- Continue using OpenCV for smooth frame-to-frame tracking.
- Compare local tracker boxes and AI boxes with the selected reference.
- Combine appearance similarity, motion consistency, box overlap, and provider
  confidence into a target score.
- Reject low-confidence or implausible jumps and hold the last safe camera
  target.
- Reinitialize the local tracker only after a candidate passes validation.
- Show the score and accept/reject reason in diagnostics.

### 3. Human tracking toggle and face detection

- Add a `Track human` toggle to the Tracking panel.
- Use local OpenCV person/face detection as an additional signal.
- Show face-detected state and human-match confidence.
- Allow tracking to continue when a face is temporarily hidden or turned away.
- Keep generic object tracking available when the toggle is off.

### 4. AI reference context

- Extend the provider request contract so AI can receive the current frame and,
  when available, the selected target reference crop.
- Ask providers to locate the selected target, not any convenient person.
- Continue validating every structured result locally before camera movement.
- Keep Gemini and Groq optional, independently failing, and rate-limited.

### 5. Verification and documentation

- Add unit tests for target scoring, box overlap, reference extraction, and
  human/face-mode decisions.
- Run syntax and non-GUI tests in this workspace.
- Run a Windows smoke test with a webcam, video file, or Android stream.
- Update README usage and privacy notes.

## Out of scope for the first implementation

- Face recognition or identity matching
- Saving biometric or camera references across sessions
- Security alarms, recording, or notifications
- Multi-target tracking
- Vehicle-specific detection

These can be added later using the same target-profile and confidence-gating
interfaces.