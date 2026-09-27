# Windows smoke test

Run this checklist from a Windows machine after installing the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q --ignore=github
python windows_camera_app.py
```

## 1. Video-file path

Use a short local video containing one visible person and confirm:

- [ ] The video starts and the Source Camera panel updates.
- [ ] Clicking the person creates a yellow local-tracker box and moves the
      virtual camera toward the selection.
- [ ] Ctrl-dragging a rectangle around the person creates a new yellow
      tracking box and captures a target reference.
- [ ] Ctrl+Shift-dragging a replacement rectangle resizes the selected box
      while keeping the existing target reference.
- [ ] A plain click still selects the object using the original centered box.
- [ ] The yellow tracker box moves with the target independently of the
      deliberately smoother green crop.
- [ ] Move the target rapidly out of the yellow box, then back into view;
      confirm local template matching reacquires it without replacing the
      saved target reference.
- [ ] A portrait video remains portrait-shaped in provider image requests; the
      diagnostics report source and encoded JPEG dimensions with each frame ID.
- [ ] `Refresh target reference` succeeds.
- [ ] `Clear tracking` removes the target and returns to Manual control.
- [ ] Enabling **Track human (local face detection)** shows face state and
      detected eye/mouth feature boxes and human-match confidence.
- [ ] Turning away or briefly occluding the face does not automatically clear
      the selected target or force Manual mode.
- [ ] Selecting a non-human object with the toggle off still works as generic
      object tracking.

## 2. Webcam path

Repeat the selection, refresh, clear, and human-toggle checks with **Webcam**
using camera index `0`. If the camera is unavailable, try index `1` and record
the index that works.

## 3. Android stream path

With the phone and Windows machine on the same network, use:

```text
http://PHONE_IP:8080/video
```

Confirm that the stream starts, the source remains responsive while tracking,
and the virtual-camera crop follows the selected target.

## 4. Optional AI path

Only if provider keys are configured in the Windows environment:

- [ ] Enable Automatic AI after selecting a target.
- [ ] Lose the local target and confirm AI activates on that same frame, even
      if Automatic AI was previously off.
- [ ] Confirm the provider panel shows the returned label and analyzed frame
      number, and the source overlay labels successful detections.
- [ ] Confirm HTTP 403/429 failures show provider-specific errors and cooldowns
      without stopping the local tracker or crashing the app.
- [ ] Confirm diagnostics show provider requests, target-match scores, and
      arbitration results.
- [ ] Confirm a selected target reference is used to distinguish the target
      from another visible person.
- [ ] Confirm disabling AI leaves local Assisted tracking available.

Do not commit camera footage, API keys, `.env` files, or virtual environments.
If a check fails, record the source type, camera index or stream URL shape
(without credentials), the visible status text, and the relevant diagnostics
lines.