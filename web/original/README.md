# The original Chrome Dino game (vendored, unmodified)

These files are the Chrome offline "T-Rex runner" easter egg, from the Chromium source code, as
extracted by https://github.com/wayou/t-rex-runner (commit 5455bfa408ec6b707c7300ff194b7390733a766d).

| File | Origin |
|---|---|
| `index.js`, `index.css` | Chromium, via wayou/t-rex-runner, byte-for-byte |
| `assets/` | the original sprite sheets and icons, byte-for-byte |
| `audio-resources.html` | the original sound effects (the `<template id="audio-resources">` block), for reference; the same block is inlined in `../index.html` |

Licences: the Chromium code and assets are BSD-3-Clause, see `LICENSE.chromium`; the extraction is
BSD-3-Clause, see `LICENSE.t-rex-runner`.

**Do not edit these files.** Everything this project needs is done from outside, in
`../bridge.js` and `../dino-env.js`, through the game's own global `Runner.instance_`. Keeping them
unmodified is what lets us say the agent plays the original game. To update them, replace them with a
newer extraction and re-run the tests.
