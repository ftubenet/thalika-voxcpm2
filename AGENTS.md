# Thalika Agent Instructions

These instructions are for coding agents working on Thalika Voice Clone Studio.

Read `SKILL.md` first for the full project brain. Use this file as the day-to-day working contract.

## Project Identity

Thalika is a local-first Burmese voice-over and voice-clone studio.

Core principles:

- Keep the app local-first.
- Keep generated audio as real PCM WAV.
- Keep Burmese Production as the recommended provider.
- Keep user data on local disk.
- Use remote inference only when generation or rewrite is requested.
- Do not add databases, auth, cloud storage, local AI model hosting, Python backends, Docker, Runpod, payments, or video features.

## Current Product Surface

Main pages:

- `Script`: paste an original script, rewrite it for narration using Gemini, then send it to Voice Over.
- `Voice Over`: approve script normalization, select/upload voice reference, configure Burmese Production, and generate audio.
- `History`: play, download, delete, and review generated voice-over files.
- `Folders`: inspect local storage, open managed folders, and migrate legacy audio into real PCM WAV.

Current providers:

- `Burmese Production`: recommended Burmese preset built on VoxCPM2.
- `VoxCPM2 Multilingual`: direct VoxCPM2 flow for supported multilingual scripts.

Do not bring back Mock Provider, GPT-SoVITS, or CosyVoice unless the user explicitly asks.

## Architecture Rules

Use:

- Next.js App Router
- React
- TypeScript
- TailwindCSS
- Next.js Route Handlers for backend APIs
- Zod for validation
- local Markdown/JSON/files under `data/`

Do not use:

- Python backend
- FastAPI
- Docker
- Runpod
- Supabase
- PostgreSQL
- MongoDB
- Firebase
- authentication
- payments
- local GPU/model hosting

All filesystem API routes must include:

```ts
export const runtime = "nodejs";
```

## Local Storage Contract

Store data only in app-managed folders:

- `data/scripts/`
- `data/jobs/`
- `data/outputs/`
- `data/outputs/legacy-backup/`
- `data/memory/`
- `data/profiles/`
- `data/reviews/`
- `data/logs/`

Never expose arbitrary filesystem paths through API responses. Never accept user-supplied paths. Sanitize filenames and prevent path traversal.

Do not store raw reference audio bytes or sensitive transcript contents inside job Markdown. Profile lists must not return raw audio bytes.

## Audio Pipeline Rules

Final output must stay:

- WAV only
- real `RIFF/WAVE`
- `48kHz`
- mono
- `24-bit PCM`

Remote HF Space may return MP3. Decode it immediately in memory, write validated temporary WAV chunks, discard MP3 bytes, and merge WAV chunks only.

Never concatenate MP3 chunks. Never write mislabeled `.wav` files that contain MPEG audio.

Long script chunk pauses:

- `။`, `.`, `!`, `?`: `260ms`
- `၊`, `,`, `;`, `:`: `160ms`
- no explicit punctuation: `120ms`

Key files:

- `src/lib/script-chunker.ts`
- `src/lib/audio-utils.ts`
- `src/lib/providers/voxcpm2-provider.ts`
- `src/lib/storage/output-wav-migration.ts`

## Burmese Production Quality Rules

Protect these layers:

- reference audio or saved local profile
- exact reference transcript when using high-fidelity behavior
- browser-side reference quality report
- Burmese normalization
- local Burmese pronunciation lexicon
- normalization approval before generation
- chunked long-script generation
- listening QA after generation

Do not claim 100% human identity match. Phrase quality goals as improving similarity, stability, repeatability, and human review score.

## UI/UX Rules

The app should feel like a calm Apple-style creator tool:

- light theme
- clean rounded cards
- minimal copy
- clear tabs
- lucide icons
- mobile-responsive layout
- no clutter
- no marketing landing page

Prefer shared components. Do not split identical UI into many near-duplicate components.

Important UI files:

- `src/components/AppHeader.tsx`
- `src/components/StudioPageShell.tsx`
- `src/components/ScriptInput.tsx`
- `src/components/VoiceSettings.tsx`
- `src/components/GenerateButton.tsx`
- `src/components/HistoryPanel.tsx`
- `src/components/NormalizationApprovalPanel.tsx`
- `src/app/page.tsx`
- `src/app/script/page.tsx`
- `src/app/history/page.tsx`
- `src/app/storage/page.tsx`

## Common Debug Paths

Generation validation errors:

- `src/lib/validators.ts`
- `src/app/api/generate/route.ts`
- `src/lib/services/generation-service.ts`

Remote HF/VoxCPM2 failures:

- `src/lib/providers/hf-utils.ts`
- `src/lib/providers/voxcpm2-health.ts`
- `src/lib/providers/voxcpm2-provider.ts`

Long-script or merge failures:

- `src/lib/script-chunker.ts`
- `src/lib/audio-utils.ts`
- `src/lib/storage/generation-log.ts`

Profile/reference quality issues:

- `src/lib/browser-reference-audio.ts`
- `src/lib/reference-audio-quality.ts`
- `src/lib/storage/voice-profile-store.ts`
- `src/app/api/voice-profiles/route.ts`

Gemini rewrite issues:

- `src/lib/script-rewrite.ts`
- `src/lib/storage/env-store.ts`
- `src/app/api/rewrite/route.ts`

## Verification Checklist

Before finishing backend/provider/audio changes:

```bash
npm run lint
npm run build
```

For audio changes, also verify:

- short script generation works
- long multi-chunk script generation works
- final output starts with `RIFF`
- final output contains `WAVE`
- `file data/outputs/*.wav` reports PCM WAV, not MPEG audio
- `/api/audio/{filename}` returns `audio/wav`
- History playback and download still work

For UI changes, verify:

- Script tab rewrite flow still works
- rewritten script can move into Voice Over
- Voice Over generation still works
- History player layout does not overlap
- Folders storage cards still render
- mobile and desktop views remain readable

## Git And Data Safety

Do not commit user-generated private data from:

- `data/outputs/`
- `data/profiles/`
- `.env.local`
- logs containing private data

Do not delete generated user data unless the user explicitly asks or an app delete action is being implemented and tested carefully.

The repo may have unrelated local changes. Do not revert them without explicit permission.

## Recommended Future Handoff Docs

If the project grows, add deeper docs instead of bloating this file:

- `docs/ARCHITECTURE.md`
- `docs/VOICE_PIPELINE.md`
- `docs/UI_GUIDE.md`
- `docs/PROVIDER_CONTRACTS.md`

Keep this file short enough for agents to read at startup.
