import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  convertRemoteAudioToPcm24Wav,
  getPunctuationAwarePauseMilliseconds,
  mergeWavFiles,
  normalizeMasterPeak,
  pcm24DurationSeconds,
  trimSilenceEdges,
  type PcmWavConversionResult
} from "../audio-utils";
import { ensureDataDirs, idStamp, outputsDir, safeJoin, sanitizeFilename } from "../file-utils";
import { REMOTE_TTS_CHUNK_CHARACTERS } from "../script-limits";
import { splitScriptIntoChunks } from "../script-chunker";
import type { GenerateVoiceInput, GenerateVoiceResult, ReferenceAudioPayload, VoiceEmotion } from "../types";
import { appendGenerationLog } from "../storage/generation-log";
import type { TTSProvider } from "./base";
import {
  assertOkResponse,
  extractAudioUrlFromEvents,
  fetchTextWithTimeout,
  fetchWithTimeout,
  getHFInferenceTimeout,
  getHFRequestTimeout,
  parseSSEData,
  parseUploadResponse,
  readJsonResponse,
  RemoteProviderError,
  shouldRetryHFError,
  summarizeRemoteEvents,
  TimeoutError,
  withRetry
} from "./hf-utils";
import { getVoxCPM2BaseUrl } from "./voxcpm2-health";

const emotionControls: Record<VoiceEmotion, string> = {
  neutral: "neutral expression",
  calm: "calm and steady expression",
  energetic: "energetic but speaker-consistent expression",
  dramatic: "expressive but speaker-consistent delivery"
};

function speedControl(speed: number) {
  if (speed <= 0.85) return "slow, deliberate pacing";
  if (speed <= 0.95) return "slightly slower pacing";
  if (speed >= 1.15) return "brisk pacing";
  if (speed >= 1.05) return "slightly faster pacing";
  return "natural pacing";
}

// Optional escape hatch for a self-hosted VoxCPM2 Space whose /generate exposes extra controls.
// Set VOXCPM2_EXTRA_PARAMS to a JSON array appended to the Gradio data[] in the order your Space
// expects (e.g. [20, true] for inference_timesteps + retry_badcase). Unset keeps the demo payload.
function parseExtraGenerateParams(): unknown[] {
  const raw = process.env.VOXCPM2_EXTRA_PARAMS?.trim();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// QA flag: keep an un-mastered raw sibling output for A/B comparison in History.
function keepRawOutput() {
  const value = process.env.THALIKA_KEEP_RAW_OUTPUT?.trim().toLowerCase();
  return value === "1" || value === "true" || value === "yes";
}

function decodeReferenceAudio(referenceAudio: ReferenceAudioPayload) {
  const match = referenceAudio.dataUrl.match(/^data:(audio\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) {
    throw new RemoteProviderError("Invalid reference audio", {
      publicMessage: "VoxCPM2 requires a valid audio reference file."
    });
  }

  return {
    mimeType: match[1],
    bytes: Buffer.from(match[2], "base64")
  };
}

async function uploadReferenceAudio(baseUrl: string, referenceAudio: ReferenceAudioPayload) {
  const { bytes, mimeType } = decodeReferenceAudio(referenceAudio);
  const filename = sanitizeFilename(referenceAudio.filename || "reference.wav");
  const form = new FormData();
  form.append("files", new Blob([bytes], { type: mimeType }), filename);

  const response = await fetchWithTimeout(`${baseUrl}/gradio_api/upload`, {
    method: "POST",
    body: form
  });
  assertOkResponse(response, "VoxCPM2 reference audio upload failed");

  const json = await readJsonResponse<unknown>(response, "Invalid response from VoxCPM2 Space.");
  return parseUploadResponse(json);
}

// A submission (POST) enqueues remote inference and returns an event id; the result (GET)
// only reads that queued job's output. Retrying the POST starts a *new* inference, so the
// two stages need different retry policies — see shouldRetrySubmit and the staged retries below.
async function submitVoxCPM2Generation(
  baseUrl: string,
  input: GenerateVoiceInput,
  uploadedReferencePath: string,
  scriptChunk: string,
  chunkIndex: number,
  chunkCount: number
) {
  const cloneMode = input.cloneMode || "high_fidelity";
  // cfg_value: VoxCPM2's documented sweet spot is ~2.0. Higher improves prompt adherence but
  // reduces naturalness (more robotic), so keep defaults near 2.0 — the slider can still override.
  const cloneStrength = Math.min(3, Math.max(1, input.cloneStrength ?? (cloneMode === "high_fidelity" ? 2 : 1.7)));
  const denoiseReference = input.denoiseReference ?? false;
  const normalizeText = input.normalizeText ?? true;
  // NEVER send prompt_text (use_prompt_text=false): VoxCPM speaks the prompt transcript and
  // prepends it to the output ("ultimate mode" leak). Audio-only cloning is clean — proven by
  // direct testing — and means the user never has to type the reference transcript.
  const continuityInstruction =
    chunkCount > 1
      ? ` This is segment ${chunkIndex + 1} of ${chunkCount}; keep the same speaker identity, pace, volume, accent, and emotional style so all segments join naturally.`
      : "";
  const controlInstruction =
    cloneMode === "high_fidelity"
      ? `Preserve the uploaded speaker identity as closely as possible: timbre, accent, pitch range, rhythm, breath, tone, speaking style, and Burmese pronunciation. Use ${emotionControls[input.emotion]} with ${speedControl(input.speed)}.${continuityInstruction}`
      : `Clone the uploaded speaker while keeping natural speech. Use ${emotionControls[input.emotion]} with ${speedControl(input.speed)}.${continuityInstruction}`;
  const data: unknown[] = [
    scriptChunk,
    controlInstruction,
    {
      path: uploadedReferencePath,
      orig_name: sanitizeFilename(input.referenceAudio?.filename || "reference.wav"),
      mime_type: input.referenceAudio?.mimeType || "audio/wav",
      meta: { _type: "gradio.FileData" }
    },
    false, // use_prompt_text — always off; see comment above
    "", // prompt_text — never sent
    cloneStrength,
    normalizeText,
    denoiseReference,
    // Config-only Path B hook: a self-hosted Space whose /generate exposes extra controls
    // (inference_timesteps, retry_badcase, …) receives them here without a code change.
    ...parseExtraGenerateParams()
  ];
  const body = {
    data
  };

  const response = await fetchWithTimeout(`${baseUrl}/gradio_api/call/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  assertOkResponse(response, "VoxCPM2 remote inference failed");

  const json = await readJsonResponse<{ event_id?: string }>(response, "Invalid response from VoxCPM2 Space.");
  if (!json.event_id) {
    throw new RemoteProviderError("Missing Gradio event id", {
      publicMessage: "Invalid response from VoxCPM2 Space."
    });
  }

  return { eventId: json.event_id };
}

async function fetchVoxCPM2Result(
  baseUrl: string,
  eventId: string,
  input: GenerateVoiceInput,
  chunkIndex: number,
  chunkCount: number
) {
  const { response: resultResponse, text: resultText } = await fetchTextWithTimeout(`${baseUrl}/gradio_api/call/generate/${eventId}`, {
    method: "GET",
    headers: { Accept: "text/event-stream" }
  });
  assertOkResponse(resultResponse, "VoxCPM2 remote inference failed");

  const events = parseSSEData(resultText);
  try {
    return extractAudioUrlFromEvents(events, baseUrl);
  } catch (error) {
    await appendGenerationLog("remote_sse_without_audio", {
      jobId: input.jobId,
      chunk: chunkIndex + 1,
      chunks: chunkCount,
      events: JSON.stringify(summarizeRemoteEvents(events)),
      error: diagnosticError(error)
    });
    throw error;
  }
}

// A submission timeout is ambiguous: the job may already be queued, so re-POSTing would run
// inference twice. Only retry the submit on an explicit pre-enqueue rejection (429/503).
function shouldRetrySubmit(error: unknown) {
  return error instanceof RemoteProviderError && error.retryable;
}

// The public Space rejects bursts of submissions with 503 under load; retry patiently (with
// capped backoff) so a transient busy window doesn't surface as a hard generation failure.
// Safe because a rejected submission never enqueued inference — no duplicate-run risk.
const SUBMIT_RETRY_ATTEMPTS = 5;

// Client-side retry_badcase: VoxCPM occasionally emits a take far longer than the text warrants —
// a repeated phrase or an echoed reference tail. Detect it by chars-per-second (normal Burmese TTS
// runs ~8-20 cps; a leaked/repeated take drops well below) and regenerate, keeping the densest take.
const BADCASE_MAX_RETRIES = 2;
const BADCASE_MIN_SECONDS = 6; // never second-guess short clips, where cps is noisy
const BADCASE_MIN_CHARS_PER_SECOND = 4.5;

function charsPerSecond(chunkText: string, wav: Buffer) {
  const seconds = pcm24DurationSeconds(wav);
  return seconds > 0 ? chunkText.trim().length / seconds : Infinity;
}

function isBadCaseTake(chunkText: string, wav: Buffer) {
  const seconds = pcm24DurationSeconds(wav);
  if (seconds <= BADCASE_MIN_SECONDS) return false;
  return charsPerSecond(chunkText, wav) < BADCASE_MIN_CHARS_PER_SECOND;
}

async function downloadRemoteAudio(audioUrl: string) {
  const response = await fetchWithTimeout(audioUrl, { method: "GET" });
  assertOkResponse(response, "VoxCPM2 audio download failed");

  const contentType = response.headers.get("content-type") || "";
  if (contentType && !contentType.includes("audio") && !contentType.includes("octet-stream")) {
    throw new RemoteProviderError("Unexpected VoxCPM2 audio response type", {
      publicMessage: "Invalid response from VoxCPM2 Space."
    });
  }

  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length === 0) {
    throw new RemoteProviderError("Empty VoxCPM2 audio response", {
      publicMessage: "VoxCPM2 audio download failed."
    });
  }

  return bytes;
}

function normalizeVoxCPM2Error(error: unknown) {
  if (error instanceof TimeoutError) return "Remote inference timed out.";
  if (error instanceof RemoteProviderError) return error.publicMessage;
  return "VoxCPM2 remote inference failed";
}

function diagnosticError(error: unknown) {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return "Unknown remote inference error";
}

async function generateRemote(input: GenerateVoiceInput) {
  if (!input.referenceAudio) {
    throw new RemoteProviderError("Missing reference audio", {
      publicMessage: "VoxCPM2 requires reference audio for voice cloning."
    });
  }
  const referenceAudio = input.referenceAudio;

  await ensureDataDirs();
  const baseUrl = await getVoxCPM2BaseUrl();
  const chunks = splitScriptIntoChunks(input.script, REMOTE_TTS_CHUNK_CHARACTERS);
  if (chunks.length === 0) {
    throw new RemoteProviderError("Empty script", {
      publicMessage: "Script is required."
    });
  }

  const uploadedReferencePath = await withRetry(
    () => uploadReferenceAudio(baseUrl, referenceAudio),
    shouldRetryHFError,
    2,
    async (error, attempt) => {
      await appendGenerationLog("reference_upload_retry", {
        jobId: input.jobId,
        attempt,
        error: diagnosticError(error)
      });
    }
  );
  const outputStem = sanitizeFilename(`voice_${idStamp()}`);
  const temporaryDir = await fs.mkdtemp(path.join(os.tmpdir(), "thalika-voxcpm2-"));
  let result: GenerateVoiceResult | undefined;

  try {
    const audioChunkPaths: string[] = [];
    const remoteFormats = new Set<string>();
    await appendGenerationLog("generation_started", {
      jobId: input.jobId,
      provider: "voxcpm2",
      characters: input.script.length,
      chunks: chunks.length
    });
    await input.onProgress?.({
      completedChunks: 0,
      totalChunks: chunks.length,
      message: `Preparing ${chunks.length} audio segment${chunks.length === 1 ? "" : "s"}.`
    });

    for (const [chunkIndex, chunk] of chunks.entries()) {
      await appendGenerationLog("chunk_started", {
        jobId: input.jobId,
        chunk: chunkIndex + 1,
        chunks: chunks.length,
        characters: chunk.length
      });
      await input.onProgress?.({
        completedChunks: chunkIndex,
        totalChunks: chunks.length,
        message: `Generating audio segment ${chunkIndex + 1} of ${chunks.length}.`
      });
      const logStageRetry = (stage: string) => async (error: unknown, attempt: number) => {
        await appendGenerationLog("chunk_retry", {
          jobId: input.jobId,
          chunk: chunkIndex + 1,
          chunks: chunks.length,
          stage,
          attempt,
          error: diagnosticError(error)
        });
      };

      // One full attempt: enqueue (POST, retry 429/503 only — an ambiguous timeout must not
      // re-enqueue), read the SAME event id (retry safe), download (idempotent), decode to PCM.
      const produceTake = async (): Promise<PcmWavConversionResult> => {
        const submission = await withRetry(
          () => submitVoxCPM2Generation(baseUrl, input, uploadedReferencePath, chunk, chunkIndex, chunks.length),
          shouldRetrySubmit,
          SUBMIT_RETRY_ATTEMPTS,
          logStageRetry("submit")
        );
        const remoteAudioUrl = await withRetry(
          () => fetchVoxCPM2Result(baseUrl, submission.eventId, input, chunkIndex, chunks.length),
          shouldRetryHFError,
          2,
          logStageRetry("result")
        );
        const audio = await withRetry(() => downloadRemoteAudio(remoteAudioUrl), shouldRetryHFError, 2, logStageRetry("download"));
        try {
          return await convertRemoteAudioToPcm24Wav(audio);
        } catch {
          throw new RemoteProviderError("Remote audio decode failed", {
            publicMessage: "VoxCPM2 returned an audio segment that could not be decoded into PCM WAV."
          });
        }
      };

      // Client-side retry_badcase: if a take runs far longer than the text warrants (a repeat or a
      // leaked reference echo), regenerate and keep the densest (least-padded) take.
      let converted = await produceTake();
      for (let attempt = 1; attempt <= BADCASE_MAX_RETRIES && isBadCaseTake(chunk, converted.wav); attempt += 1) {
        await appendGenerationLog("chunk_badcase_retry", {
          jobId: input.jobId,
          chunk: chunkIndex + 1,
          chunks: chunks.length,
          attempt,
          seconds: pcm24DurationSeconds(converted.wav).toFixed(2),
          charsPerSecond: charsPerSecond(chunk, converted.wav).toFixed(2)
        });
        const candidate = await produceTake();
        if (charsPerSecond(chunk, candidate.wav) > charsPerSecond(chunk, converted.wav)) {
          converted = candidate;
        }
      }

      const chunkPath = path.join(temporaryDir, `chunk-${chunkIndex}.wav`);
      await fs.writeFile(chunkPath, converted.wav);
      audioChunkPaths.push(chunkPath);
      remoteFormats.add(converted.remoteFormat);
      await appendGenerationLog("chunk_completed", {
        jobId: input.jobId,
        chunk: chunkIndex + 1,
        chunks: chunks.length,
        remoteFormat: converted.remoteFormat,
        pcmWavBytes: converted.wav.length,
        seconds: pcm24DurationSeconds(converted.wav).toFixed(2),
        charsPerSecond: charsPerSecond(chunk, converted.wav).toFixed(2)
      });
      await input.onProgress?.({
        completedChunks: chunkIndex + 1,
        totalChunks: chunks.length,
        message: `Generated audio segment ${chunkIndex + 1} of ${chunks.length}.`
      });
    }

    const format = "wav";
    const filename = sanitizeFilename(`${outputStem}.wav`);
    const audioFilePath = safeJoin(outputsDir, filename);
    const punctuationAwarePauses = chunks
      .slice(0, -1)
      .map(getPunctuationAwarePauseMilliseconds);
    await appendGenerationLog("merge_started", {
      jobId: input.jobId,
      chunks: chunks.length,
      format,
      encoding: "pcm_s24le",
      pausesMilliseconds: punctuationAwarePauses.join(",")
    });

    // QA-only: when THALIKA_KEEP_RAW_OUTPUT is set, keep an un-mastered sibling merged from the
    // SAME chunks (no trim, no normalize) so History can A/B the mastering on identical content.
    let rawAudioFile: string | undefined;
    if (keepRawOutput()) {
      rawAudioFile = sanitizeFilename(`${outputStem}_raw.wav`);
      await mergeWavFiles([...audioChunkPaths], safeJoin(outputsDir, rawAudioFile), punctuationAwarePauses);
    }

    // Trim each chunk's edge silence, then merge so the only inter-chunk gap is the controlled
    // punctuation pause (steadier rhythm), then master (peak-normalize + edge fades). Only new
    // generations are mastered — legacy-file migration must not re-level existing user audio.
    for (const chunkPath of audioChunkPaths) {
      await trimSilenceEdges(chunkPath);
    }
    await mergeWavFiles(audioChunkPaths, audioFilePath, punctuationAwarePauses);
    await normalizeMasterPeak(audioFilePath);
    await appendGenerationLog("generation_completed", {
      jobId: input.jobId,
      chunks: chunks.length,
      filename,
      format,
      rawCopy: Boolean(rawAudioFile)
    });
    result = {
      filename,
      audioFilePath,
      format,
      localAudioUrl: `/api/audio/${filename}`,
      rawAudioFile,
      metadata: {
        remoteProvider: "huggingface-space",
        remoteBaseUrl: baseUrl,
        remoteFormats: [...remoteFormats].join(","),
        outputEncoding: "pcm_s24le",
        outputSampleRate: 48_000,
        outputChannels: 1,
        outputBitDepth: 24,
        pausePolicy: "punctuation-aware",
        mode: "voxcpm2-controllable-cloning",
        cloneMode: input.cloneMode || "high_fidelity",
        cloneStrength: input.cloneStrength ?? 2,
        denoiseReference: input.denoiseReference ?? false,
        normalizeText: input.normalizeText ?? true,
        referenceTranscriptUsed: false,
        paceGuidance: speedControl(input.speed),
        chunkedGeneration: chunks.length > 1,
        chunkCount: chunks.length,
        chunkMaxCharacters: REMOTE_TTS_CHUNK_CHARACTERS,
        originalCharacters: input.script.length,
        timeoutMs: getHFRequestTimeout(),
        inferenceTimeoutMs: getHFInferenceTimeout()
      }
    };
  } catch (error) {
    await appendGenerationLog("generation_failed", {
      jobId: input.jobId,
      chunks: chunks.length,
      error: diagnosticError(error),
      publicMessage: normalizeVoxCPM2Error(error)
    });
    throw error;
  } finally {
    await fs.rm(temporaryDir, { recursive: true, force: true });
  }

  if (!result) throw new Error("VoxCPM2 generation completed without a local audio result.");
  return result;
}

export const voxcpm2Provider: TTSProvider = {
  id: "voxcpm2",
  name: "VoxCPM2",
  async generate(input) {
    try {
      return await generateRemote(input);
    } catch (error) {
      throw new RemoteProviderError("VoxCPM2 remote inference failed", {
        publicMessage: normalizeVoxCPM2Error(error)
      });
    }
  }
};
