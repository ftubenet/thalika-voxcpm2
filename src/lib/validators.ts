import { z } from "zod";
import { MAX_SCRIPT_CHARACTERS } from "./script-limits";
import { detectScriptLanguage } from "./language-utils";

const referenceAudioSchema = z.object({
  dataUrl: z.string().startsWith("data:audio/", "Reference audio must be an audio data URL"),
  filename: z.string().min(1, "Reference audio filename is required").max(150, "Reference audio filename is too long"),
  mimeType: z.string().startsWith("audio/", "Reference audio must be an audio file"),
  size: z.number().positive("Reference audio is empty").max(10 * 1024 * 1024, "Reference audio must be 10MB or smaller"),
  durationSeconds: z.number().positive().optional()
});

const referenceQualitySchema = z.object({
  durationSeconds: z.number().positive(),
  silenceRatio: z.number().min(0).max(1),
  clippingRatio: z.number().min(0).max(1),
  rms: z.number().finite().min(0).max(16, "Reference RMS level is unexpectedly high"),
  peak: z.number().finite().min(0).max(16, "Reference peak level is unexpectedly high"),
  score: z.number().min(0).max(100),
  status: z.enum(["pass", "warn", "block"]),
  issues: z.array(z.string().max(200)).max(20)
});

export const generateRequestSchema = z
  .object({
    title: z.string().trim().max(100, "Title must be 100 characters or fewer").optional().or(z.literal("")),
    script: z
      .string()
      .trim()
      .min(10, "Script must be at least 10 characters")
      .max(MAX_SCRIPT_CHARACTERS, `Script must be ${MAX_SCRIPT_CHARACTERS.toLocaleString()} characters or fewer`),
    provider: z.enum(["voxcpm2"]),
    format: z.literal("wav"),
    speed: z.number().min(0.8, "Speed must be at least 0.8").max(1.2, "Speed must be at most 1.2"),
    emotion: z.enum(["neutral", "calm", "energetic", "dramatic"]),
    cloneMode: z.enum(["balanced", "high_fidelity"]).optional(),
    cloneStrength: z.number().min(1, "Clone strength must be at least 1.0").max(3, "Clone strength must be at most 3.0").optional(),
    inferenceTimesteps: z.number().int().min(4, "Timesteps must be at least 4").max(50, "Timesteps must be at most 50").optional(),
    denoiseReference: z.boolean().optional(),
    normalizeText: z.boolean().optional(),
    referenceAudio: referenceAudioSchema.optional(),
    referenceText: z.string().trim().max(2000, "Reference transcript must be 2000 characters or fewer").optional().or(z.literal("")),
    voiceDescription: z.string().trim().max(500, "Voice description must be 500 characters or fewer").optional().or(z.literal("")),
    voiceProfileId: z.string().regex(/^profile_[a-zA-Z0-9_-]+$/, "Invalid voice profile id").optional(),
    referenceQualityReport: referenceQualitySchema.optional(),
    approvedNormalizedScript: z.string().trim().max(MAX_SCRIPT_CHARACTERS).optional(),
    lexiconRevision: z.string().trim().max(100).optional(),
    normalizationApproved: z.boolean().optional()
  })
  .superRefine((value, context) => {
    // Burmese scripts get the production QA layer (normalization approval + reference-quality
    // gate) automatically — the trigger is the detected language, not a separate provider.
    const isBurmeseScript = detectScriptLanguage(value.script).code === "my";

    // Need one source of voice identity: reference audio, a saved profile, or — for Voice Design —
    // a text description. (Design = no reference + a description; the model creates a new voice.)
    if (!value.referenceAudio && !value.voiceProfileId && !value.voiceDescription?.trim()) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceAudio"],
        message: "Add a reference clip to clone, or a voice description to design a new voice"
      });
    }
    // The reference transcript is intentionally NOT required: it is never sent to the model
    // (use_prompt_text=false), because sending it makes VoxCPM speak the transcript and prepend
    // it to the output. Audio-only cloning is clean and needs no transcript from the user.
    if (isBurmeseScript && (!value.normalizationApproved || !value.approvedNormalizedScript || !value.lexiconRevision)) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["normalizationApproved"],
        message: "Review and approve the normalized Burmese script before generation"
      });
    }
    if (isBurmeseScript && value.referenceQualityReport?.status === "block") {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceQualityReport"],
        message: "Reference audio quality is blocked. Upload a cleaner voice sample"
      });
    }
    if (value.referenceAudio?.durationSeconds) {
      if (value.referenceAudio.durationSeconds < 3) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["referenceAudio"],
          message: "Reference audio is too short. Use at least 3 seconds, ideally 6-15 seconds"
        });
      }
      if (value.referenceAudio.durationSeconds > 50) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["referenceAudio"],
          message: "Reference audio is too long for VoxCPM2. Trim it to 6-30 seconds of clean speech"
        });
      }
    }
  });

export function formatValidationError(error: z.ZodError) {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "request"}: ${issue.message}`)
    .join(". ");
}
