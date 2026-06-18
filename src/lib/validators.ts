import { z } from "zod";
import { MAX_SCRIPT_CHARACTERS } from "./script-limits";

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
    provider: z.enum(["voxcpm2", "burmese_production"]),
    format: z.literal("wav"),
    speed: z.number().min(0.8, "Speed must be at least 0.8").max(1.2, "Speed must be at most 1.2"),
    emotion: z.enum(["neutral", "calm", "energetic", "dramatic"]),
    cloneMode: z.enum(["balanced", "high_fidelity"]).optional(),
    cloneStrength: z.number().min(1, "Clone strength must be at least 1.0").max(3, "Clone strength must be at most 3.0").optional(),
    denoiseReference: z.boolean().optional(),
    normalizeText: z.boolean().optional(),
    referenceAudio: referenceAudioSchema.optional(),
    referenceText: z.string().trim().max(2000, "Reference transcript must be 2000 characters or fewer").optional().or(z.literal("")),
    voiceProfileId: z.string().regex(/^profile_[a-zA-Z0-9_-]+$/, "Invalid voice profile id").optional(),
    referenceQualityReport: referenceQualitySchema.optional(),
    approvedNormalizedScript: z.string().trim().max(MAX_SCRIPT_CHARACTERS).optional(),
    lexiconRevision: z.string().trim().max(100).optional(),
    normalizationApproved: z.boolean().optional()
  })
  .superRefine((value, context) => {
    if (value.provider === "burmese_production" && !value.referenceAudio && !value.voiceProfileId) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceAudio"],
        message: "Burmese production cloning requires clean reference voice data"
      });
    }
    if (value.provider === "voxcpm2" && !value.referenceAudio && !value.voiceProfileId) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceAudio"],
        message: "VoxCPM2 requires reference audio for voice cloning"
      });
    }
    if (
      (value.provider === "burmese_production" || value.provider === "voxcpm2") &&
      !value.referenceText?.trim() &&
      !value.voiceProfileId
    ) {
      // VoxCPM leaks the reference audio tail into the output when prompt_text is empty
      // (worst on short scripts), regardless of clone mode. Every clone must send the exact
      // transcript. A saved profile is exempt here only because generation-service backfills
      // its transcript and re-checks server-side before inference.
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceText"],
        message: "Voice cloning requires the exact reference transcript spoken in the reference audio"
      });
    }
    if (value.provider === "burmese_production" && (!value.normalizationApproved || !value.approvedNormalizedScript || !value.lexiconRevision)) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["normalizationApproved"],
        message: "Review and approve the normalized Burmese script before generation"
      });
    }
    if (value.provider === "burmese_production" && value.referenceQualityReport?.status === "block") {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceQualityReport"],
        message: "Reference audio quality is blocked. Upload a cleaner voice sample"
      });
    }
    if ((value.provider === "voxcpm2" || value.provider === "burmese_production") && value.referenceAudio?.durationSeconds) {
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
