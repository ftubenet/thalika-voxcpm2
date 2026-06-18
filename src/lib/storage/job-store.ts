import fs from "node:fs/promises";
import { ensureDataDirs, idStamp, localIsoString, readMarkdownFiles, safeJoin, jobsDir, outputsDir } from "../file-utils";
import { parseMarkdown, serializeMarkdown, toNumber } from "../markdown-utils";
import type { JobRecord, OutputFormat, VoiceEmotion } from "../types";
import { deleteListeningReview, readListeningReview } from "./listening-review-store";

export function createJobId() {
  return `job_${idStamp()}`;
}

export async function saveJob(record: Omit<JobRecord, "createdAt"> & { createdAt?: string }) {
  await ensureDataDirs();
  const job: JobRecord = {
    ...record,
    createdAt: record.createdAt || localIsoString()
  };
  const markdown = serializeMarkdown(
    {
      id: job.id,
      scriptId: job.scriptId,
      title: job.title,
      provider: job.provider,
      format: job.format,
      speed: job.speed,
      emotion: job.emotion,
      status: job.status,
      audioFile: job.audioFile,
      error: job.error,
      completedChunks: job.completedChunks,
      totalChunks: job.totalChunks,
      progressMessage: job.progressMessage,
      voiceProfileId: job.voiceProfileId,
      lexiconRevision: job.lexiconRevision,
      normalizationChanges: job.normalizationChanges,
      referenceQualityScore: job.referenceQualityScore,
      referenceTranscriptUsed: job.referenceTranscriptUsed,
      createdAt: job.createdAt
    },
    job.content
  );

  await fs.writeFile(safeJoin(jobsDir, `${job.id}.md`), markdown, "utf8");
  return job;
}

export async function listJobs(limit = 20) {
  const files = await readMarkdownFiles(jobsDir);
  const jobs = files
    .map(({ content }) => {
      const parsed = parseMarkdown(content);
      return {
        id: parsed.frontmatter.id || "",
        scriptId: parsed.frontmatter.scriptId || "",
        title: parsed.frontmatter.title || "Untitled Script",
        provider: parsed.frontmatter.provider || "unknown",
        format: (parsed.frontmatter.format || "wav") as OutputFormat,
        speed: toNumber(parsed.frontmatter.speed, 1),
        emotion: (parsed.frontmatter.emotion || "neutral") as VoiceEmotion,
        status: parsed.frontmatter.status === "failed" ? "failed" : parsed.frontmatter.status === "generating" ? "generating" : "completed",
        audioFile: parsed.frontmatter.audioFile,
        error: parsed.frontmatter.error,
        completedChunks: toNumber(parsed.frontmatter.completedChunks, 0),
        totalChunks: toNumber(parsed.frontmatter.totalChunks, 0),
        progressMessage: parsed.frontmatter.progressMessage,
        voiceProfileId: parsed.frontmatter.voiceProfileId,
        lexiconRevision: parsed.frontmatter.lexiconRevision,
        normalizationChanges: toNumber(parsed.frontmatter.normalizationChanges, 0),
        referenceQualityScore: toNumber(parsed.frontmatter.referenceQualityScore, 0),
        referenceTranscriptUsed: parsed.frontmatter.referenceTranscriptUsed === "true",
        createdAt: parsed.frontmatter.createdAt || "",
        content: parsed.body
      } satisfies JobRecord;
    })
    .filter((job) => job.id)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .slice(0, limit);
  return Promise.all(jobs.map(async (job) => ({ ...job, review: await readListeningReview(job.id) })));
}

export async function deleteJob(jobId: string) {
  await ensureDataDirs();

  if (!/^job_[a-zA-Z0-9_-]+$/.test(jobId)) {
    throw new Error("Invalid job id");
  }

  const jobFilePath = safeJoin(jobsDir, `${jobId}.md`);
  const markdown = await fs.readFile(jobFilePath, "utf8");
  const parsed = parseMarkdown(markdown);
  const audioFile = parsed.frontmatter.audioFile;

  await fs.unlink(jobFilePath);
  await deleteListeningReview(jobId);

  let audioDeleted = false;
  if (audioFile) {
    // Resolve the path defensively: a malformed legacy audioFile would make safeJoin throw,
    // and that must not fail the whole delete after the job record is already gone.
    let audioPath: string | undefined;
    try {
      audioPath = safeJoin(outputsDir, audioFile);
    } catch {
      audioPath = undefined;
    }
    if (audioPath) {
      try {
        await fs.unlink(audioPath);
        audioDeleted = true;
      } catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
          throw error;
        }
      }
    }
  }

  return {
    jobId,
    audioFile,
    audioDeleted
  };
}
