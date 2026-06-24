import { NextResponse } from "next/server";
import { spawn } from "node:child_process";
import { openSync } from "node:fs";
import path from "node:path";
import { ensureDataDirs, logsDir } from "@/lib/file-utils";
import { readEnvKey } from "@/lib/storage/env-store";

export const runtime = "nodejs";

const LOCAL_URL = "http://localhost:7860";

async function alreadyRunning() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const response = await fetch(`${LOCAL_URL}/gradio_api/info`, { signal: controller.signal });
    clearTimeout(timeout);
    return response.ok;
  } catch {
    return false;
  }
}

// ponytail: fire-and-forget background launch of the user's own command. No stop/status/logs in
// app (the health badge IS status; kill it in the terminal; output tails to data/logs). Local
// dev/electron-main only — needs a node host that can spawn `sh`. Command is read ONLY from
// .env.local (never the request body) so a stray POST can't run arbitrary shell.
export async function POST() {
  if (await alreadyRunning()) {
    return NextResponse.json({ started: true, alreadyRunning: true });
  }

  const command = (await readEnvKey("VOXCPM_LOCAL_CMD")) || process.env.VOXCPM_LOCAL_CMD?.trim() || "";
  if (!command) {
    return NextResponse.json(
      { ok: false, error: "Set VOXCPM_LOCAL_CMD in .env.local to the command that launches your local VoxCPM on :7860." },
      { status: 400 }
    );
  }

  await ensureDataDirs();
  const logFd = openSync(path.join(logsDir, "voxcpm-local.log"), "a");
  spawn("sh", ["-c", command], { detached: true, stdio: ["ignore", logFd, logFd] }).unref();
  return NextResponse.json({ started: true });
}
