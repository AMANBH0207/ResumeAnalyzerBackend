import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import multer from "multer";
import mammoth from "mammoth";
import pdfParse from "pdf-parse/lib/pdf-parse.js";
import { isIP } from "node:net";
import { analyzeWithGroq } from "./service/groq.service.js";

const app = express();
const PORT = Number(process.env.PORT || 5000);
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";
const MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

app.use(helmet());
app.use(cors({ origin: FRONTEND_URL.split(",").map((item) => item.trim()), methods: ["GET", "POST"] }));
app.use(express.json({ limit: "1mb" }));

const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024, files: 1 },
  fileFilter(_req, file, callback) {
    const allowed = /\.(pdf|docx)$/i.test(file.originalname);
    if (!allowed) return callback(new Error("Only PDF and DOCX resumes are supported."));
    callback(null, true);
  },
});

app.get("/api/health", (_req, res) => res.json({ success: true, service: "AIAnalyzer API", model: MODEL }));

function stripHtml(html) {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<(br|\/p|\/div|\/li|\/h[1-6]|\/section|\/article)\b[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"')
    .replace(/[ \t]+/g, " ").replace(/\n\s*\n+/g, "\n\n").trim();
}

function isUnsafeHost(hostname) {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (["localhost", "localhost.localdomain", "metadata.google.internal"].includes(host) || host.endsWith(".local")) return true;
  if (host === "::1" || host.startsWith("fe80:") || host.startsWith("fc") || host.startsWith("fd")) return true;
  if (isIP(host) === 4) {
    const [a, b] = host.split(".").map(Number);
    return a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || a >= 224;
  }
  return false;
}

async function getJobText(jobUrl) {
  let url;
  try { url = new URL(jobUrl); } catch { throw new Error("Enter a valid job URL or paste the job description instead."); }
  if (!(["http:", "https:"].includes(url.protocol)) || isUnsafeHost(url.hostname)) {
    throw new Error("This job URL cannot be fetched. Paste the job description instead.");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 9000);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      redirect: "manual",
      headers: { "User-Agent": "AIAnalyzer/1.0 (job description extraction)", Accept: "text/html,application/xhtml+xml" },
    });
    if (response.status >= 300 && response.status < 400) throw new Error("This job site redirected the request. Paste the job description instead.");
    if (!response.ok) throw new Error("We couldn't access that job page. Paste the job description instead.");
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html") && !contentType.includes("text/plain")) throw new Error("That URL doesn't appear to contain a readable job page. Paste the job description instead.");
    const html = (await response.text()).slice(0, 2_000_000);
    const text = contentType.includes("text/plain") ? html : stripHtml(html);
    if (text.length < 100) throw new Error("We couldn't extract enough job details from that page. Paste the job description instead.");
    return text.slice(0, 18000);
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The job page took too long to load. Paste the job description instead.");
    throw error;
  } finally { clearTimeout(timeout); }
}

async function extractResume(file) {
  const name = file.originalname.toLowerCase();
  let text = "";
  if (name.endsWith(".pdf")) {
    const parsed = await pdfParse(file.buffer);
    text = parsed.text || "";
  } else if (name.endsWith(".docx")) {
    const parsed = await mammoth.extractRawText({ buffer: file.buffer });
    text = parsed.value || "";
  }
  text = text.replace(/\r/g, "\n").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  if (text.length < 50) throw new Error("We couldn't read enough text from this resume. If it's a scanned PDF, try an OCR-enabled or text-based PDF.");
  return text.slice(0, 18000);
}

function normalizeList(value) {
  return Array.isArray(value) ? value : [];
}
function cleanSkill(item) {
  if (typeof item === "string") return { name: item, evidence: "" };
  return { name: String(item?.name || "Unknown skill").slice(0, 100), evidence: String(item?.evidence || "").slice(0, 300) };
}
function validateAnalysis(raw) {
  const statuses = new Set(["found", "partial", "missing"]);
  const matchedSkills = normalizeList(raw.matchedSkills).map(cleanSkill).slice(0, 30);
  const partialSkills = normalizeList(raw.partialSkills).map(cleanSkill).slice(0, 30);
  const missingSkills = normalizeList(raw.missingSkills).map(cleanSkill).slice(0, 30);
  const keywords = normalizeList(raw.keywords).map((item) => ({
    name: String(item?.name || "keyword").slice(0, 80),
    status: statuses.has(item?.status) ? item.status : "missing",
  })).slice(0, 60);
  const suggestions = normalizeList(raw.suggestions).map((item) => ({ title: String(item?.title || "Review your resume").slice(0, 150), detail: String(item?.detail || "").slice(0, 500) })).slice(0, 8);
  const requirements = normalizeList(raw.requirements).map((item) => String(item).slice(0, 250)).slice(0, 25);
  // The backend owns the score so it is repeatable and inspectable, rather than trusting an arbitrary model-generated number.
  const total = matchedSkills.length + partialSkills.length + missingSkills.length;
  const score = total ? Math.round((matchedSkills.length + partialSkills.length * 0.5) / total * 100) : 0;
  return {
    score,
    summary: String(raw.summary || "Review the skill evidence and recommendations below.").slice(0, 700),
    matchedSkills,
    partialSkills,
    missingSkills,
    keywords,
    suggestions,
    requirements,
    demo: false,
  };
}

function parseJson(text) {
  const trimmed = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try { return JSON.parse(trimmed); } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1));
    throw new Error("The AI returned an unreadable response. Please try again.");
  }
}

app.post("/api/analyze", upload.single("resume"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Upload a PDF or DOCX resume first." });
    const jobUrl = String(req.body.jobUrl || "").trim();
    let jobDescription = String(req.body.jobDescription || "").trim();
    if (jobDescription.length > 20000) return res.status(400).json({ error: "Job description must be 20,000 characters or fewer." });
    if (!jobDescription && !jobUrl) return res.status(400).json({ error: "Enter a job URL or paste a job description." });
    if (!jobDescription && jobUrl) jobDescription = await getJobText(jobUrl);
    if (jobDescription.length < 80) return res.status(400).json({ error: "Please provide a more complete job description (at least 80 characters)." });
    const resumeText = await extractResume(req.file);

    let rawText;

    try {
      rawText = await analyzeWithGroq(
        resumeText,
        jobDescription
      );
    } catch (error) {
      console.error("Groq request failed:", error.message);

      if (
        error.status === 429 ||
        error.status === 413 ||
        /rate.?limit|quota/i.test(error.message || "")
      ) {
        if (error.status === 413) {
          return res.status(413).json({
            error: "The request is too large for the selected model.",
          });
        }

        return res.status(429).json({
          error: "Groq rate limit reached. Please try again later.",
        });
      }

      return res.status(502).json({
        error:
          "Groq analysis failed. Check your API key, model ID, and server logs.",
      });
    }
    const raw = parseJson(rawText);
    return res.json({ success: true, analysis: validateAnalysis(raw), source: { resumeName: req.file.originalname, jobSource: jobUrl && !req.body.jobDescription ? "url" : "provided-description" } });
  } catch (error) { next(error); }
});

app.use((error, _req, res, _next) => {
  console.error("API error:", error.message);
  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") return res.status(413).json({ error: "Resume exceeds the 5 MB upload limit." });
    return res.status(400).json({ error: "Unable to upload the resume. Please try another file." });
  }
  const status = error.status || 400;
  return res.status(status).json({ error: error.message || "Unexpected server error." });
});

app.listen(PORT, () => console.log(`AIAnalyzer API running at http://localhost:${PORT} (model: ${MODEL})`));
