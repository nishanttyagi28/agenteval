import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");
const port = Number(process.env.PORT || 4174);

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

const server = createServer(async (req, res) => {
  const requestedPath = decodeURIComponent((req.url || "/").split("?")[0]);
  const relative = requestedPath === "/" ? "index.html" : requestedPath.replace(/^\/+/, "");
  const resolved = normalize(join(root, relative));

  if (!resolved.startsWith(root)) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  try {
    await stat(resolved);
    const ext = resolved.slice(resolved.lastIndexOf("."));
    res.writeHead(200, { "Content-Type": CONTENT_TYPES[ext] || "application/octet-stream" });
    createReadStream(resolved).pipe(res);
  } catch {
    res.writeHead(404).end("Not found");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`AgentEval docs site: http://127.0.0.1:${port}`);
});
