import { join } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createStaticServer } from "./server-lib.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");
const port = Number(process.env.PORT || 4173);
const server = createStaticServer(root);

server.listen(port, "127.0.0.1", () => {
  console.log(`AgentEval landing page: http://127.0.0.1:${port}`);
});
