import { join } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createStaticServer } from "../scripts/server-lib.mjs";

export default async function globalSetup() {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");
  const server = createStaticServer(root);

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(4173, "127.0.0.1", resolve);
  });

  return async () => {
    await new Promise((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  };
}
