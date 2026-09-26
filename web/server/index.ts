import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { join } from "node:path";
import { createApp } from "./app";
import { PythonBridge, root } from "./bridge";
const bridge = new PythonBridge();
const app = createApp(bridge, join(root, ".robot-runtime/web-routines"));
app.use("*", serveStatic({ root: "./dist" }));
app.get("*", serveStatic({ path: "./dist/index.html" }));
const server = serve(
  {
    fetch: app.fetch,
    hostname: "127.0.0.1",
    port: Number(process.env.STUDIO_API_PORT || 8787),
  },
  () =>
    console.log(
      `Motion Studio API: http://127.0.0.1:${process.env.STUDIO_API_PORT || 8787}`,
    ),
);
function shutdown() {
  bridge.close();
  server.close();
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
