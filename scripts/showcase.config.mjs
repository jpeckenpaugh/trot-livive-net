export default {
  key: "trot-livive-net",
  serve: { dir: "." },
  viewport: { width: 1280, height: 800 },
  durationMs: 30000,
  settleMs: 1800,

  async ready(page) {
    await page.waitForSelector("#btn-100", { timeout: 30000 });
    await page
      .waitForFunction(
        () => (document.querySelector("#m-total")?.textContent || "").trim().length > 0,
        null,
        { timeout: 8000 },
      )
      .catch(() => {});
  },

  beats: [
    {
      label: "Train a network you can read, 100 steps at a time",
      action: { click: "#btn-100" },
      wait: 2200,
      shot: "train-and-loss",
      thumb: true,
    },
    {
      label: "Export the hand-encoded ONNX graph",
      action: { click: "#btn-export" },
      wait: 2200,
      shot: "export-onnx",
    },
    {
      label: "Round-trip check against onnxruntime-web",
      action: { click: "#btn-roundtrip" },
      wait: 3000,
      shot: "roundtrip-check",
    },
    {
      label: "Inspect every weight in W1, b1, and W2",
      action: { scrollToSelector: "#panel-weights" },
      wait: 1500,
      shot: "weights",
    },
    {
      label: "Trace a prediction token by token",
      action: [
        { scrollToSelector: "#panel-trace" },
        { fill: { sel: "#trace-text", text: "the service was great" } },
        { click: "#btn-trace" },
      ],
      wait: 2000,
      shot: "trace-prediction",
    },
    {
      label: "Back to the loss curve",
      action: { scrollToSelector: "#panel-train" },
      wait: 1200,
    },
  ],

  thumbFrom: "train-and-loss",
};
