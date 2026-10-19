TROT — Transparent Rust ONNX Trainer.

TROT is a small neural network you can read, run, and export. The model is written in plain Rust — forward pass, backward pass, loss, weight update — and compiled to WebAssembly so it trains in the browser with no server, no Python, and no install. Each step renders its numbers to the DOM: weight matrices, activations, loss, one multiplication at a time. Train it, trace a single forward pass by hand, then export the result to ONNX and load it in any standard runtime. The point is not the model's accuracy. The point is that nothing is hidden.
