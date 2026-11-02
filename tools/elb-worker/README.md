# Experimental stdio inference worker (no HTTP, no Qt)

This standalone CMake entry point adds llama-worker without changing llama-server.
The model is loaded once. Requests are sequential and stateless; a context is
allocated per chat and released afterwards. History must be supplied by the client.

## Build

From the repository root, using a recent CMake and C++17 compiler:

```sh
cmake -S tools/elb-worker -B build/elb-worker \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_SSE42=ON -DGGML_BMI2=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF -DGGML_F16C=OFF
cmake --build build/elb-worker --target llama-worker -j4
cmake --install build/elb-worker --prefix "$PWD/runtime" --component Unspecified
```

Alternatively copy the built executable from build/elb-worker/llama-worker to
runtime/bin. Inspect ldd before packaging: BUILD_SHARED_LIBS=OFF does not mean a
fully static binary. Do not bundle glibc casually. SSE4.2 still needs to be present
on the destination CPU. CentOS 7 build/runtime compatibility has NOT been tested.
The code targets the batch_ext/process API of this fork, not older llama releases.

## Run

```sh
./runtime/bin/llama-worker ./models/current.gguf 4096 8 2>./logs/worker.log
```

Create logs inside the deployment directory first. stdout is NDJSON only; stderr
is diagnostics. Read BOTH pipes concurrently in a parent process. Do not launch a
new worker per request. EOF on stdin shuts the worker down after the active request.

Example input lines:

```json
{"version":1,"type":"ping","request_id":"p1"}
{"version":1,"type":"chat","request_id":"c1","messages":[{"role":"user","content":"Explain RAII briefly. /no_think"}],"max_tokens":128,"temperature":0.2}
```

The worker emits ready after loading, pong for ping and result/error for each
request. Results include request_id, answer, finish_reason and usage. Error lines
are recoverable; a fatal startup error is written to stderr with a nonzero exit.
Input lines are limited to 1 MiB, messages to 128, max_tokens to 4096. Oversized
lines are drained without retaining excess bytes. Empty lines are ignored.

## Template/thinking limitations

messages use llama_chat_apply_template with the model's built-in template. This
is NOT the server's complete Jinja/chat-template engine. Unsupported templates
return errors. As an escape hatch, pass prompt containing a fully formatted model
prompt instead of messages. No role markup or model-specific template is invented.
For Qwen3 /no_think is only a soft instruction. enable_thinking and
chat_template_kwargs are deliberately rejected rather than silently ignored.
The answer may contain reasoning tags; there is no server-style reasoning parser.
No tool calling, vision, streaming, cancellation, deadlines, deduplication, prefix
cache or parallel requests are implemented in v1. Use only decoder-only text GGUFs.
A malformed model/backend failure may require restarting the worker.

## Main application integration contract

illmtransport.h is a dependency-free interface sketch, NOT a ready-made process
or HTTP client. Its exchange method carries the same request/result envelopes.
Implement ProcessLlmTransport (fork/exec with separate pipes, NOT shell commands),
HttpLlmTransport (map envelopes to /v1/chat/completions), and optionally an ActiveMQ
adapter in AiElbHelper. No files in AiElbHelper are changed by this commit.

The parent must serialize exchanges, validate matching request_id, bound replies,
read stderr, enforce deadlines and restart on crash. A timeout alone does not stop
inference. Until cancellation exists, terminate/restart the worker to abort. Do
not blindly replay timed-out requests. For ActiveMQ, define reply destinations,
correlation IDs, acknowledgements and redelivery/deduplication explicitly.

This implementation is uncompiled and untested in the GitHub tool environment.
Before production test: two chats in one process; malformed JSON followed by ping;
oversized line followed by ping; context overflow; EOF; unavailable model; pipe
closure; and runtime dependencies/CPU flags on the actual CentOS 7 target.
