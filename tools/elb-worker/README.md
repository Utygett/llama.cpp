# Experimental asynchronous stdio worker (no HTTP, no Qt)

The model is loaded once. stdin is read by the main thread; chat jobs enter a
bounded FIFO queue and run in a fixed thread pool. Each running job owns its
context, KV cache, batch and sampler. CPU model weights are shared. Context
initialization is serialized; prefill and decoding are not. Contexts are released
per request; history must be supplied by the client. This is independent-context
concurrency, not continuous batching. Validate the shared CPU backend on your
actual deployment before production; concurrent inference has not been tested here.

## Build

From the repository root, using a recent CMake and C++17 compiler:

```sh
cmake -S tools/elb-worker -B build/elb-worker \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_SSE42=ON -DGGML_BMI2=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF -DGGML_F16C=OFF
cmake --build build/elb-worker --target llama-worker -j4
mkdir -p runtime/bin logs
cp build/elb-worker/llama-worker runtime/bin/
```

The existing target links llama, which carries the platform thread dependency.
Inspect ldd before packaging; this is not a fully static executable. Do not copy
glibc into the project casually. SSE4.2 must be available on the destination CPU.
Use the project-local CMake and devtoolset GCC on CentOS 7 as before.

## Run

```sh
./runtime/bin/llama-worker ./models/current.gguf 4096 4 2 16 2>./logs/worker.log
```

Arguments: MODEL [CTX=4096] [THREADS_PER_JOB=4] [WORKERS=1] [QUEUE=16].
Old invocations remain valid with one generator, but stdin is now asynchronous.
WORKERS range: 1..16. QUEUE range: 1..256, counts waiting jobs only.
For an 8-vCPU VM start with 2 workers and 4 threads per job. This is not a CPU
quota; backend/helper threads also exist. More jobs need more KV/scratch memory
and may increase latency instead of improving aggregate throughput.

## Protocol v1 (NDJSON)

Each input/output is one JSON line, at most 1 MiB for input. Stdout writes are
mutex-protected and flushed per complete line. Read stderr independently. IDs
must be nonempty strings of at most 128 bytes. Use unique IDs for ALL requests.
Pending/active duplicate chat IDs are rejected, not replayed or deduplicated
across restarts. Reusing an active ID creates ambiguous error correlation, so
clients must never do so. IDs are released just after the terminal output write.

```json
{"version":1,"type":"chat","request_id":"A","stream":true,"messages":[{"role":"user","content":"List five C++ features. /no_think"}],"max_tokens":160,"temperature":0.2}
{"version":1,"type":"chat","request_id":"B","stream":true,"messages":[{"role":"user","content":"List five Linux commands. /no_think"}],"max_tokens":160,"temperature":0.2}
{"version":1,"type":"ping","request_id":"P"}
```

Events:
- ready: model loaded and pool created; includes workers/threads_per_job/queue_capacity.
- started: job dequeued; includes request_id, zero-based slot, elapsed_ms from pool creation.
  This marks dispatch, NOT the first model token. Allocation and prefill follow.
- delta: request_id, zero-based index, content; only for stream:true. UTF-8 may
  span tokens, so a delta can combine tokens. Incomplete final UTF-8 returns error.
- result: terminal success, full answer, finish_reason and usage. Do NOT append
  answer again if you already displayed deltas. finish_reason:length is truncation.
- error: terminal rejection/failure, error.code and error.message. Partial deltas
  may precede a generation failure; discard/mark the partial answer as incomplete.
- pong: handled by the reader, without waiting for inference or a free worker.

If the waiting queue is full, a new chat gets error.code=busy. Admission timing
is scheduler-dependent: a burst can fill the queue before idle workers dequeue.
There is no queued acknowledgement. Ordering between different IDs is undefined;
within a job, started precedes deltas and exactly one normal terminal result/error.
On EOF, accepted jobs are drained and all threads joined before model destruction.
A slow/full stdout pipe applies backpressure to all output, including pong. This
is not a real-time ping guarantee. Keep reading outputs while sending inputs.

## Limits and integration

No HTTP listener or broker connection is created. messages use the built-in
llama_chat_apply_template, NOT full Jinja. Unsupported templates return errors;
use a fully formatted prompt as an escape hatch. Thinking kwargs are rejected.
/no_think is only a model instruction; think tags are not removed or parsed.
No cancellation, per-job deadline, vision, tools, persistent sessions, durable
queue, ActiveMQ adapter or HTTP adapter is implemented. The ILlmTransport header
remains an interface sketch; AiElbHelper has not been changed by this commit.
A broken stdout/SIGPIPE or fatal backend failure may terminate the process.
Untrusted workloads should be isolated/limited by the parent application.

## Validation required

This change has not been compiled or benchmarked in the tool environment.
Test A/B streaming with WORKERS=2: both started events before the first result;
look for overlapping first-delta to terminal intervals, not just started events.
Also test immediate ping, malformed JSON, overflow, queue saturation (busy),
duplicate active IDs, non-streaming, EOF drain, failed model load and closed pipes.
Measure RSS and compare throughput/latency for 1x8 versus 2x4 threads on your VM.
