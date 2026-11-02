#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
namespace {
constexpr size_t max_line = 1024 * 1024;
std::mutex output_mutex, context_init_mutex;
std::atomic<bool> output_failed{false};
void emit(const json & value) {
    const std::string line = value.dump(-1, ' ', false, json::error_handler_t::replace);
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << line << '\n' << std::flush;
    if (!std::cout) {
        output_failed = true;
        throw std::runtime_error("stdout closed");
    }
}
void error(const json & id, const std::string & code, const std::string & message) {
    emit({{"version", 1}, {"type", "error"}, {"request_id", id}, {"success", false},
          {"error", {{"code", code}, {"message", message}}}});
}
void log_to_stderr(enum ggml_log_level, const char * text, void *) { std::fputs(text, stderr); }
bool read_line(std::string & line, bool & oversized) {
    line.clear(); oversized = false;
    char ch; bool any = false;
    while (std::cin.get(ch)) {
        any = true;
        if (ch == '\n') return true;
        if (line.size() < max_line) line.push_back(ch); else oversized = true;
    }
    return any;
}
void evaluate(llama_context * ctx, llama_batch_ext * batch,
              const llama_token * tokens, int count, llama_pos start) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = start + i;
        llama_batch_ext_set_pos(batch, index, &pos);
    }
    llama_batch_ext_set_output_logits(batch, count - 1, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
        throw std::runtime_error("decode failed");
}
std::string format_prompt(llama_model * model, const json & request) {
    if (request.contains("prompt")) return request.at("prompt").get<std::string>();
    const auto & input = request.at("messages");
    if (!input.is_array() || input.empty() || input.size() > 128)
        throw std::runtime_error("messages must contain 1..128 items");
    std::vector<std::string> roles, contents;
    for (const auto & item : input) {
        const auto role = item.at("role").get<std::string>();
        if (role != "system" && role != "user" && role != "assistant")
            throw std::runtime_error("unsupported message role");
        roles.push_back(role); contents.push_back(item.at("content").get<std::string>());
    }
    std::vector<llama_chat_message> messages;
    for (size_t i = 0; i < roles.size(); ++i) messages.push_back({roles[i].c_str(), contents[i].c_str()});
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("model has no chat template; supply formatted prompt");
    std::vector<char> output(4096);
    int n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, output.data(), output.size());
    if (n < 0) throw std::runtime_error("unsupported built-in chat template; supply formatted prompt");
    if (static_cast<size_t>(n) > output.size()) {
        output.resize(n);
        n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, output.data(), output.size());
        if (n < 0 || static_cast<size_t>(n) > output.size()) throw std::runtime_error("chat template failed");
    }
    return std::string(output.data(), n);
}
json generate(llama_model * model, const json & request, int ctx_size, int threads) {
    const bool streaming = request.value("stream", false);
    const int limit = request.value("max_tokens", 256);
    const double temp = request.value("temperature", 0.2);
    if (limit < 1 || limit > 4096 || !std::isfinite(temp) || temp < 0 || temp > 2)
        throw std::runtime_error("invalid max_tokens [1,4096] or temperature [0,2]");
    if (request.contains("enable_thinking") || request.contains("chat_template_kwargs"))
        throw std::runtime_error("thinking kwargs unsupported; use formatted prompt");
    const auto prompt = format_prompt(model, request);
    if (prompt.empty() || prompt.size() > max_line) throw std::runtime_error("invalid prompt size");
    const auto * vocab = llama_model_get_vocab(model);
    const int needed = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (needed <= 0 || needed + limit > ctx_size) throw std::runtime_error("context overflow or empty prompt");
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != needed)
        throw std::runtime_error("tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = ctx_size; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.offload_kqv = false; cp.op_offload = false;
    // Serialize allocation/setup only, not prefill or generation.
    llama_context * raw_ctx;
    {
        std::lock_guard<std::mutex> lock(context_init_mutex);
        raw_ctx = llama_init_from_model(model, cp);
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(raw_ctx, llama_free);
    if (!ctx) throw std::runtime_error("context allocation failed");
    if (tokens.size() + limit > llama_n_ctx(ctx.get())) throw std::runtime_error("actual context too small");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("batch allocation failed");
    for (int pos = 0; pos < needed;) {
        const int count = std::min(512, needed - pos);
        evaluate(ctx.get(), batch.get(), tokens.data() + pos, count, pos); pos += count;
    }
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (!sampler) throw std::runtime_error("sampler allocation failed");
    if (temp == 0) llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(static_cast<float>(temp)));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    std::string answer, pending_utf8, reason = "length";
    int generated = 0, delta_index = 0;
    for (int i = 0; i < limit; ++i) {
        if (output_failed.load()) throw std::runtime_error("output unavailable");
        const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) { reason = "stop"; break; }
        std::vector<char> piece(256);
        int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
        if (n < 0) {
            piece.resize(-n);
            n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
        }
        if (n < 0) throw std::runtime_error("detokenization failed");
        answer.append(piece.data(), n);
        if (streaming) {
            pending_utf8.append(piece.data(), n);
            bool valid = true;
            try { const auto checked = json(pending_utf8).dump(); (void) checked; }
            catch (const json::type_error &) { valid = false; }
            if (valid && !pending_utf8.empty()) {
                emit({{"version", 1}, {"type", "delta"}, {"request_id", request.at("request_id")},
                      {"index", delta_index++}, {"content", pending_utf8}});
                pending_utf8.clear();
            }
        }
        ++generated;
        if (i + 1 < limit) evaluate(ctx.get(), batch.get(), &token, 1, needed + i);
    }
    if (streaming && !pending_utf8.empty()) throw std::runtime_error("incomplete or invalid UTF-8 fragment");
    return {{"version", 1}, {"type", "result"}, {"success", true},
        {"request_id", request.at("request_id")}, {"answer", answer}, {"finish_reason", reason},
        {"usage", {{"prompt_tokens", needed}, {"completion_tokens", generated}, {"total_tokens", needed + generated}}}};
}
class WorkerPool {
    llama_model * model_;
    int ctx_, threads_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<json> queue_;
    std::set<std::string> pending_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
    const std::chrono::steady_clock::time_point epoch_ = std::chrono::steady_clock::now();
    void run(int slot) noexcept {
        for (;;) {
            json request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;
                request = std::move(queue_.front()); queue_.pop_front();
            }
            const auto id = request.at("request_id").get<std::string>();
            try {
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - epoch_).count();
                emit({{"version", 1}, {"type", "started"}, {"request_id", id},
                      {"slot", slot}, {"elapsed_ms", ms}});
                emit(generate(model_, request, ctx_, threads_));
            } catch (const std::exception & e) {
                try { error(id, "request_failed", e.what()); }
                catch (...) { output_failed = true; }
            } catch (...) {
                try { error(id, "request_failed", "unknown inference failure"); }
                catch (...) { output_failed = true; }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending_.erase(id);
            }
        }
    }
public:
    WorkerPool(llama_model * model, int ctx, int threads, int count, size_t capacity)
        : model_(model), ctx_(ctx), threads_(threads), capacity_(capacity) {
        try {
            for (int i = 0; i < count; ++i) workers_.emplace_back([this, i] { run(i); });
        } catch (...) { close(); throw; }
    }
    ~WorkerPool() { close(); }
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool & operator=(const WorkerPool &) = delete;
    // Empty result means admitted; queue capacity excludes running jobs.
    std::string submit(json request) {
        const auto id = request.at("request_id").get<std::string>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return "shutting_down";
            if (pending_.count(id)) return "duplicate_request_id";
            if (queue_.size() >= capacity_) return "busy";
            pending_.insert(id);
            try { queue_.push_back(std::move(request)); }
            catch (...) { pending_.erase(id); throw; }
        }
        cv_.notify_one();
        return {};
    }
    void close() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto & thread : workers_) if (thread.joinable()) thread.join();
    }
};
}
int main(int argc, char ** argv) {
    llama_log_set(log_to_stderr, nullptr);
    try {
        if (argc < 2 || argc > 6)
            throw std::runtime_error("usage: llama-worker MODEL [CTX=4096] [THREADS_PER_JOB=4] [WORKERS=1] [QUEUE=16]");
        const int ctx = argc > 2 ? std::stoi(argv[2]) : 4096;
        const int threads = argc > 3 ? std::stoi(argv[3]) : 4;
        const int workers = argc > 4 ? std::stoi(argv[4]) : 1;
        const int capacity = argc > 5 ? std::stoi(argv[5]) : 16;
        if (ctx < 128 || ctx > 131072 || threads < 1 || threads > 1024 ||
            workers < 1 || workers > 16 || capacity < 1 || capacity > 256)
            throw std::runtime_error("invalid context, threads, workers [1,16] or queue [1,256]");
        llama_backend_init();
        struct BackendGuard { ~BackendGuard() { llama_backend_free(); } } guard;
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        if (llama_model_has_encoder(model.get())) throw std::runtime_error("only decoder-only text models supported");
        WorkerPool pool(model.get(), ctx, threads, workers, capacity);
        emit({{"version", 1}, {"type", "ready"}, {"workers", workers},
              {"threads_per_job", threads}, {"queue_capacity", capacity}});
        std::string line; bool oversized;
        while (!output_failed.load() && read_line(line, oversized)) {
            if (line.empty() && !oversized) continue;
            json id = nullptr;
            try {
                if (oversized) throw std::runtime_error("line exceeds 1 MiB");
                json request = json::parse(line);
                if (!request.is_object()) throw std::runtime_error("request must be an object");
                if (request.contains("request_id") && request["request_id"].is_string()) id = request["request_id"];
                if (id.is_null() || id.get<std::string>().empty() || id.get<std::string>().size() > 128)
                    throw std::runtime_error("request_id must be a string of 1..128 bytes");
                if (request.value("version", 0) != 1) throw std::runtime_error("unsupported protocol version");
                const auto type = request.value("type", std::string());
                if (type == "ping") emit({{"version", 1}, {"type", "pong"}, {"request_id", id}});
                else if (type == "chat") {
                    const auto code = pool.submit(std::move(request));
                    if (!code.empty()) error(id, code, "request was not admitted");
                } else throw std::runtime_error("unsupported request type");
            } catch (const std::exception & e) { error(id, "invalid_request", e.what()); }
        }
        // Drain accepted jobs before freeing model/backend, including on stdin EOF.
        pool.close();
        if (output_failed.load()) return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-worker: %s\n", e.what()); return 1;
    }
    return 0;
}
