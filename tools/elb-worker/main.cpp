#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace {
constexpr size_t max_line = 1024 * 1024;
void emit(const json & value) {
    std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
    if (!std::cout) throw std::runtime_error("stdout closed");
}
void log_to_stderr(enum ggml_log_level, const char * text, void *) {
    std::fputs(text, stderr);
}
// Read an entire oversized line without retaining it, preserving framing.
bool read_line(std::string & line, bool & oversized) {
    line.clear(); oversized = false;
    char ch;
    bool any = false;
    while (std::cin.get(ch)) {
        any = true;
        if (ch == '\n') return true;
        if (line.size() < max_line) line.push_back(ch);
        else oversized = true;
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
        throw std::runtime_error("messages must be a nonempty array of at most 128 items");
    std::vector<std::string> roles, contents;
    for (const auto & item : input) {
        const auto role = item.at("role").get<std::string>();
        if (role != "system" && role != "user" && role != "assistant")
            throw std::runtime_error("unsupported message role");
        roles.push_back(role);
        contents.push_back(item.at("content").get<std::string>());
    }
    std::vector<llama_chat_message> messages;
    for (size_t i = 0; i < roles.size(); ++i)
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("model has no chat template; supply a formatted prompt");
    std::vector<char> output(4096);
    int n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                      output.data(), output.size());
    if (n < 0) throw std::runtime_error("unsupported built-in chat template; supply a formatted prompt");
    if (static_cast<size_t>(n) > output.size()) {
        output.resize(n);
        n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                      output.data(), output.size());
        if (n < 0 || static_cast<size_t>(n) > output.size())
            throw std::runtime_error("chat template failed");
    }
    return std::string(output.data(), n);
}
json generate(llama_model * model, const json & request, int ctx_size, int threads) {
    const bool streaming = request.value("stream", false);
    const int limit = request.value("max_tokens", 256);
    const double temp = request.value("temperature", 0.2);
    if (limit < 1 || limit > 4096 || !std::isfinite(temp) || temp < 0 || temp > 2)
        throw std::runtime_error("invalid max_tokens [1,4096] or temperature [0,2]");
    // The built-in template API has no Jinja kwargs. Do not silently ignore them.
    if (request.contains("enable_thinking") || request.contains("chat_template_kwargs"))
        throw std::runtime_error("thinking kwargs are not supported in v1; use a fully formatted prompt");
    const std::string prompt = format_prompt(model, request);
    if (prompt.empty() || prompt.size() > max_line) throw std::runtime_error("invalid prompt size");
    const auto * vocab = llama_model_get_vocab(model);
    const int needed = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (needed <= 0 || needed + limit > ctx_size)
        throw std::runtime_error("context overflow or empty tokenized prompt");
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != needed)
        throw std::runtime_error("tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = ctx_size; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.offload_kqv = false; cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    if (!ctx) throw std::runtime_error("context allocation failed");
    if (tokens.size() + limit > llama_n_ctx(ctx.get())) throw std::runtime_error("actual context too small");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("batch allocation failed");
    for (int pos = 0; pos < needed;) {
        const int count = std::min(512, needed - pos);
        evaluate(ctx.get(), batch.get(), tokens.data() + pos, count, pos);
        pos += count;
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
    std::string answer, reason = "length";
    std::string pending_utf8;
    int delta_index = 0;
    int generated = 0;
    for (int i = 0; i < limit; ++i) {
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

            // A token can contain only part of a UTF-8 character.
            // Emit only when the accumulated fragment is valid UTF-8.
            bool valid_utf8 = true;
            try {
                const auto checked = json(pending_utf8).dump();
                (void) checked;
            } catch (const json::type_error &) {
                valid_utf8 = false;
            }

            if (valid_utf8 && !pending_utf8.empty()) {
                emit({
                    {"version", 1},
                    {"type", "delta"},
                    {"request_id", request.at("request_id")},
                    {"index", delta_index++},
                    {"content", pending_utf8}
                });
                pending_utf8.clear();
            }
        }

        ++generated;
        if (i + 1 < limit) evaluate(ctx.get(), batch.get(), &token, 1, needed + i);
    }
    if (streaming && !pending_utf8.empty()) {
        throw std::runtime_error(
            "generation ended with an incomplete or invalid UTF-8 fragment"
        );
    }

    return {{"version", 1}, {"type", "result"}, {"success", true},
        {"request_id", request.at("request_id")}, {"answer", answer}, {"finish_reason", reason},
        {"usage", {{"prompt_tokens", needed}, {"completion_tokens", generated}, {"total_tokens", needed + generated}}}};
}
}
int main(int argc, char ** argv) {
    llama_log_set(log_to_stderr, nullptr);
    try {
        if (argc < 2 || argc > 4) throw std::runtime_error("usage: llama-worker MODEL.gguf [CTX=4096] [THREADS=4]");
        const int ctx_size = argc > 2 ? std::stoi(argv[2]) : 4096;
        const int threads = argc > 3 ? std::stoi(argv[3]) : 4;
        if (ctx_size < 128 || ctx_size > 131072 || threads < 1 || threads > 1024)
            throw std::runtime_error("invalid context or thread count");
        llama_backend_init();
        struct BackendGuard { ~BackendGuard() { llama_backend_free(); } } guard;
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        if (llama_model_has_encoder(model.get())) throw std::runtime_error("only decoder-only text models supported");
        emit({{"version", 1}, {"type", "ready"}});
        std::string line; bool oversized;
        while (read_line(line, oversized)) {
            if (line.empty() && !oversized) continue;
            json id = nullptr;
            try {
                if (oversized) throw std::runtime_error("line exceeds 1 MiB");
                const json request = json::parse(line);
                if (!request.is_object()) throw std::runtime_error("request must be an object");
                if (request.contains("request_id") && request["request_id"].is_string()) id = request["request_id"];
                if (id.is_null() || id.get<std::string>().empty()) throw std::runtime_error("request_id must be a nonempty string");
                if (request.value("version", 0) != 1) throw std::runtime_error("unsupported protocol version");
                const auto type = request.value("type", std::string());
                if (type == "ping") emit({{"version", 1}, {"type", "pong"}, {"request_id", id}});
                else if (type == "chat") emit(generate(model.get(), request, ctx_size, threads));
                else throw std::runtime_error("unsupported request type");
            } catch (const std::exception & e) {
                emit({{"version", 1}, {"type", "error"}, {"request_id", id}, {"success", false},
                      {"error", {{"code", "request_failed"}, {"message", e.what()}}}});
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-worker: %s\n", e.what());
        return 1;
    }
    return 0;
}
