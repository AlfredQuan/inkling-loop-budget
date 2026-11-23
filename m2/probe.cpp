// Read fixed-depth logits and a first-pass state through the public llama.cpp API.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int MAX_TOKENS = 512;
static constexpr int PROJECTION = 96;

struct State {
    std::vector<float> projected_input;
    int captures = 0;
};

static bool capture(ggml_tensor * t, bool ask, void * user) {
    const bool wanted = std::strcmp(t->name, "result_norm") == 0;
    if (ask || !wanted) {
        return wanted;
    }
    auto & state = *static_cast<State *>(user);
    ++state.captures;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return false;
    }
    state.projected_input.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, state.projected_input.data(), 0, ggml_nbytes(t));
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s, bool special) {
    std::vector<llama_token> ids(s.size() + 16);
    int n = llama_tokenize(vocab, s.data(), s.size(), ids.data(), ids.size(), false, special);
    if (n <= 0 || n > MAX_TOKENS) {
        throw std::runtime_error("Invalid prompt token count");
    }
    ids.resize(n);
    return ids;
}

static uint8_t hex_digit(char c) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    throw std::runtime_error("Invalid hex input");
}

static std::string decode_hex(const std::string & hex) {
    if (hex.empty() || hex.size() % 2) throw std::runtime_error("Invalid hex length");
    std::string text;
    text.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        text.push_back(char(hex_digit(hex[i]) * 16 + hex_digit(hex[i + 1])));
    }
    return text;
}

struct Example {
    std::string name;
    std::vector<llama_token> tokens;
};

struct Batch {
    explicit Batch(int count) : data(llama_batch_init(count, 0, 1)) {}
    ~Batch() { llama_batch_free(data); }
    llama_batch data;
};

static void emit(const std::vector<float> & xs, std::ostream & dst) {
    dst << '[';
    bool first = true;
    for (float x : xs) {
        if (!std::isfinite(x)) throw std::runtime_error("Non-finite tensor");
        if (!first) dst << ',';
        dst << x;
        first = false;
    }
    dst << ']';
}

static int run(int argc, char ** argv) {
    if (argc != 6) throw std::runtime_error("Usage: loop-probe model.gguf depth workers prompts.tsv output.jsonl");
    const int depth = std::stoi(argv[2]);
    const int workers = std::stoi(argv[3]);
    if (depth < 1 || depth > 3 || workers < 1 || workers > 8) throw std::runtime_error("Invalid depth or workers");
    std::ifstream input(argv[4]);
    std::ofstream output(argv[5]);
    if (!input || !output) throw std::runtime_error("Cannot open input or output");
    output << std::setprecision(9);
    ggml_backend_load_all();
    llama_model_kv_override kv[2]{};
    std::strcpy(kv[0].key, "inkling2_loop.loop.count");
    kv[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    kv[0].val_i64 = depth;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.kv_overrides = kv;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model || llama_model_n_layer(model.get()) != 12 + 12 * depth) {
        throw std::runtime_error("Wrong model architecture or effective layer count");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    std::array<llama_token, 4> letters{};
    for (size_t i = 0; i < letters.size(); ++i) {
        const auto ids = tokenize(vocab, std::string(1, 'A' + i), false);
        if (ids.size() != 1) throw std::runtime_error("Answer is not a single token");
        letters[i] = ids[0];
    }
    const int dim = llama_model_n_embd_out(model.get());
    State state;
    auto cp = llama_context_default_params();
    cp.n_ctx = workers * MAX_TOKENS;
    cp.n_batch = workers * MAX_TOKENS;
    cp.n_ubatch = cp.n_batch;
    cp.n_seq_max = workers;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.kv_unified = false;
    if (depth == 1) {
        cp.cb_eval = capture;
        cp.cb_eval_user_data = &state;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context) throw std::runtime_error("Cannot initialize llama context");

    std::vector<float> projection(PROJECTION * dim);
    if (depth == 1) {
        std::mt19937 rng(938537);
        const float scale = 1.0f / std::sqrt(float(dim));
        for (float & x : projection) x = rng() & 1 ? scale : -scale;
    }
    std::vector<Example> examples;
    for (std::string line; std::getline(input, line);) {
        const auto end = line.find('\t');
        if (end == std::string::npos) throw std::runtime_error("Missing TSV separator");
        const auto name = line.substr(0, end);
        if (name.empty() || name.find_first_not_of("0123456789abcdefcanry-ABCD") != std::string::npos) {
            throw std::runtime_error("Invalid example ID");
        }
        examples.push_back({name, tokenize(vocab, decode_hex(line.substr(end + 1)), true)});
    }
    if (examples.empty()) throw std::runtime_error("No examples supplied");
    for (size_t offset = 0; offset < examples.size(); offset += workers) {
        const size_t count = std::min<size_t>(workers, examples.size() - offset);
        size_t padded = 0;
        for (size_t i = 0; i < count; ++i) padded = std::max(padded, examples[offset + i].tokens.size());
        Batch batch(count * padded);
        std::vector<int> last;
        for (size_t i = 0; i < count; ++i) {
            const auto & tokens = examples[offset + i].tokens;
            for (size_t j = 0; j < padded; ++j) {
                int at = batch.data.n_tokens++;
                batch.data.token[at] = j < tokens.size() ? tokens[j] : letters[0];
                batch.data.pos[at] = j;
                batch.data.n_seq_id[at] = 1;
                batch.data.seq_id[at][0] = i;
                batch.data.logits[at] = j + 1 == tokens.size();
                if (batch.data.logits[at]) last.push_back(at);
            }
        }
        llama_memory_clear(llama_get_memory(context.get()), true);
        state.projected_input.clear();
        state.captures = 0;
        if (llama_decode(context.get(), batch.data) != 0) throw std::runtime_error("llama_decode failed");
        if (depth == 1 && (state.captures != 1 || state.projected_input.size() != count * dim)) {
            throw std::runtime_error("Unexpected readout tensor shape or count");
        }
        for (size_t i = 0; i < count; ++i) {
            const auto & ex = examples[offset + i];
            const float * logits = llama_get_logits_ith(context.get(), last.at(i));
            if (!logits) throw std::runtime_error("Missing logits");
            std::vector<float> scores;
            for (auto token : letters) scores.push_back(logits[token]);
            output << "{\"id\":\"" << ex.name << "\",\"depth\":" << depth
                   << ",\"tokens\":" << ex.tokens.size() << ",\"logits\":";
            emit(scores, output);
            if (depth == 1) {
                const float * src = state.projected_input.data() + i * dim;
                std::vector<float> features(PROJECTION);
                for (int j = 0; j < PROJECTION; ++j) {
                    float x = 0;
                    for (int k = 0; k < dim; ++k) x += src[k] * projection[j * dim + k];
                    features[j] = x;
                }
                output << ",\"state\":";
                emit(features, output);
            }
            output << "}\n";
        }
        output.flush();
        if ((offset + count) % 128 == 0 || offset + count == examples.size()) {
            std::cerr << "depth " << depth << ": " << offset + count << "/" << examples.size() << '\n';
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
