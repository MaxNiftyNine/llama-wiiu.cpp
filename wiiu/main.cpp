#include "llama.h"

#include <coreinit/launch.h>
#include <coreinit/thread.h>
#include <coreinit/time.h>
#include <dirent.h>
#include <cstdio>
#include <cstdarg>
#include <sys/stat.h>
#include <unistd.h>
#include <sysapp/launch.h>
#include <sndcore2/core.h>
#if __has_include(<rpxloader/rpxloader.h>)
#define HAVE_RPXLOADER 1
#include <rpxloader/rpxloader.h>
#endif
#include <vpad/input.h>
#include <whb/log.h>
#include <whb/log_console.h>
#include <whb/proc.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>

#include <nn/swkbd.h>
#include <coreinit/filesystem.h>
#include <coreinit/memdefaultheap.h>
#include <whb/gfx.h>
namespace {

//CONFIG ######################
const int n_predict = 32; // cap total tokens
std::string textInput = "Hi!";
const bool use_buggy_keyboard = false; // set to true to use the software keyboard but it causes issues.
//#############################




struct ModelList {
    std::vector<std::string> entries;
    size_t selected = 0;
};

static std::vector<std::string> g_debug_lines;
static FILE * g_log_file = nullptr;

void log_open() {
    // Ensure directory exists
    mkdir("/vol/external01/llama", 0777);
    g_log_file = fopen("/vol/external01/lama-wiiu.log", "a");
    if (!g_log_file) {
        WHBLogPrintf("Could not open log file on SD");
    }
}

void log_close() {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = nullptr;
    }
}

void log_flush() {
    if (g_log_file) {
        fflush(g_log_file);
        const int fd = fileno(g_log_file);
        if (fd >= 0) {
            fsync(fd);
        }
    }
}

void log_line(const char * fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_file) {
        va_start(args, fmt);
        vfprintf(g_log_file, fmt, args);
        fputc('\n', g_log_file);
        va_end(args);
    }
}

void push_debug_line(const std::string & line) {
    constexpr size_t kMaxLines = 16;
    g_debug_lines.push_back(line);
    if (g_debug_lines.size() > kMaxLines) {
        g_debug_lines.erase(g_debug_lines.begin(), g_debug_lines.begin() + (g_debug_lines.size() - kMaxLines));
    }
}

ModelList load_models(const char *root) {
    ModelList list;
    DIR *dir = opendir(root);
    if (!dir) {
        return list;
    }

    while (auto *entry = readdir(dir)) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        // pick a few likely formats
        if (name.size() > 5) {
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.rfind(".gguf") != std::string::npos || lower.rfind(".bin") != std::string::npos) {
                list.entries.push_back(std::string(root) + "/" + name);
            }
        }
    }
    closedir(dir);

    return list;
}

void clear_console() {
    // Re-init console to clear prior lines
    WHBLogConsoleFree();
    WHBLogConsoleInit();
    WHBLogConsoleSetColor(0x000000FF); // white on black
}

struct FileInfo {
    bool exists = false;
    off_t size  = 0;
};

struct ProgressState {
    uint32_t last_pct = 0;
};

FileInfo get_file_info(const std::string & path) {
    struct stat st {};
    FileInfo info;
    if (stat(path.c_str(), &st) == 0) {
        info.exists = true;
        info.size   = st.st_size;
    }
    return info;
}

void whb_llama_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    const char * lvl = "DBG";
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: lvl = "ERR"; break;
        case GGML_LOG_LEVEL_WARN:  lvl = "WRN"; break;
        case GGML_LOG_LEVEL_INFO:  lvl = "INF"; break;
        default: break;
    }
    log_line("[llama %s] %s", lvl, text ? text : "");
    if (text) {
        push_debug_line(std::string(lvl) + ": " + text);
    }
}

void unmount_current_bundle() {
#ifdef HAVE_RPXLOADER
    RPXLoaderStatus res = RPXLoader_InitLibrary();
    if (res == RPX_LOADER_RESULT_SUCCESS) {
        res = RPXLoader_UnmountCurrentRunningBundle();
        if (res != RPX_LOADER_RESULT_SUCCESS) {
            WHBLogPrintf("RPX unmount failed: %d %s", res, RPXLoader_GetStatusStr(res));
        }
        RPXLoader_DeInitLibrary();
    } else {
    WHBLogPrintf("RPX init failed: %d %s", res, RPXLoader_GetStatusStr(res));
    }
#else
    //WHBLogPrintf("RPXLoader not available; skipping unmount.");
#endif
}

bool progress_cb(float progress, void * user) {
    auto * state = static_cast<ProgressState *>(user);
    const uint32_t pct = (uint32_t)(progress * 100.0f);
    if (pct != state->last_pct) {
        state->last_pct = pct;
        log_line("Loading model... %u%%", pct);
        push_debug_line("Load " + std::to_string(pct) + "%");
        WHBLogPrintf("Loading model... %u%%", pct);
        WHBLogConsoleDraw();
        log_flush(); // persist progress updates; rare so shouldn't impact perf
    }
    return WHBProcIsRunning();
}
} // namespace

struct LlamaSession {
    std::string model_path;
    llama_model * model  = nullptr;
    llama_context * ctx  = nullptr;
};

static void free_session(LlamaSession & s) {
    if (s.ctx) {
        llama_free(s.ctx);
        s.ctx = nullptr;
    }
    if (s.model) {
        llama_model_free(s.model);
        s.model = nullptr;
    }
    s.model_path.clear();
}

static bool ensure_session(LlamaSession & s, const std::string & path, std::string & status) {
    if (s.model && s.ctx && s.model_path == path) {
        return true;
    }

    const FileInfo finfo = get_file_info(path);
    if (!finfo.exists) {
        status = "Model not found: " + path;
        WHBLogPrintf("%s", status.c_str());
        return false;
    }

    free_session(s);
    status = "Loading model...";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap  = false;
    mparams.use_mlock = false;
    mparams.use_extra_bufts = false; // avoid extra repack buffers on low-memory Wii U
    mparams.no_host   = true;  // skip host shadow buffers to save RAM
    ProgressState pstate{};
    mparams.progress_callback = progress_cb;
    mparams.progress_callback_user_data = &pstate;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = 64;  // shrink KV cache and attention work
    cparams.n_batch        = 64;
    cparams.n_ubatch       = 64;
    cparams.n_threads      = 3; // try to use all 3 cores
    cparams.n_threads_batch= 3;
    cparams.offload_kqv    = false;
    cparams.op_offload     = false;
    cparams.flash_attn_type= LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.swa_full       = false; // keep memory footprint tiny

    WHBLogPrintf("Loading model: %s (%lld bytes)", path.c_str(), (long long) finfo.size);
    WHBLogConsoleDraw();

    s.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!s.model) {
        status = "Failed to load model (see log)";
        WHBLogPrintf("Load failed for %s", path.c_str());
        return false;
    }
    s.ctx = llama_init_from_model(s.model, cparams);
    if (!s.ctx) {
        status = "Failed to create context";
        free_session(s);
        return false;
    }

    s.model_path = path;
    status = "Model loaded.";
    return true;
}

static std::string tokens_to_text(const llama_model * model, const std::vector<llama_token> & tokens) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::string out;
    char buf[256];
    bool first = true;
    for (llama_token t : tokens) {
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) {
            // insert a space between decoded pieces when appropriate to avoid glued output,
            // but don't add spaces before punctuation
            const bool needs_space = !first && out.size() && !isspace((unsigned char)out.back()) && buf[0] != ' ' && buf[0] != '\n';
            if (needs_space && !ispunct((unsigned char) buf[0]) && buf[0] != '\'') {
                out.push_back(' ');
            }
            out.append(buf, buf + n);
            first = false;
        }
    }
    return out;
}

static llama_token pick_top_non_eos(const float * logits, int32_t n_vocab, llama_token eos) {
    if (!logits || n_vocab <= 0) return eos;
    llama_token best = 0;
    float best_logit = -1e30f;
    for (int32_t t = 0; t < n_vocab; ++t) {
        if (t == eos) continue;
        const float v = logits[t];
        if (v > best_logit) {
            best_logit = v;
            best = t;
        }
    }
    return best;
}

static std::string build_prompt(const std::string & user) {
    std::ostringstream ss;
    ss << "### Instruction:\n"
       << "You are a helpful assistant. Reply conversationally and directly.\n\n"
       << "### Input:\n"
       << user << "\n\n"
       << "### Response:\n";
    return ss.str();
}


static std::string char16_to_string(const char16_t* src)
{
    std::string out;
    if (!src) return out;

    for (int i = 0; src[i] != 0; ++i)
    {
        if (src[i] <= 0x7F)
            out.push_back(static_cast<char>(src[i]));
        else
            out.push_back('?'); // non-ASCII fallback
    }
    return out;
}

std::string get_keyboard_input(const char16_t* hintText)
{
    std::string result;

    // --- FSClient for swkbd ---
    FSClient* fsClient =
        (FSClient*)MEMAllocFromDefaultHeap(sizeof(FSClient));
    FSAddClient(fsClient, FS_ERROR_FLAG_NONE);

    // --- Create keyboard ---
    nn::swkbd::CreateArg createArg;
    createArg.regionType = nn::swkbd::RegionType::Europe;
    createArg.workMemory =
        MEMAllocFromDefaultHeap(nn::swkbd::GetWorkMemorySize(0));
    createArg.fsClient = fsClient;

    if (!nn::swkbd::Create(createArg))
        return result;

    nn::swkbd::MuteAllSound(false);

    // --- Show keyboard ---
    nn::swkbd::AppearArg appearArg;
    appearArg.keyboardArg.configArg.languageType =
        nn::swkbd::LanguageType::English;
    appearArg.inputFormArg.hintText = hintText;

    if (!nn::swkbd::AppearInputForm(appearArg))
        return result;

    // --- Keyboard loop ---
    while (WHBProcIsRunning())
    {
        VPADStatus vpad;
        VPADRead(VPAD_CHAN_0, &vpad, 1, nullptr);
        VPADGetTPCalibratedPoint(
            VPAD_CHAN_0, &vpad.tpNormal, &vpad.tpNormal);

        nn::swkbd::ControllerInfo controller{};
        controller.vpad = &vpad;

        nn::swkbd::Calc(controller);

        if (nn::swkbd::IsNeedCalcSubThreadFont())
            nn::swkbd::CalcSubThreadFont();

        if (nn::swkbd::IsNeedCalcSubThreadPredict())
            nn::swkbd::CalcSubThreadPredict();

        // OK pressed
        if (nn::swkbd::IsDecideOkButton(nullptr))
        {
            nn::swkbd::DisappearInputForm();
            break;
        }

        // Draw
        WHBGfxBeginRender();
        WHBGfxBeginRenderTV();
        nn::swkbd::DrawTV();
        WHBGfxFinishRenderTV();

        WHBGfxBeginRenderDRC();
        nn::swkbd::DrawDRC();
        WHBGfxFinishRenderDRC();
        WHBGfxFinishRender();
    }

    // --- Get typed text ---
    const char16_t* text = nn::swkbd::GetInputFormString();
    result = char16_to_string(text);

    // --- Cleanup ---
    nn::swkbd::Destroy();
    MEMFreeToDefaultHeap(createArg.workMemory);
    FSDelClient(fsClient, FS_ERROR_FLAG_NONE);
    MEMFreeToDefaultHeap(fsClient);
    return result;
}
static bool generate_simple(LlamaSession & s, const std::string & prompt, std::string & response, std::string & status, size_t & tokens_generated) {
    response.clear();
    tokens_generated = 0;
    if (!s.model || !s.ctx) {
        status = "No model loaded.";
        return false;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(s.model);
    std::vector<llama_token> tokens(256);
    const int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false);
    if (n_tokens <= 0) {
        status = "Tokenization failed.";
        return false;
    }
    tokens.resize(n_tokens);
    log_line("Prompt tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        const char * ttext = llama_vocab_get_text(vocab, tokens[i]);
        log_line("  %d: %s", tokens[i], ttext ? ttext : "<null>");
        push_debug_line(ttext ? ttext : "<tok>");
    }

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = (i == n_tokens - 1);
    }
    if (llama_decode(s.ctx, batch) != 0) {
        llama_batch_free(batch);
        status = "Decode failed.";
        return false;
    }
    llama_batch_free(batch);

    auto chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<llama_token> generated;
    const int min_gen = 1;    // force at least token before stopping
    const llama_token eos = llama_vocab_eos(vocab);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_predict; ++i) {
        float * logits = llama_get_logits_ith(s.ctx, -1);
        if (logits && generated.size() < (size_t) min_gen && eos >= 0 && eos < n_vocab) {
            logits[eos] = -1e9f; // suppress early EOS
        }

        const llama_token tok = llama_sampler_sample(smpl, s.ctx, -1);
        llama_token chosen = tok;
        if (tok == eos && (int)generated.size() < min_gen) {
            chosen = pick_top_non_eos(logits, n_vocab, eos);
            log_line("EOS sampled early; forcing token %d", chosen);
            push_debug_line("force non-EOS");
        }

        if (llama_vocab_is_eog(vocab, chosen)) {
            log_line("Hit EOG token %d, stopping.", chosen);
            push_debug_line("stop: EOG");
            break;
        }
        generated.push_back(chosen);
        tokens_generated++;

        // log decoded piece
        char piece[256];
        const int n = llama_token_to_piece(vocab, chosen, piece, sizeof(piece), 0, true);
        if (n > 0) {
            piece[n] = '\0';
            log_line("Gen %d: id=%d text=%s", i, chosen, piece);
            push_debug_line(piece);
        } else {
            log_line("Gen %d: id=%d", i, chosen);
            push_debug_line("id " + std::to_string(chosen));
        }

        llama_batch next = llama_batch_init(1, 0, 1);
        next.n_tokens   = 1;
        next.token[0]    = chosen;
        next.pos[0]      = n_tokens + i;
        next.n_seq_id[0] = 1;
        next.seq_id[0][0]= 0;
        next.logits[0]   = 1;
        if (llama_decode(s.ctx, next) != 0) {
            llama_batch_free(next);
            status = "Decode failed mid-gen";
            break;
        }
        llama_batch_free(next);
    }

    llama_sampler_free(smpl);

    response = tokens_to_text(s.model, generated);
    if (response.empty()) {
        status = "Generated empty response (EOS?)";
    } else {
        status = "Done.";
    }
    return true;
}

int main() {
    WHBProcInit();
    WHBLogConsoleInit();
    WHBLogConsoleSetColor(0x000000FF); // white on black
    log_open();
    llama_log_set(whb_llama_log, nullptr);
    AXInit(); // stop menu music; we don't play audio but init/quit for hygiene
    //unmount_current_bundle();
    WHBGfxInit();
    FSInit();
    VPADInit();

    ModelList models = load_models("/vol/external01/model");
    LlamaSession session;
    std::string status;
    std::string last_response;

    clear_console();

    if (models.entries.empty()) {
        WHBLogPrintf("Response: No model found in /vol/external01/model");
        WHBLogConsoleDraw();
        while (WHBProcIsRunning()) {
            OSSleepTicks(OSMillisecondsToTicks(50));
        }
        goto cleanup;
    }

    // Always use the first discovered model.
    models.selected = 0;

    status = "Loading Model...";
    WHBLogPrintf("%s", status.c_str());
    WHBLogConsoleDraw();

    if (ensure_session(session, models.entries[models.selected], status)) {
        status = "Generating...";
        clear_console();
        WHBLogPrintf("%s", status.c_str());
        WHBLogConsoleDraw();

        size_t tokens_generated = 0;
        const uint64_t t_start = OSGetTime();
        if (use_buggy_keyboard) {
            textInput = get_keyboard_input(u"Enter your prompt");
        }
        clear_console();
        WHBLogConsoleSetColor(0x000000FF);
        WHBLogConsoleDraw();

        const std::string prompt = build_prompt(textInput);
        if (generate_simple(session, prompt, last_response, status, tokens_generated)) {
            const uint64_t t_end = OSGetTime();
            const uint64_t ms = OSTicksToMilliseconds(t_end - t_start);
            const double secs = ms > 0 ? (double) ms / 1000.0 : 0.0;
            const double tps = secs > 0.0 ? (double) tokens_generated / secs : 0.0;
            clear_console();
            WHBLogPrintf("Response:");
            if (last_response.empty()) {
                WHBLogPrintf("  <empty>");
            } else {
                const size_t wrap = 68;
                size_t start = 0;
                while (start < last_response.size()) {
                    size_t len = std::min(wrap, last_response.size() - start);
                    WHBLogPrintf("  %s", last_response.substr(start, len).c_str());
                    start += len;
                }
            }
            WHBLogPrintf("");
            WHBLogPrintf("Tokens/sec: %.2f (%zu in %llu ms)", tps, tokens_generated, (unsigned long long) ms);
        } else {
            clear_console();
            WHBLogPrintf("Response: %s", status.c_str());
        }
    } else {
        clear_console();
        WHBLogPrintf("Response: %s", status.c_str());
    }

    WHBLogConsoleDraw();

    while (WHBProcIsRunning()) {

        VPADStatus vpad{};
        VPADRead(VPAD_CHAN_0, &vpad, 1, nullptr);

        if (vpad.trigger & VPAD_BUTTON_PLUS) {
            SYSLaunchMenu();  // triggers ProcUI event
        }

        // Keep drawing so GX2 state stays sane
        WHBLogConsoleDraw();
    }


cleanup:
    free_session(session);
    AXQuit();
    log_close();
    llama_backend_free();
    WHBLogConsoleFree();
    WHBProcShutdown();
    return 0;
}
