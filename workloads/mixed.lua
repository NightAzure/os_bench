-- mixed.lua: wrk workload script for the OS scheduling experiment.
-- Implements the 70/30 two-tier request mix described in the paper.
--
-- Tier A (70%): POST /encode  — CPU-bound, GIL-releasing via BLAS.
--   16 rotating text inputs prevent any request-level caching from
--   eliminating the encoding computation after the first call.
--
-- Tier B (30%): GET /health  — Near-zero processing, I/O floor.
--   Produces voluntary context switches to contrast with Tier A
--   preemptions, giving pidstat -w diagnostic meaning.
--
-- Usage:
--   wrk -t2 -c10 -d120s --latency -s workloads/mixed.lua http://localhost:8000

math.randomseed(42)

local texts = {
    "What is machine learning and how does it work in practice?",
    "Explain the concept of gradient descent optimization step by step.",
    "How do neural networks learn representations from raw data?",
    "Describe the Python Global Interpreter Lock mechanism in CPython.",
    "What are transformer models in natural language processing tasks?",
    "How does backpropagation compute gradients through a neural network?",
    "Explain the self-attention mechanism used in BERT and GPT models.",
    "What is transfer learning and why is it useful for downstream tasks?",
    "Describe the architecture of convolutional neural networks for vision.",
    "How does the Linux CFS scheduler allocate CPU time across processes?",
    "What is CPU affinity and why does it matter for tail latency reduction?",
    "Explain involuntary context switches and their effect on p99 latency.",
    "How do BLAS routines accelerate matrix computations on modern CPUs?",
    "What is the difference between soft and hard CPU affinity in Linux?",
    "Describe cache coherence protocols in multicore processor systems.",
    "How does PyTorch release the GIL during inference and use BLAS threads?",
}

request = function()
    local r = math.random()
    if r < 0.70 then
        -- Tier A: CPU-bound encoding request
        local idx = math.random(#texts)
        local body = string.format('{"text": "%s"}', texts[idx])
        return wrk.format(
            "POST", "/encode",
            {["Content-Type"] = "application/json"},
            body
        )
    else
        -- Tier B: lightweight health check
        return wrk.format("GET", "/health", nil, nil)
    end
end

function done(summary, latency, requests_per_sec)
    -- Use wrk's C-level summary object (more reliable than per-response Lua callbacks,
    -- which are not guaranteed to fire in all wrk builds/thread configurations).
    local total  = summary.requests
    local errors = summary.errors.status  + summary.errors.connect +
                   summary.errors.read    + summary.errors.write   +
                   summary.errors.timeout
    local error_rate = total > 0 and (errors / total) * 100 or 0
    io.write(string.format(
        "[wrk] done: %d requests, %d errors (%.2f%% error rate)\n",
        total, errors, error_rate
    ))
    if error_rate > 0.1 then
        io.write("[wrk] WARNING: error rate > 0.1% — flag this trial for exclusion.\n")
    end
    io.flush()
end
