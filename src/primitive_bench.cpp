#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct Args {
    std::string primitive = "gemv";
    int dim = 128;
    int64_t scale = 1024;
    int64_t db_vectors = 0;
    int64_t fixed_repetitions = 0;
    int threads = 0;
    double target_seconds = 1.0;
    uint64_t seed = 42;
    std::string csv_out;
    std::string json_out;
};

struct Metrics {
    double elapsed_sec = 0.0;
    int64_t repetitions = 0;
    long double flop_per_rep = 0.0;
    long double bytes_per_rep = 0.0;
    long double flop_total = 0.0;
    long double bytes_total = 0.0;
    double ai = 0.0;
    double achieved_gflops = 0.0;
    double achieved_gbs = 0.0;
    double sink = 0.0;
};

static void usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " --primitive <name> [options]\n"
        << "Primitives: gemv, cos_db_db, cos_q_db, ip_q_db, softmax\n"
        << "Options:\n"
        << "  --dim <int>              vector dimension (default: 128)\n"
        << "  --scale <int64>          scaling dimension per primitive\n"
        << "  --db-vectors <int64>     database vector count for random-access primitives\n"
        << "  --fixed-repetitions <n>  run exactly n repetitions (bypass auto-scaling)\n"
        << "  --target-seconds <float> minimum timed runtime (default: 1.0)\n"
        << "  --threads <int>          OMP threads (0 keeps runtime default)\n"
        << "  --seed <int64>           RNG seed (default: 42)\n"
        << "  --csv <path>             append one-row result to CSV\n"
        << "  --json <path>            write one-run metrics as JSON\n"
        << "  --help                   show this message\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto require_value = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (key == "--primitive") {
            args.primitive = require_value("--primitive");
        } else if (key == "--dim") {
            args.dim = std::stoi(require_value("--dim"));
        } else if (key == "--scale") {
            args.scale = std::stoll(require_value("--scale"));
        } else if (key == "--db-vectors") {
            args.db_vectors = std::stoll(require_value("--db-vectors"));
        } else if (key == "--fixed-repetitions") {
            args.fixed_repetitions = std::stoll(require_value("--fixed-repetitions"));
        } else if (key == "--target-seconds") {
            args.target_seconds = std::stod(require_value("--target-seconds"));
        } else if (key == "--threads") {
            args.threads = std::stoi(require_value("--threads"));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value("--seed"));
        } else if (key == "--csv") {
            args.csv_out = require_value("--csv");
        } else if (key == "--json") {
            args.json_out = require_value("--json");
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.dim <= 0) {
        throw std::runtime_error("--dim must be > 0");
    }
    if (args.scale <= 0) {
        throw std::runtime_error("--scale must be > 0");
    }
    if (args.target_seconds <= 0) {
        throw std::runtime_error("--target-seconds must be > 0");
    }
    if (args.fixed_repetitions < 0) {
        throw std::runtime_error("--fixed-repetitions must be >= 0");
    }
    return args;
}

template <typename F>
static Metrics run_with_auto_repetitions(
    F&& kernel,
    long double flop_per_rep,
    long double bytes_per_rep,
    double target_seconds,
    int64_t fixed_repetitions
) {
    Metrics m;
    m.flop_per_rep = flop_per_rep;
    m.bytes_per_rep = bytes_per_rep;

    auto run_once = [&](int64_t r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double sink = kernel(r);
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        return std::make_pair(sec, sink);
    };

    if (fixed_repetitions > 0) {
        auto [sec, sink] = run_once(fixed_repetitions);
        m.elapsed_sec = sec;
        m.repetitions = fixed_repetitions;
        m.sink = sink;
    } else {
        int64_t reps = 1;
        while (true) {
            auto [sec, sink] = run_once(reps);
            if (sec >= target_seconds) {
                m.elapsed_sec = sec;
                m.repetitions = reps;
                m.sink = sink;
                break;
            }
            if (reps > (1LL << 60) / 2) {
                m.elapsed_sec = sec;
                m.repetitions = reps;
                m.sink = sink;
                break;
            }
            reps *= 2;
        }
    }

    m.flop_total = m.flop_per_rep * static_cast<long double>(m.repetitions);
    m.bytes_total = m.bytes_per_rep * static_cast<long double>(m.repetitions);
    m.ai = m.bytes_total > 0.0 ? static_cast<double>(m.flop_total / m.bytes_total) : 0.0;
    m.achieved_gflops = static_cast<double>(m.flop_total / m.elapsed_sec / 1e9L);
    m.achieved_gbs = static_cast<double>(m.bytes_total / m.elapsed_sec / 1e9L);
    return m;
}

static void fill_random(std::vector<float>& x, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x) {
        v = dist(rng);
    }
}

static float dot_product(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float norm_l2(const float* a, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

static int64_t choose_default_db_size(int64_t scale) {
    int64_t lower = 65536;
    int64_t upper = 262144;
    return std::max(lower, std::min(upper, scale * 2));
}

static Metrics bench_gemv(const Args& args) {
    const int k = args.dim;
    const int64_t m = args.scale;

    std::vector<float> x(k);
    std::vector<float> a(static_cast<size_t>(m) * k);
    std::vector<float> y(m);
    std::mt19937 rng(args.seed);
    fill_random(x, rng);
    fill_random(a, rng);

    long double flop_per_rep = static_cast<long double>(2.0L * m * k);
    long double bytes_per_rep = static_cast<long double>(4.0L * (m * k + k + m));

    auto kernel = [&](int64_t reps) -> double {
        double sink = 0.0;
        for (int64_t rep = 0; rep < reps; ++rep) {
            #pragma omp parallel for reduction(+:sink) schedule(static)
            for (int64_t row = 0; row < m; ++row) {
                const float* row_ptr = &a[static_cast<size_t>(row) * k];
                float sum = 0.0f;
                for (int i = 0; i < k; ++i) {
                    sum += row_ptr[i] * x[i];
                }
                y[row] = sum;
                sink += static_cast<double>(sum);
            }
        }
        return sink;
    };

    return run_with_auto_repetitions(
        kernel,
        flop_per_rep,
        bytes_per_rep,
        args.target_seconds,
        args.fixed_repetitions
    );
}

static Metrics bench_cos_db_db(const Args& args) {
    const int d = args.dim;
    const int64_t pairs = args.scale;
    const int64_t db_size = args.db_vectors > 0 ? args.db_vectors : choose_default_db_size(pairs);

    std::vector<float> db(static_cast<size_t>(db_size) * d);
    std::vector<int32_t> idx_a(pairs), idx_b(pairs);
    std::mt19937 rng(args.seed);
    fill_random(db, rng);
    std::uniform_int_distribution<int32_t> uni(0, static_cast<int32_t>(db_size - 1));
    for (int64_t i = 0; i < pairs; ++i) {
        idx_a[i] = uni(rng);
        idx_b[i] = uni(rng);
    }

    long double flop_per_pair = 6.0L * d + 3.0L;
    long double bytes_per_pair = 8.0L * d + 4.0L;
    long double flop_per_rep = flop_per_pair * pairs;
    long double bytes_per_rep = bytes_per_pair * pairs;

    auto kernel = [&](int64_t reps) -> double {
        double sink = 0.0;
        for (int64_t rep = 0; rep < reps; ++rep) {
            #pragma omp parallel for reduction(+:sink) schedule(static)
            for (int64_t i = 0; i < pairs; ++i) {
                const float* a = &db[static_cast<size_t>(idx_a[i]) * d];
                const float* b = &db[static_cast<size_t>(idx_b[i]) * d];
                float ip = 0.0f;
                float na = 0.0f;
                float nb = 0.0f;
                for (int j = 0; j < d; ++j) {
                    ip += a[j] * b[j];
                    na += a[j] * a[j];
                    nb += b[j] * b[j];
                }
                float cosv = ip / (std::sqrt(na) * std::sqrt(nb));
                sink += static_cast<double>(cosv);
            }
        }
        return sink;
    };

    return run_with_auto_repetitions(
        kernel,
        flop_per_rep,
        bytes_per_rep,
        args.target_seconds,
        args.fixed_repetitions
    );
}

static Metrics bench_cos_q_db(const Args& args) {
    const int d = args.dim;
    const int64_t n = args.scale;
    const int64_t db_size = args.db_vectors > 0 ? args.db_vectors : choose_default_db_size(n);

    std::vector<float> db(static_cast<size_t>(db_size) * d);
    std::vector<int32_t> idx(n);
    std::mt19937 rng(args.seed);
    fill_random(db, rng);
    std::uniform_int_distribution<int32_t> uni(0, static_cast<int32_t>(db_size - 1));
    for (int64_t i = 0; i < n; ++i) {
        idx[i] = uni(rng);
    }
    const int32_t q_index = uni(rng);
    const float* q = &db[static_cast<size_t>(q_index) * d];
    const float norm_q = norm_l2(q, d);

    long double flop_per_vec = 4.0L * d + 1.0L;
    long double bytes_per_vec = 4.0L * d + 4.0L;
    long double flop_per_rep = flop_per_vec * n;
    long double bytes_per_rep = bytes_per_vec * n + 4.0L * d;

    auto kernel = [&](int64_t reps) -> double {
        double sink = 0.0;
        for (int64_t rep = 0; rep < reps; ++rep) {
            #pragma omp parallel for reduction(+:sink) schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float* b = &db[static_cast<size_t>(idx[i]) * d];
                float ip = 0.0f;
                float nb = 0.0f;
                for (int j = 0; j < d; ++j) {
                    ip += q[j] * b[j];
                    nb += b[j] * b[j];
                }
                float cosv = ip / (norm_q * std::sqrt(nb));
                sink += static_cast<double>(cosv);
            }
        }
        return sink;
    };

    return run_with_auto_repetitions(
        kernel,
        flop_per_rep,
        bytes_per_rep,
        args.target_seconds,
        args.fixed_repetitions
    );
}

static Metrics bench_ip_q_db(const Args& args) {
    const int d = args.dim;
    const int64_t n = args.scale;
    const int64_t db_size = args.db_vectors > 0 ? args.db_vectors : choose_default_db_size(n);

    std::vector<float> db(static_cast<size_t>(db_size) * d);
    std::vector<int32_t> idx(n);
    std::mt19937 rng(args.seed);
    fill_random(db, rng);
    std::uniform_int_distribution<int32_t> uni(0, static_cast<int32_t>(db_size - 1));
    for (int64_t i = 0; i < n; ++i) {
        idx[i] = uni(rng);
    }
    const int32_t q_index = uni(rng);
    const float* q = &db[static_cast<size_t>(q_index) * d];

    long double flop_per_vec = 2.0L * d;
    long double bytes_per_vec = 4.0L * d + 4.0L;
    long double flop_per_rep = flop_per_vec * n;
    long double bytes_per_rep = bytes_per_vec * n + 4.0L * d;

    auto kernel = [&](int64_t reps) -> double {
        double sink = 0.0;
        for (int64_t rep = 0; rep < reps; ++rep) {
            #pragma omp parallel for reduction(+:sink) schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float* b = &db[static_cast<size_t>(idx[i]) * d];
                float ip = dot_product(q, b, d);
                sink += static_cast<double>(ip);
            }
        }
        return sink;
    };

    return run_with_auto_repetitions(
        kernel,
        flop_per_rep,
        bytes_per_rep,
        args.target_seconds,
        args.fixed_repetitions
    );
}

static Metrics bench_softmax(const Args& args) {
    const int64_t d = args.scale;
    std::vector<float> x(d);
    std::vector<float> out(d);
    std::mt19937 rng(args.seed);
    fill_random(x, rng);

    // Model assumption (as requested): treat softmax as one fused kernel
    // with no repeated intermediate read/write traffic.
    // FLOP model remains algorithm-level approximation.
    long double flop_per_rep = 4.0L * d;
    long double bytes_per_rep = 8.0L * d;

    const int max_threads = omp_get_max_threads();
    std::vector<double> sink_per_thread(max_threads, 0.0);
    std::vector<float> thread_max(max_threads, -std::numeric_limits<float>::infinity());
    std::vector<double> thread_sum(max_threads, 0.0);

    auto kernel = [&](int64_t reps) -> double {
        float m_global = -std::numeric_limits<float>::infinity();
        double l_global = 0.0;
        float inv_l_global = 0.0f;
        int used_threads = 0;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            const int64_t begin = (d * tid) / nth;
            const int64_t end = (d * (tid + 1)) / nth;
            double sink_local = 0.0;

            #pragma omp single
            {
                used_threads = nth;
            }

            for (int64_t rep = 0; rep < reps; ++rep) {
                // High-throughput softmax:
                // pass 1: parallel max reduction.
                // pass 2: parallel exp + sum reduction (write temporary exp to out).
                // pass 3: parallel normalization (reusing out buffer).
                float local_max = -std::numeric_limits<float>::infinity();
                #pragma omp simd reduction(max:local_max)
                for (int64_t i = begin; i < end; ++i) {
                    local_max = std::max(local_max, x[i]);
                }
                thread_max[tid] = local_max;

                #pragma omp barrier
                #pragma omp single
                {
                    m_global = -std::numeric_limits<float>::infinity();
                    for (int t = 0; t < used_threads; ++t) {
                        m_global = std::max(m_global, thread_max[t]);
                    }
                }
                #pragma omp barrier

                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int64_t i = begin; i < end; ++i) {
                    const float e = std::exp(x[i] - m_global);
                    out[i] = e;
                    local_sum += static_cast<double>(e);
                }
                thread_sum[tid] = local_sum;

                #pragma omp barrier
                #pragma omp single
                {
                    l_global = 0.0;
                    for (int t = 0; t < used_threads; ++t) {
                        l_global += thread_sum[t];
                    }
                    inv_l_global = static_cast<float>(1.0 / l_global);
                }
                #pragma omp barrier

                double rep_sink = 0.0;
                #pragma omp simd reduction(+:rep_sink)
                for (int64_t i = begin; i < end; ++i) {
                    const float v = out[i] * inv_l_global;
                    out[i] = v;
                    rep_sink += static_cast<double>(v);
                }
                sink_local += rep_sink;
            }

            sink_per_thread[tid] = sink_local;
        }

        double sink = 0.0;
        for (int t = 0; t < used_threads; ++t) {
            sink += sink_per_thread[t];
        }
        return sink;
    };

    return run_with_auto_repetitions(
        kernel,
        flop_per_rep,
        bytes_per_rep,
        args.target_seconds,
        args.fixed_repetitions
    );
}

static void maybe_append_csv(const Args& args, const Metrics& m) {
    if (args.csv_out.empty()) {
        return;
    }
    const bool write_header = !std::ifstream(args.csv_out).good();
    std::ofstream out(args.csv_out, std::ios::app);
    if (!out) {
        throw std::runtime_error("Cannot open CSV output: " + args.csv_out);
    }
    if (write_header) {
        out << "primitive,dim,scale,db_vectors,repetitions,elapsed_sec,flop_total,byte_total,ai,achieved_gflops,achieved_gbs,target_seconds,threads\n";
    }
    out << args.primitive << ","
        << args.dim << ","
        << args.scale << ","
        << args.db_vectors << ","
        << m.repetitions << ","
        << m.elapsed_sec << ","
        << static_cast<double>(m.flop_total) << ","
        << static_cast<double>(m.bytes_total) << ","
        << m.ai << ","
        << m.achieved_gflops << ","
        << m.achieved_gbs << ","
        << args.target_seconds << ","
        << (args.threads > 0 ? args.threads : omp_get_max_threads())
        << "\n";
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += ' ';
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

static void maybe_write_json(const Args& args, const Metrics& m) {
    if (args.json_out.empty()) {
        return;
    }
    std::ofstream out(args.json_out, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Cannot open JSON output: " + args.json_out);
    }
    const int runtime_threads = (args.threads > 0 ? args.threads : omp_get_max_threads());
    out << "{\n";
    out << "  \"primitive\": \"" << json_escape(args.primitive) << "\",\n";
    out << "  \"dim\": " << args.dim << ",\n";
    out << "  \"scale\": " << args.scale << ",\n";
    out << "  \"db_vectors\": " << args.db_vectors << ",\n";
    out << "  \"target_seconds\": " << args.target_seconds << ",\n";
    out << "  \"threads\": " << runtime_threads << ",\n";
    out << "  \"seed\": " << args.seed << ",\n";
    out << "  \"repetitions\": " << m.repetitions << ",\n";
    out << "  \"elapsed_sec\": " << m.elapsed_sec << ",\n";
    out << "  \"flop_total\": " << static_cast<double>(m.flop_total) << ",\n";
    out << "  \"byte_total\": " << static_cast<double>(m.bytes_total) << ",\n";
    out << "  \"ai\": " << m.ai << ",\n";
    out << "  \"achieved_gflops\": " << m.achieved_gflops << ",\n";
    out << "  \"achieved_gbs\": " << m.achieved_gbs << ",\n";
    out << "  \"sink\": " << m.sink << "\n";
    out << "}\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.threads > 0) {
            omp_set_num_threads(args.threads);
        }

        Metrics m;
        if (args.primitive == "gemv") {
            m = bench_gemv(args);
        } else if (args.primitive == "cos_db_db") {
            m = bench_cos_db_db(args);
        } else if (args.primitive == "cos_q_db") {
            m = bench_cos_q_db(args);
        } else if (args.primitive == "ip_q_db") {
            m = bench_ip_q_db(args);
        } else if (args.primitive == "softmax") {
            m = bench_softmax(args);
        } else {
            throw std::runtime_error("Unsupported primitive: " + args.primitive);
        }

        maybe_append_csv(args, m);
        maybe_write_json(args, m);

        std::cout << "primitive: " << args.primitive << "\n";
        std::cout << "dim: " << args.dim << "\n";
        std::cout << "scale: " << args.scale << "\n";
        std::cout << "repetitions: " << m.repetitions << "\n";
        std::cout << "elapsed_sec: " << m.elapsed_sec << "\n";
        std::cout << "ai: " << m.ai << "\n";
        std::cout << "achieved_gflops: " << m.achieved_gflops << "\n";
        std::cout << "achieved_gbs: " << m.achieved_gbs << "\n";
        std::cout << "threads: " << (args.threads > 0 ? args.threads : omp_get_max_threads()) << "\n";
        std::cout << "sink: " << m.sink << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
