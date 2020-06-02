// Dmitry _kun_ Sayutin (2019)

#include <bits/stdc++.h>
// #include <unistd.h>

using std::cin;
using std::cout;
using std::cerr;

using std::vector;
using std::map;
using std::array;
using std::set;
using std::string;
using std::queue;

using std::pair;
using std::make_pair;

using std::tuple;
using std::make_tuple;
using std::get;

using std::min;
using std::abs;
using std::max;
using std::swap;

using std::unique;
using std::sort;
using std::generate;
using std::reverse;
using std::min_element;
using std::max_element;

#define SZ(vec)         int((vec).size())
#define ALL(data)       data.begin(),data.end()
#define RALL(data)      data.rbegin(),data.rend()
#define TYPEMAX(type)   std::numeric_limits<type>::max()
#define TYPEMIN(type)   std::numeric_limits<type>::min()

#define ensure(cond) if (not (cond)) {fprintf(stderr, "Failed: %s", #cond); exit(1);}

struct instance_t {
    int n;
    int m;
    vector<vector<int>> graph;    
};

struct result_t {
    int depth;
    int root_id;
    vector<int> tree_parents;
};

instance_t parse(std::istream& in) {
    string s;
    int n = -1, m = -1;
    
    vector<vector<int>> adj;
    
    while (std::getline(in, s)) {
        if (s.empty() or s[0] == 'c')
            continue;

        if (s[0] == 'p') {
            ensure(s.substr(0, 6) == "p tdp ");
            s = s.substr(6, 100000);

            const char* ptr = s.data();
            n = strtoll(ptr, const_cast<char**>(&ptr), 10);
            m = strtoll(ptr, const_cast<char**>(&ptr), 10);
            
            adj.resize(n);
        } else {
            const char* ptr = s.data();
            
            int a = strtoll(ptr, const_cast<char**>(&ptr), 10) - 1;
            int b = strtoll(ptr, const_cast<char**>(&ptr), 10) - 1;
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }

    return instance_t {n, m, adj};
}

// instance_t parse_args(char** argv) {
//     int n = atoi(argv[0]);
//     int m = 0;
//     vector<vector<int>> adj(n);
    
//     for (++argv; *argv != 0; ++argv) {
//         int v = atoi(*argv);
//         int u = atoi(strchr(*argv, '_') + 1);

//         adj[v].push_back(u);
//         adj[u].push_back(v);
//         ++m;
//     }

//     return instance_t {n, m, adj};
// }

uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

struct hashmap_elem {
    uint64_t key;
    uint64_t dpval;
    //uint64_t store;
};

class hashmap_t {
private:
    const size_t size;
    size_t used;
    
    std::unique_ptr<hashmap_elem[]> data;

    /*
      special assumptions:
      1) key == 0 -> cell empty
    */
    
public:
    hashmap_t(size_t size): size(size) {
        used = 0;
        data = std::make_unique<hashmap_elem[]>(size);
        memset(data.get(), 0, size * sizeof(hashmap_elem));
    }

    hashmap_elem& operator[](uint64_t key) {
        size_t ptr = hash(key) % size;

        while (data[ptr].key and data[ptr].key != key)
            if (++ptr == size)
                ptr = 0;

        if (not data[ptr].key and ++used / 3 >= size / 4) {
            used = 1;
            memset(data.get(), 0, size * sizeof(hashmap_elem));
            ptr = hash(key) % size;
            fprintf(stderr, "wiped\n");
        }

        data[ptr].key = key;
        return data[ptr];
    };
};

result_t solve(instance_t& instance) {
#ifdef LOCAL
    hashmap_t hashmap(int(1e5));
#else
    hashmap_t hashmap(int(2e8) + 33);
#endif

    vector<uint64_t> bitgraph(instance.n);
    for (int v = 0; v < instance.n; ++v)
        for (int u: instance.graph[v])
            bitgraph[v] |= uint64_t(1) << u;

    // auto get_reasonable_selections = [&](uint64_t mask) {
    //     if (not mask)
    //         return (uint64_t)0;
        
    //     auto mask_size = __builtin_popcountll(mask);
        
    //     for (auto tmp = mask; tmp;) {
    //         int bit = __builtin_ctzll(tmp);
    //         tmp ^= (uint64_t(1) << bit);

    //         if (__builtin_popcountll(bitgraph[bit] & mask) == mask_size - 1)
    //             return (uint64_t(1) << bit);
    //     }
        
    //     uint64_t rs = 0;
    //     for (auto tmp = mask; tmp;) {
    //         int bit = __builtin_ctzll(tmp);
    //         tmp ^= (uint64_t(1) << bit);
            
    //         if (__builtin_popcountll(bitgraph[bit] & mask) != 1)
    //             rs |= (uint64_t(1) << bit);
    //     }

    //     return rs;
    // };
    
    std::function<uint64_t(uint64_t)> solve_for_mask, solve_for_mask_internal;

    solve_for_mask_internal = [&](uint64_t mask) {
        auto& entry = hashmap[mask];
        if (entry.dpval)
            return entry.dpval;
        
        uint64_t result = TYPEMAX(uint64_t);
        auto mask_size = __builtin_popcountll(mask);
        
        for (auto tmp = mask; tmp;) {
            int bit = __builtin_ctzll(tmp);
            tmp ^= (uint64_t(1) << bit);

            if (__builtin_popcountll(bitgraph[bit] & mask) == mask_size - 1) {
                return 1 + solve_for_mask(mask ^ (uint64_t(1) << bit));
            }
        }
        
        for (auto tmp = mask; tmp;) {
            int bit = __builtin_ctzll(tmp);
            tmp ^= (uint64_t(1) << bit);

            if (__builtin_popcountll(bitgraph[bit] & mask) > 1)
                result = min(result, 1 + solve_for_mask(mask ^ (uint64_t(1) << bit)));
        }

        entry = hashmap[mask];
        return (entry.dpval = result);
    };

    solve_for_mask = [&](uint64_t mask) {
        if (mask == 0)
            return (uint64_t)0;
        
        uint64_t visited = 0;
        uint64_t result = 0;
        while (visited != mask) {
            int bit = __builtin_ctzll(visited ^ mask);
            visited ^= (uint64_t(1) << bit);

            uint64_t comp = 0;
            queue<int> qu;
            qu.push(bit);

            while (not qu.empty()) {
                int v = qu.front();
                qu.pop();
                comp |= (uint64_t(1) << v);

                for (int u: instance.graph[v])
                    if (mask & ~visited & (uint64_t(1) << u)) {
                        visited |= (uint64_t(1) << u);
                        qu.push(u);
                    }
            }

            result = max(result, solve_for_mask_internal(comp));
        }

        return result;
    };

    uint64_t all = 0;
    for (int i = 0; i < instance.n; ++i)
        all |= (uint64_t(1) << i);
    
    result_t result;
    result.tree_parents.resize(instance.n);
    result.depth = solve_for_mask(all);

    std::function<void(uint64_t, int, int)> recover, recover_internal;

    recover_internal = [&](uint64_t mask, int p, int allowed_depth) {
        int choice = -1;
        auto mask_size = __builtin_popcountll(mask);

        for (auto tmp = mask; tmp;) {
            int bit = __builtin_ctzll(tmp);
            tmp ^= (uint64_t(1) << bit);

            if (__builtin_popcountll(bitgraph[bit] & mask) == mask_size - 1) {
                choice = bit;
                break;
            }
        }

        if (choice == -1)
            for (auto tmp = mask; tmp;) {
                int bit = __builtin_ctzll(tmp);
                tmp ^= (uint64_t(1) << bit);

                if (__builtin_popcountll(bitgraph[bit] & mask) > 1)
                    if (1 + (int)solve_for_mask(mask ^ (uint64_t(1) << bit)) <= allowed_depth) {
                        choice = bit;
                        break;
                    }
            }
        
        assert(choice != -1);
        
        result.tree_parents[choice] = p;
        recover(mask ^ (uint64_t(1) << choice), choice, allowed_depth - 1);
        return;
    };
    
    recover = [&](uint64_t mask, int p, int allowed_depth) {
        if (not mask)
            return;

        uint64_t visited = 0;
        while (visited != mask) {
            int bit = __builtin_ctzll(visited ^ mask);
            visited ^= (uint64_t(1) << bit);

            uint64_t comp = 0;
            queue<int> qu;
            qu.push(bit);

            while (not qu.empty()) {
                int v = qu.front();
                qu.pop();
                comp |= (uint64_t(1) << v);

                for (int u: instance.graph[v])
                    if (mask & ~visited & (uint64_t(1) << u)) {
                        visited |= (uint64_t(1) << u);
                        qu.push(u);
                    }
            }

            recover_internal(comp, p, allowed_depth);
        }
    };

    fprintf(stderr, "recover started\n");
    recover(all, -1, result.depth);
    result.root_id = std::find(ALL(result.tree_parents), -1) - result.tree_parents.begin();

    return result;
}

void print_result(std::ostream& os, instance_t& instance, result_t& result) {
    os << result.depth << "\n";
    for (int i = 0; i < instance.n; ++i)
        os << (i == result.root_id ? 0 : result.tree_parents[i] + 1) << "\n";
}

extern "C" {
    void python_enter_point(int n, int m, int* edges, int* vc) {
        instance_t instance;
        instance.n = n;
        instance.m = m;

        instance.graph.resize(n);
    
        for (int i = 0; i < m; ++i) {
            int v = edges[i] / n;
            int u = edges[i] % n;

            instance.graph[v].push_back(u);
            instance.graph[u].push_back(v);
        }

        auto result = solve(instance);
        print_result(cout, instance, result);

        exit(0);
    }
}

#ifdef AS_MAIN
int main() {
    std::iostream::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    // code here
    instance_t instance = parse(cin);
    
    if (instance.n > 64) {
        assert(false);
    }
    
    auto result = solve(instance);
    print_result(cout, instance, result);
    
    return 0;
}
#endif
