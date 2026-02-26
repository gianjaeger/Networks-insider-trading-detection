// match_making.cpp
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <cctype>
#include <limits>
#include <cmath>

using namespace std;

// ------------ Parameters ------------
static const int HZ_DEFAULT = 8;       // strictly >5 trades
static const double HM_DEFAULT = 0.80; // similarity threshold
static const bool ACTIVITY_WEIGHTED = true; // if false, uses equal-weight combo when possible
static const int MIN_OVERLAP_WEEKS = 4;     // NEW: require >=2 distinct overlapping weeks

// ------------ small string helpers ------------
static inline void rstrip_crlf(string& s){
    while(!s.empty() && (s.back()=='\r' || s.back()=='\n')) s.pop_back();
}
static inline void trim_spaces(string& s){
    size_t i=0, j=s.size();
    while(i<j && isspace(static_cast<unsigned char>(s[i]))) ++i;
    while(j>i && isspace(static_cast<unsigned char>(s[j-1]))) --j;
    if (i>0 || j<s.size()) s = s.substr(i, j-i);
}

// parse last 3 commas: [...company...],[insider],[action],[date]
static inline bool parse_line_from_end(const string& line,
                                       string& company, string& insider,
                                       string& action, string& date_raw)
{
    if (line.empty()) return false;
    size_t c3 = line.rfind(',');
    if (c3 == string::npos) return false;
    size_t c2 = line.rfind(',', c3-1);
    if (c2 == string::npos) return false;
    size_t c1 = line.rfind(',', c2-1);
    if (c1 == string::npos) return false;

    company  = line.substr(0, c1);
    insider  = line.substr(c1+1, c2-c1-1);
    action   = line.substr(c2+1, c3-c2-1);
    date_raw = line.substr(c3+1);

    // clean pieces
    string parts[4] = {company, insider, action, date_raw};
    for (int i=0;i<4;++i){ rstrip_crlf(parts[i]); trim_spaces(parts[i]); }
    company = parts[0]; insider = parts[1]; action = parts[2]; date_raw = parts[3];
    return !(company.empty() || insider.empty() || action.empty() || date_raw.empty());
}

// ------------ Date utils (days since 1970-01-01) ------------
long long days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<long long>(era) * 146097LL + static_cast<long long>(doe) - 719468LL;
}

inline bool parse_date_yyyy_mm_dd(const string& s, int& out_days){
    if (s.size() < 10) return false;
    // take first 10 chars
    string d10 = s.substr(0,10);
    if (!(isdigit(d10[0])&&isdigit(d10[1])&&isdigit(d10[2])&&isdigit(d10[3])&&
          d10[4]=='-' && isdigit(d10[5])&&isdigit(d10[6]) && d10[7]=='-' &&
          isdigit(d10[8])&&isdigit(d10[9]))) return false;
    int y = stoi(d10.substr(0,4));
    int m = stoi(d10.substr(5,2));
    int d = stoi(d10.substr(8,2));
    out_days = static_cast<int>(days_from_civil(y,(unsigned)m,(unsigned)d));
    return true;
}

// ------------ Weekly kernel w(d) ------------
inline double time_weight(int gap_days) {
    if (gap_days < 0) gap_days = -gap_days;
    if (gap_days > 7) return 0.0;
    return 1.0 - (static_cast<double>(gap_days) / 7.0);
}

// ------------ Hungarian (Munkres) for square cost matrix ------------
// We solve a *minimization* on a square matrix. To do maximum-weight matching
// on weights W in [0,1], we pass costs C = -W (or equivalently a large
// constant minus W). Padding/dummy rows/cols with cost 0 lets the solver
// freely assign unmatched events (weight 0).
//
// Returns assignment as a vector `assign_col_of_row` of size N, where
// assign_col_of_row[r] = c chosen for row r (0..N-1).
static vector<int> hungarian_min_cost(const vector<vector<double>>& cost) {
    const int N = static_cast<int>(cost.size());
    vector<double> u(N+1, 0.0), v(N+1, 0.0);
    vector<int> p(N+1, 0), way(N+1, 0);

    for (int i = 1; i <= N; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(N+1, numeric_limits<double>::infinity());
        vector<char> used(N+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = numeric_limits<double>::infinity();
            for (int j = 1; j <= N; ++j) if (!used[j]) {
                double cur = cost[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= N; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    vector<int> assign_col_of_row(N, -1);
    for (int j = 1; j <= N; ++j) {
        if (p[j] > 0) assign_col_of_row[p[j]-1] = j-1;
    }
    return assign_col_of_row;
}

// Build square *cost* matrix for Hungarian from two sorted day lists X and Y.
// Rows correspond to X events; columns to Y events; matrix padded to N=max(m,n).
// For eligible pairs (|x-y|<=7), weight=w(d) and cost=-weight; otherwise cost=0
// (which equals weight 0), allowing matching to dummy columns/rows.
static vector<vector<double>> build_cost_matrix_max_weight(const vector<int>& X,
                                                           const vector<int>& Y)
{
    const int m = static_cast<int>(X.size());
    const int n = static_cast<int>(Y.size());
    const int N = max(m, n);
    vector<vector<double>> C(N, vector<double>(N, 0.0)); // defaults to cost 0 (weight 0)

    // Fill real pairs
    for (int i = 0; i < m; ++i) {
        // For efficiency: restrict j to Y indices with |x - y| <= 7 using two-pointer window
        int x = X[i];
        // lower bound for y >= x-7
        auto lb = lower_bound(Y.begin(), Y.end(), x - 7);
        // iterate until y > x+7
        for (auto it = lb; it != Y.end() && *it <= x + 7; ++it) {
            int j = static_cast<int>(it - Y.begin());
            double w = time_weight(*it - x); // already handles abs and >7
            if (w > 0.0) {
                C[i][j] = -w; // minimize negative weight == maximize weight
            } else {
                // leave as 0
            }
        }
    }
    // Remaining cells (including dummies) already 0 => weight 0
    return C;
}

// ------------ Assignment-based similarity per category ------------
static double symmetric_assignment_similarity(const vector<int>& X, const vector<int>& Y) {
    const int m = static_cast<int>(X.size());
    const int n = static_cast<int>(Y.size());
    if (m == 0 || n == 0) return 0.0;

    // Build cost matrix and run Hungarian
    vector<vector<double>> C = build_cost_matrix_max_weight(X, Y);
    const int N = static_cast<int>(C.size());
    vector<int> assign = hungarian_min_cost(C); // size N

    // Recover total matched weight W* from real X<->Y matches (ignore dummies)
    double total_weight = 0.0;
    for (int i = 0; i < N; ++i) {
        int j = assign[i];
        if (i < m && j >= 0 && j < n) {
            double cost = C[i][j];
            // If this was an eligible edge, cost is negative of weight.
            if (cost < 0.0) total_weight += -cost;
            // If cost == 0.0, this is either dummy or ineligible => contributes 0.
        }
    }

    // Directional scores use same M*:
    // s_{X|Y} = W*/|X|, s_{Y|X} = W*/|Y|
    double sXY = total_weight / static_cast<double>(m);
    double sYX = total_weight / static_cast<double>(n);
    return 0.5 * (sXY + sYX);
}

// ------------ Combining categories ------------
double combined_similarity_assignment(
    const vector<int>& XA, const vector<int>& YA,
    const vector<int>& XD, const vector<int>& YD
) {
    double SA = (!XA.empty() && !YA.empty()) ? symmetric_assignment_similarity(XA, YA) : 0.0;
    double SD = (!XD.empty() && !YD.empty()) ? symmetric_assignment_similarity(XD, YD) : 0.0;

    const double TA = static_cast<double>(XA.size() + YA.size());
    const double TD = static_cast<double>(XD.size() + YD.size());
    const double T  = TA + TD;
    if (T <= 0.0) return 0.0;

    if (ACTIVITY_WEIGHTED) {
        // Activity-weighted (primary)
        return (TA / T) * SA + (TD / T) * SD;
    } else {
        // Equal-weight companion: only if both categories exist for both insiders
        if (!XA.empty() && !YA.empty() && !XD.empty() && !YD.empty())
            return 0.5 * (SA + SD);
        if (!XA.empty() && !YA.empty()) return SA;
        if (!XD.empty() && !YD.empty()) return SD;
        return 0.0;
    }
}

// ------------ Containers ------------
using InsiderDays = unordered_map<string, vector<int>>; // insider -> days
using CompanyDir  = unordered_map<string, InsiderDays>; // company -> insider->days
static const vector<int> EMPTY_VEC;

inline const vector<int>& get_or_empty(const InsiderDays& m, const string& k) {
    auto it = m.find(k);
    return (it == m.end()) ? EMPTY_VEC : it->second;
}

void dedup_and_sort(CompanyDir& mp) {
    for (auto& [symbol, per_insider] : mp) {
        for (auto& [insider, days] : per_insider) {
            sort(days.begin(), days.end());
            days.erase(unique(days.begin(), days.end()), days.end());
        }
    }
}

// ------------ NEW: count overlapping calendar-week bins ------------
// Weeks are coarse bins using floor(days_since_epoch / 7). We require that two
// insiders both have at least one trade (buy or sell) in the same week bin,
// in at least MIN_OVERLAP_WEEKS distinct weeks.
static int count_overlapping_weeks(const vector<int>& buy1, const vector<int>& sell1,
                                   const vector<int>& buy2, const vector<int>& sell2)
{
    unordered_set<long long> w1;
    w1.reserve(buy1.size() + sell1.size());
    for (int d : buy1)  w1.insert(d / 7);
    for (int d : sell1) w1.insert(d / 7);

    unordered_set<long long> w2;
    w2.reserve(buy2.size() + sell2.size());
    for (int d : buy2)  w2.insert(d / 7);
    for (int d : sell2) w2.insert(d / 7);

    int cnt = 0;
    // Iterate smaller set for efficiency
    if (w1.size() > w2.size()) {
        for (const auto& w : w2) if (w1.count(w)) ++cnt;
    } else {
        for (const auto& w : w1) if (w2.count(w)) ++cnt;
    }
    return cnt;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // CLI: input [output] [HM] [HZ]
    string in_file  = (argc >= 2 ? argv[1] : "trades_by_day.csv");
    string out_file = (argc >= 3 ? argv[2] : "match-making-edges.csv");
    double HM = (argc >= 4 ? atof(argv[3]) : HM_DEFAULT);
    int HZ    = (argc >= 5 ? atoi(argv[4]) : HZ_DEFAULT);

    ifstream file(in_file);
    if (!file) {
        cerr << "Error: cannot open input file: " << in_file << "\n";
        return 1;
    }

    CompanyDir buys, sells;

    string header;
    getline(file, header); // skip header

    string line;
    long long bad_rows = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;
        rstrip_crlf(line);

        string symbol, insider, action, date_raw;
        if (!parse_line_from_end(line, symbol, insider, action, date_raw)) { ++bad_rows; continue; }

        // Expect 'A' or 'D' (case-insensitive)
        if (action.empty()) { ++bad_rows; continue; }
        char act = toupper(static_cast<unsigned char>(action[0]));
        if (act != 'A' && act != 'D') { ++bad_rows; continue; }

        int day;
        if (!parse_date_yyyy_mm_dd(date_raw, day)) { ++bad_rows; continue; }

        if (act == 'A') buys[symbol][insider].push_back(day);
        else            sells[symbol][insider].push_back(day);
    }

    dedup_and_sort(buys);
    dedup_and_sort(sells);

    ofstream out(out_file);
    if (!out) {
        cerr << "Error: cannot open output file: " << out_file << "\n";
        return 1;
    }
    out << "source,target,company,similarity\n";

    unordered_set<string> symbols;
    symbols.reserve(buys.size() + sells.size());
    for (const auto& kv : buys)  symbols.insert(kv.first);
    for (const auto& kv : sells) symbols.insert(kv.first);

    set<string> all_nodes;
    long long edge_count = 0;

    for (const auto& symbol : symbols) {
        const InsiderDays* buy_map  = (buys.count(symbol)  ? &buys.at(symbol)  : nullptr);
        const InsiderDays* sell_map = (sells.count(symbol) ? &sells.at(symbol) : nullptr);

        vector<string> insiders;
        if (buy_map)  for (const auto& [name,_] : *buy_map)  insiders.push_back(name);
        if (sell_map) for (const auto& [name,_] : *sell_map) insiders.push_back(name);
        sort(insiders.begin(), insiders.end());
        insiders.erase(unique(insiders.begin(), insiders.end()), insiders.end());

        for (size_t i = 0; i < insiders.size(); ++i) {
            const string& i1 = insiders[i];
            const vector<int>& buy1  = buy_map  ? get_or_empty(*buy_map,  i1) : EMPTY_VEC;
            const vector<int>& sell1 = sell_map ? get_or_empty(*sell_map, i1) : EMPTY_VEC;
            size_t count1 = buy1.size() + sell1.size();
            if (count1 < static_cast<size_t>(HZ)) continue;

            for (size_t j = i + 1; j < insiders.size(); ++j) {
                const string& i2 = insiders[j];
                const vector<int>& buy2  = buy_map  ? get_or_empty(*buy_map,  i2) : EMPTY_VEC;
                const vector<int>& sell2 = sell_map ? get_or_empty(*sell_map, i2) : EMPTY_VEC;
                size_t count2 = buy2.size() + sell2.size();
                if (count2 < static_cast<size_t>(HZ)) continue;

                // NEW: require at least MIN_OVERLAP_WEEKS distinct overlapping weeks (any action)
                int overlap_weeks = count_overlapping_weeks(buy1, sell1, buy2, sell2);
                if (overlap_weeks < MIN_OVERLAP_WEEKS) continue;

                // Assignment-based combined similarity (AW primary, UW optional)
                double sim = combined_similarity_assignment(buy1, buy2, sell1, sell2);
                if (sim >= HM) {
                    out << i1 << "," << i2 << "," << symbol << "," << sim << "\n";
                    all_nodes.insert(i1);
                    all_nodes.insert(i2);
                    ++edge_count;
                }
            }
        }
    }

    cerr << "Skipped malformed rows: " << bad_rows << "\n";
    cerr << "Similarity threshold (HM): " << HM << "\n";
    cerr << "Minimum trades per insider (HZ): " << HZ << "\n";
    cerr << "Required overlapping weeks: " << MIN_OVERLAP_WEEKS << "\n";
    cerr << "Nodes (in edges): " << all_nodes.size() << "\n";
    cerr << "Edges: " << edge_count << "\n";
    cerr << "Written: " << out_file << "\n";
    return 0;
}