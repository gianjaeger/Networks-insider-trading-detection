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

using namespace std;

// ------------ Parameters ------------
static const int HZ_DEFAULT = 8;       // strictly >7 trades
static const double HM_DEFAULT = 0.80; // similarity threshold
static const bool ACTIVITY_WEIGHTED = true;

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

// ------------ Kernel ------------
inline double time_weight(int gap_days) {
    if (gap_days < 0) gap_days = -gap_days;
    if (gap_days > 7) return 0.0;
    return 1.0 - (static_cast<double>(gap_days) / 7.0);
}

// ------------ Best-match similarity ------------
double directional_best_match(const vector<int>& X, const vector<int>& Y) {
    if (X.empty() || Y.empty()) return 0.0;
    double sum = 0.0;
    for (int x : X) {
        auto it = lower_bound(Y.begin(), Y.end(), x);
        double best = 0.0;
        if (it != Y.end()) {
            int d = *it - x;
            if (d <= 7) best = max(best, time_weight(d));
        }
        if (it != Y.begin()) {
            int d = x - *prev(it);
            if (d <= 7) best = max(best, time_weight(d));
        }
        sum += best;
    }
    return sum / static_cast<double>(X.size());
}

double symmetric_best_match(const vector<int>& X, const vector<int>& Y) {
    double sXY = directional_best_match(X, Y);
    double sYX = directional_best_match(Y, X);
    return 0.5 * (sXY + sYX);
}

double combined_similarity_best_match(
    const vector<int>& XA, const vector<int>& YA,
    const vector<int>& XD, const vector<int>& YD
) {
    double SA = (!XA.empty() && !YA.empty()) ? symmetric_best_match(XA, YA) : 0.0;
    double SD = (!XD.empty() && !YD.empty()) ? symmetric_best_match(XD, YD) : 0.0;

    const double TA = static_cast<double>(XA.size() + YA.size());
    const double TD = static_cast<double>(XD.size() + YD.size());
    const double T  = TA + TD;
    if (T <= 0.0) return 0.0;

    if (ACTIVITY_WEIGHTED) {
        return (TA / T) * SA + (TD / T) * SD;
    } else {
        if (!XA.empty() && !YA.empty() && !XD.empty() && !YD.empty()) return 0.5 * (SA + SD);
        if (!XA.empty() && !YA.empty()) return SA;
        if (!XD.empty() && !YD.empty()) return SD;
        return 0.0;
    }
}

// ------------ Overlap by distinct weeks (NEW) ------------
// Counts how many DISTINCT week bins (size 7 days, aligned to epoch) contain
// at least one pair of A/B trades within <= 7 days of each other.
int count_overlapping_weeks(const vector<int>& A, const vector<int>& B) {
    if (A.empty() || B.empty()) return 0;
    unordered_set<int> weeks;
    size_t i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        int da = A[i], db = B[j];
        int diff = da - db;
        int adiff = diff >= 0 ? diff : -diff;

        if (adiff <= 7) {
            // Record a single canonical week bin for this overlap
            int week = (min(da, db)) / 7; // epoch-aligned week index
            weeks.insert(week);

            // Advance the pointer(s) tied to the earlier date to explore new overlaps
            if (da == db) { ++i; ++j; }
            else if (da < db) { ++i; }
            else { ++j; }
        } else {
            // Move the earlier pointer forward to try to find a closer match
            if (da < db) ++i; else ++j;
        }
    }
    return static_cast<int>(weeks.size());
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

                // Compute similarity (unchanged)
                double sim = combined_similarity_best_match(buy1, buy2, sell1, sell2);

                // NEW: require at least 2 distinct overlapping weeks across buys and sells
                int overlap_weeks_buys  = count_overlapping_weeks(buy1,  buy2);
                int overlap_weeks_sells = count_overlapping_weeks(sell1, sell2);
                int total_overlap_weeks = overlap_weeks_buys + overlap_weeks_sells;

                if (sim >= HM && total_overlap_weeks >= 4) {
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
    cerr << "Nodes (in edges): " << all_nodes.size() << "\n";
    cerr << "Edges: " << edge_count << "\n";
    cerr << "Written: " << out_file << "\n";
    return 0;
}