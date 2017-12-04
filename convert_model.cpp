#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_map>
#include <math.h>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << argv[0] << " input_model output_binary" << endl;
    return 0;
  }

  float b = 0;
  std::unordered_map<int32_t, float> w;
  // read
  ifstream fin;
  fin.open(argv[1], ios_base::in);
  size_t num = 0;
  fin >> num;
  int32_t idx = 0;
  float weight = 0;
  for (size_t i = 0; i < num; ++i) {
    fin >> idx >> weight; 
    if (!fin || fin.eof()) {
      cerr << "read model error occurs" << endl;
      return -1;
    }
    if (fabs(weight) > 1e-7) {
      w[idx] = weight;
      cout << idx << ":" << weight << endl;
    }
  }
  fin.close();
  // write
  FILE* fp = fopen(argv[2], "wb");
  if (!fp) {
    cerr << "cannot open file for write:" << argv[2] << endl;
    return -1;
  }
  fwrite(&b, sizeof(b), 1, fp);
  int32_t tot = w.size();
  fwrite(&tot, sizeof(tot), 1, fp);
  for (const auto& kv : w) {
    fwrite(&kv.first, sizeof(kv.first), 1, fp);
    fwrite(&kv.second, sizeof(kv.second),1,fp);
  }
  fclose(fp);
  cout << "validate feature size:" << tot << endl;
  return 0;
}
