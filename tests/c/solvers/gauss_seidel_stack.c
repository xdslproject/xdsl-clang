static void run(void) {
  static float data[256 * 256];
  const int n = 256;

  for (int k = 0; k < 1000; k++) {
    for (int i = 1; i < n - 1; i++) {
      for (int j = 1; j < n - 1; j++) {
        data[i * n + j] = (data[(i - 1) * n + j] + data[(i + 1) * n + j] +
                           data[i * n + (j - 1)] + data[i * n + (j + 1)]) *
                          0.25f;
      }
    }
  }
}

int main(void) {
  run();
  return 0;
}
