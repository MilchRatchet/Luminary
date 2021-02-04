void test() {
  float* d_val;
  cudaMalloc((void**)&d_val, 8 * sizeof(float));
  cudaFree(d_val);
}
