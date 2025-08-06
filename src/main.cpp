int main() {
    int seq_len = 512;
    int d_model = 768;
    int num_heads = 12;

    Attention attention(seq_len, d_model, num_heads);

    auto input = MatrixUtils::random_matrix<float>({seq_len, d_model});

    auto output = attention.forward(input);

    attention.benchmark(/*iterations=*/100);
}