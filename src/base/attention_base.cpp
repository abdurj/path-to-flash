#include "attention_base.hpp"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace ptflash {

Attention::Attention(int seq_len, int d_model, int num_heads)
    : seq_len_(seq_len), d_model_(d_model), num_heads_(num_heads),
      head_dim_(d_model / num_heads) {
    // Constructor is now protected and validation is done in CreateValidated()
}

absl::Status Attention::ValidateInput(const std::vector<float>& input) const {
    const size_t expected_size = static_cast<size_t>(seq_len_ * d_model_);
    
    if (input.size() != expected_size) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Input size mismatch: expected %zu (%dx%d), got %zu", 
                           expected_size, seq_len_, d_model_, input.size()));
    }
    
    return absl::OkStatus();
}

} // namespace ptflash
