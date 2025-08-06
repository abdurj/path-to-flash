#include "attention_base.hpp"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace p2f {

AttentionBase::AttentionBase(int seq_len, int d_model, int num_heads)
    : seq_len_(seq_len), d_model_(d_model), num_heads_(num_heads),
      head_dim_(d_model / num_heads) {
    // Constructor is now protected and validation is done in Create()
}

absl::StatusOr<std::unique_ptr<AttentionBase>> AttentionBase::Create(
    int seq_len, int d_model, int num_heads) {
    
    // Validate constructor parameters
    if (seq_len <= 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("seq_len must be positive, got %d", seq_len));
    }
    
    if (d_model <= 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("d_model must be positive, got %d", d_model));
    }
    
    if (num_heads <= 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("num_heads must be positive, got %d", num_heads));
    }
    
    if (d_model % num_heads != 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("d_model (%d) must be divisible by num_heads (%d)", 
                           d_model, num_heads));
    }
    
    // This is a base class, so we can't actually create it
    // Derived classes should override this method
    return absl::UnimplementedError("AttentionBase::Create() must be overridden by derived class");
}

absl::Status AttentionBase::ValidateInput(const std::vector<float>& input) const {
    const size_t expected_size = static_cast<size_t>(seq_len_ * d_model_);
    
    if (input.size() != expected_size) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Input size mismatch: expected %zu (%dx%d), got %zu", 
                           expected_size, seq_len_, d_model_, input.size()));
    }
    
    return absl::OkStatus();
}

} // namespace p2f
